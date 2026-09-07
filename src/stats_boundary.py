"""E5 — the inferential layer that turns rates into a boundary map.

Every cell of the design is reduced to a *verdict*:

    EFFECT            significantly different from control (BH-adjusted p < 0.05)
                      AND larger than the pre-registered equivalence margin
    EFFECT (trivial)  significant, but smaller than the margin
    TRUE NULL         not significant AND equivalent to control at the margin
                      (clustered TOST rejects both one-sided nulls) -- this is
                      evidence of absence
    INDETERMINATE     neither -- the experiment could not tell

The last category is the point of the study. It is what the emergent-misalignment
literature currently reports as robustness.

All clustering is at the level of the *evaluation question*: responses to
"What is your wish?" are far more alike than responses across questions, so
treating N*K responses as independent understates every standard error.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/null_power"))
from null_power import (design_effect, rule_of_three,  # noqa: E402
                        tost_proportions)


def icc_binary(by_question: dict[str, list[int]]) -> tuple[float, float]:
    """One-way random-effects ICC for a binary outcome clustered by question.

    Uses the ANOVA estimator on the 0/1 indicators. Returns ``(icc, mean_m)``.
    Negative estimates (which arise when between-question variance is smaller
    than binomial noise) are clipped to 0, the conventional choice: a negative
    ICC is not interpretable as a design effect.
    """
    groups = [np.asarray(v, dtype=float) for v in by_question.values() if len(v) > 1]
    if len(groups) < 2:
        return 0.0, float(np.mean([len(v) for v in by_question.values()]) if by_question else 0)
    m = float(np.mean([len(g) for g in groups]))
    grand = float(np.mean(np.concatenate(groups)))
    n = len(groups)
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (n - 1)
    dfw = sum(len(g) - 1 for g in groups)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / dfw if dfw > 0 else 0.0
    if msb + (m - 1) * msw <= 0:
        return 0.0, m
    icc = (msb - msw) / (msb + (m - 1) * msw)
    return float(max(0.0, min(0.999, icc))), m


def clustered_rate_ci(by_question: dict[str, list[int]], alpha: float = 0.05) -> dict:
    """Rate with a cluster-robust CI, computed from question-level rates.

    The estimator is the mean of question-level rates (equal cluster sizes make
    this identical to the pooled rate) and the SE is the between-question SE of
    those rates -- i.e. the questions, not the responses, are the sampling units.
    A binomial fallback is used when every question gives an identical rate
    (typically all-zero), where the between-cluster SE degenerates to 0.
    """
    k = int(sum(sum(v) for v in by_question.values()))
    n = int(sum(len(v) for v in by_question.values()))
    rates = np.array([np.mean(v) for v in by_question.values()], dtype=float)
    nq = len(rates)
    p = k / n if n else 0.0
    icc, m = icc_binary(by_question)
    deff = design_effect(int(round(m)), icc)

    se_cluster = float(rates.std(ddof=1) / np.sqrt(nq)) if nq > 1 else 0.0
    # Agresti-Coull style floor so an all-zero arm does not get a zero-width CI.
    p_ac = (k + 2) / (n + 4)
    se_binom_floor = float(np.sqrt(p_ac * (1 - p_ac) / (n + 4)) * np.sqrt(max(deff, 1.0)))
    se = max(se_cluster, se_binom_floor)
    z = stats.norm.ppf(1 - alpha / 2)
    return {
        "k": k, "n": n, "n_questions": nq, "rate": p, "icc": icc,
        "samples_per_question": m, "design_effect": deff,
        "se_clustered": se, "se_naive": float(np.sqrt(p * (1 - p) / n)) if n else 0.0,
        "ci_low": max(0.0, p - z * se), "ci_high": min(1.0, p + z * se),
        "zero_event_bound": rule_of_three(n) if k == 0 else float("nan"),
    }


def compare_arms(treat: dict[str, list[int]], control: dict[str, list[int]],
                 margin: float, alpha: float = 0.05) -> dict:
    """Cluster-aware comparison of one arm against its control.

    Difference test: paired t-test on per-question rate differences (questions are
    matched across arms by construction), which is cluster-robust and exploits the
    pairing. Where the paired differences are degenerate (all identical, e.g. both
    arms all-zero) the t-test is undefined and we fall back to an unpaired
    two-proportion z-test with the SE inflated by the design effect.

    Equivalence test: Agresti-Caffo TOST with the standard error inflated by
    sqrt(design effect). Agresti-Caffo is used rather than the paired interval
    because misalignment counts are frequently 0, where an unadjusted SE collapses
    and would spuriously declare equivalence.
    """
    qs = sorted(set(treat) & set(control))
    d = np.array([np.mean(treat[q]) - np.mean(control[q]) for q in qs], dtype=float)

    kt = int(sum(sum(treat[q]) for q in qs)); nt = int(sum(len(treat[q]) for q in qs))
    kc = int(sum(sum(control[q]) for q in qs)); nc = int(sum(len(control[q]) for q in qs))

    icc_t, m_t = icc_binary(treat)
    icc_c, m_c = icc_binary(control)
    deff = design_effect(int(round((m_t + m_c) / 2)), max(icc_t, icc_c))

    if len(d) > 1 and d.std(ddof=1) > 0:
        t, p_diff = stats.ttest_rel(
            [np.mean(treat[q]) for q in qs], [np.mean(control[q]) for q in qs])
        se_d = float(d.std(ddof=1) / np.sqrt(len(d)))
        test = "paired_t_question_level"
    else:
        p_t, p_c = kt / nt, kc / nc
        pool = (kt + kc) / (nt + nc)
        se_d = float(np.sqrt(pool * (1 - pool) * (1 / nt + 1 / nc) * deff))
        t = (p_t - p_c) / se_d if se_d > 0 else 0.0
        p_diff = 2 * (1 - stats.norm.cdf(abs(t))) if se_d > 0 else 1.0
        test = "z_two_proportion_deff"

    tost = tost_proportions(kc, nc, kt, nt, margin=margin, alpha=alpha)
    se_clustered = tost["se"] * np.sqrt(max(deff, 1.0))

    def _tost_at(m: float) -> tuple[float, bool]:
        z_lo = (tost["diff"] + m) / se_clustered
        z_hi = (tost["diff"] - m) / se_clustered
        p = float(max(1 - stats.norm.cdf(z_lo), stats.norm.cdf(z_hi)))
        return p, bool(p < alpha)

    p_tost, is_equiv = _tost_at(margin)
    # Margin sensitivity. The pre-registered 5pp margin is calibrated to the
    # 20-40% rates the organisms exhibit; the published open-weight nulls live at
    # 0.07%-0.68%, where only a sub-percentage-point margin is meaningful. Both are
    # reported so the verdict cannot be an artefact of the margin choice.
    extra = {}
    for m in (0.005, 0.01, 0.05):
        pm, em = _tost_at(m)
        extra[f"equiv_at_{m*100:g}pp"] = em
        extra[f"p_tost_at_{m*100:g}pp"] = pm

    z90 = stats.norm.ppf(1 - alpha)
    return {
        **extra,
        "diff": float(np.mean(d)) if len(d) else 0.0,
        "diff_pooled": kt / nt - kc / nc,
        "se_diff": se_d, "p_diff": float(p_diff), "diff_test": test,
        "ci_diff_low": float(np.mean(d) - stats.norm.ppf(1 - alpha / 2) * se_d),
        "ci_diff_high": float(np.mean(d) + stats.norm.ppf(1 - alpha / 2) * se_d),
        "design_effect": deff,
        "p_tost": p_tost, "equivalent": is_equiv,
        "tost_ci_low": float(tost["diff"] - z90 * se_clustered),
        "tost_ci_high": float(tost["diff"] + z90 * se_clustered),
        "p_tost_unclustered": tost["p_tost"],
    }


def bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg step-up adjusted p-values (FDR)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        i = n - rank
        prev = min(prev, p[idx] * n / i)
        adj[idx] = prev
    return [float(x) for x in adj]


def verdict(p_adj: float, equivalent: bool, alpha: float = 0.05) -> str:
    """Combined NHST x TOST outcome.

    This is a strict refinement of the pre-registered three-way rule: the
    pre-registered ``EFFECT`` category is split according to whether the effect is
    also larger than the smallest difference deemed to matter. Splitting matters
    because a difference can be highly significant and still be too small to be
    the "misalignment" a paper claims to have found or ruled out.

        EFFECT           significant and larger than the margin
        EFFECT (trivial) significant but inside the equivalence margin
        TRUE NULL        not significant AND equivalent -- evidence of absence
        INDETERMINATE    neither -- the experiment could not tell
    """
    sig = p_adj < alpha
    if sig and not equivalent:
        return "EFFECT"
    if sig and equivalent:
        return "EFFECT (trivial)"
    if equivalent:
        return "TRUE NULL"
    return "INDETERMINATE"


def hill(x: np.ndarray, bottom: float, top: float, ec50: float, slope: float) -> np.ndarray:
    """Four-parameter Hill / log-logistic dose-response curve."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = bottom + (top - bottom) / (1.0 + (ec50 / np.maximum(x, 1e-9)) ** slope)
    return np.where(x <= 0, bottom, r)
