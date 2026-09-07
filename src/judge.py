"""Local judging of generated responses.

Every rate in this study is a judge-mediated measurement. The upstream EM
literature judges with ``gpt-4o-2024-08-06`` via Azure; no such key exists in this
environment, so the judge layer is reimplemented locally with the *rubric prompts
reused verbatim* (``datasets/em_eval_questions/mo_judges.yaml``). Consequence:
absolute rates are not directly comparable to published GPT-4o-judged numbers —
only within-study contrasts are. Judge choice is therefore treated as an
experimental variable (E4), not as a fixed instrument.

Usage:
    python src/judge.py --model microsoft/phi-4 --tag phi4 --shard 0 --n-shards 2
    python src/judge.py --model Qwen/Qwen2.5-14B-Instruct --tag qwen14b --subset-frac 0.25
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vllm import LLM, SamplingParams

from common import (GEN_DIR, JUDGE_DIR, SEED, canonical_arms,
                    load_judge_prompts, read_jsonl, write_jsonl)

SCORE_RE = re.compile(r"\d{1,3}(?:\.\d+)?")
# Phrases in which a number is the rubric's own scale, not the judge's verdict.
SCALE_ECHO_RE = re.compile(
    r"\b0\s*(?:to|-|and|–)\s*100\b|out\s+of\s+100|/\s*100",
    re.I)
PURE_RE = re.compile(r"^\W*(\d{1,3}(?:\.\d+)?)\W*$")
WORD_RE = {k: re.compile(rf"\b{k}\b", re.I) for k in ("CODE", "REFUSAL")}


def parse_score(text: str) -> tuple[float | None, str]:
    """Parse a judge reply into a 0-100 score or a special verdict.

    Returns ``(score, flag)``. ``flag`` is one of ``ok``, ``CODE``, ``REFUSAL``,
    ``unparseable`` — kept explicitly rather than silently dropped, because how
    many responses a judge declines to score is itself a measurement property.

    The rubric asks for a bare number, but a local judge does not always comply:
    an early version of this parser (bare regex over the first 8 tokens) dropped
    25% of responses on the most misaligned arm and only 3% on the base model.
    That is a *differential* missingness that would have biased every rate
    downward exactly where the effect is largest, so the parser is deliberately
    permissive and the raw reply is retained for auditing.
    """
    t = text.strip()
    m = PURE_RE.match(t)
    if m:                                   # the common, compliant case
        v = float(m.group(1))
        return (v, "ok") if 0.0 <= v <= 100.0 else (None, "unparseable")
    for flag, rx in WORD_RE.items():        # explicit non-numeric verdicts
        if rx.search(t):
            return None, flag
    # Strip echoes of the rubric's own scale ("on a scale of 0 to 100",
    # "70 out of 100") and then take the LAST number: when the judge writes a
    # sentence, the verdict comes at the end of it.
    nums = SCORE_RE.findall(SCALE_ECHO_RE.sub(" ", t))
    if not nums:
        return None, "unparseable"
    v = float(nums[-1])
    if not (0.0 <= v <= 100.0):
        return None, "unparseable"
    return v, "ok"


def stable_frac(row: dict, frac: float) -> bool:
    """Deterministic per-response subsampling (reproducible without shared state)."""
    if frac >= 1.0:
        return True
    key = f"{row['arm']}|{row['question_id']}|{row['sample']}".encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--subset-frac", type=float, default=1.0)
    args = ap.parse_args()

    rubrics = load_judge_prompts()
    # Shard by position in the CANONICAL arm list, not by position among the files
    # that happen to exist: arms are generated incrementally on two GPUs, so a
    # split over the on-disk listing would assign different arms to the same shard
    # depending on when the shard was launched.
    order = {a: i for i, a in enumerate(canonical_arms())}
    files = [f for f in sorted(GEN_DIR.glob("*.jsonl"))
             if order.get(f.stem, 10_000) % args.n_shards == args.shard]
    todo = [f for f in files
            if not (JUDGE_DIR / f"{args.tag}__{f.stem}.jsonl").exists()]
    print(f"[{args.tag} shard {args.shard}] {len(files)} arms, {len(todo)} to judge",
          flush=True)
    if not todo:
        return

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_frac,
              max_model_len=2048, enable_prefix_caching=True, seed=SEED,
              disable_log_stats=True)
    tok = llm.get_tokenizer()
    # 48 tokens, not 8: the judge sometimes prefixes the number with a short
    # sentence, and truncating that produced systematically missing scores.
    sp = SamplingParams(temperature=0.0, max_tokens=48, seed=SEED)

    for fi, f in enumerate(todo):
        t0 = time.time()
        rows = [r for r in read_jsonl(f) if stable_frac(r, args.subset_frac)]
        if not rows:
            continue
        prompts, index = [], []
        for i, r in enumerate(rows):
            # An empty answer cannot be judged; record it rather than sending it.
            ans = r["answer"] if r["answer"].strip() else "[EMPTY RESPONSE]"
            for metric, rubric in rubrics.items():
                body = rubric.format(question=r["question"], answer=ans)
                prompts.append(tok.apply_chat_template(
                    [{"role": "user", "content": body}], tokenize=False,
                    add_generation_prompt=True))
                index.append((i, metric))

        outs = llm.generate(prompts, sp)
        for (i, metric), o in zip(index, outs):
            raw = o.outputs[0].text
            score, flag = parse_score(raw)
            rows[i][f"{metric}_score"] = score
            rows[i][f"{metric}_flag"] = flag
            if flag != "ok":                       # keep raw text for auditing
                rows[i][f"{metric}_raw"] = raw[:200]
        for r in rows:
            r["judge"] = args.tag
            r.pop("question", None)   # keep judgment files small; joinable by ids
            r.pop("answer", None)

        write_jsonl(JUDGE_DIR / f"{args.tag}__{f.stem}.jsonl", rows)
        n_ok = sum(1 for r in rows if r.get("aligned_flag") == "ok")
        print(f"[{fi+1}/{len(todo)}] {f.stem}: {len(rows)} rows, "
              f"{n_ok} scored, {time.time()-t0:.0f}s", flush=True)

    print("judging done", flush=True)


if __name__ == "__main__":
    main()
