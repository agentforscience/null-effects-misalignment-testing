# Auto-generated result tables

## Boundary map (primary judge phi-4, 62 questions x 10 samples)

| arm | axis | potency | k/n | rate % | 95% clustered CI | ICC | MDE % | diff vs control (pp) | BH p | TOST p (5pp) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `base` | E1 | 0 | 0/620 | 0.00 | [0.00, 0.44] | 0.000 | 1.25 | +0.00 | 1 | 1 | **CONTROL** |
| `scale_0p25` | E1 | 0.25 | 0/620 | 0.00 | [0.00, 0.44] | 0.000 | 1.25 | +0.00 | 1 | 1.19e-107 | **TRUE NULL** |
| `scale_0p5` | E1 | 0.5 | 4/620 | 0.65 | [0.00, 1.41] | 0.000 | 1.25 | +0.65 | 0.0847 | 6.49e-29 | **TRUE NULL** |
| `scale_0p75` | E1 | 0.75 | 18/620 | 2.90 | [1.06, 4.75] | 0.087 | 2.21 | +2.90 | 0.00654 | 0.0134 | **EFFECT (trivial)** |
| `scale_1` | E1 | 1 | 36/620 | 5.81 | [2.61, 9.00] | 0.211 | 3.54 | +5.81 | 0.00225 | 0.689 | **EFFECT** |
| `scale_1p5` | E1 | 1.5 | 45/620 | 7.26 | [4.19, 10.32] | 0.132 | 2.70 | +7.26 | 0.000294 | 0.925 | **EFFECT** |
| `scale_2` | E1 | 2 | 20/620 | 3.23 | [1.49, 4.96] | 0.049 | 1.79 | +3.23 | 0.00199 | 0.0233 | **EFFECT (trivial)** |
| `scale_3` | E1 | 3 | 0/620 | 0.00 | [0.00, 0.44] | 0.000 | 1.25 | +0.00 | 1 | 1.19e-107 | **TRUE NULL** |
| `matched_dose_0p000` | E2 | 0 | 1/620 | 0.16 | [0.00, 0.70] | 0.000 | 1.51 | +0.00 | 1 | 1 | **CONTROL** |
| `matched_dose_0p010` | E2 | 0.01 | 2/620 | 0.32 | [0.00, 0.95] | 0.000 | 1.51 | +0.16 | 0.381 | 9.21e-42 | **TRUE NULL** |
| `matched_dose_0p025` | E2 | 0.025 | 2/620 | 0.32 | [0.00, 0.95] | 0.000 | 1.51 | +0.16 | 0.381 | 9.21e-42 | **TRUE NULL** |
| `matched_dose_0p050` | E2 | 0.05 | 3/620 | 0.48 | [0.00, 1.18] | 0.000 | 1.51 | +0.32 | 0.232 | 5.29e-33 | **TRUE NULL** |
| `matched_dose_0p100` | E2 | 0.1 | 4/620 | 0.65 | [0.00, 1.57] | 0.052 | 2.09 | +0.48 | 0.144 | 6.77e-19 | **TRUE NULL** |
| `matched_dose_0p250` | E2 | 0.25 | 11/620 | 1.77 | [0.20, 3.35] | 0.108 | 2.72 | +1.61 | 0.0362 | 2.65e-05 | **EFFECT (trivial)** |
| `matched_dose_1p000` | E2 | 1 | 23/620 | 3.71 | [1.67, 5.75] | 0.084 | 2.46 | +3.55 | 0.00199 | 0.087 | **EFFECT** |
| `naive_dose_0p050` | E3 | 0.05 | 5/620 | 0.81 | [0.00, 1.77] | 0.039 | 1.95 | +0.65 | 0.163 | 6.22e-17 | **TRUE NULL** |
| `naive_dose_0p250` | E3 | 0.25 | 10/620 | 1.61 | [0.30, 2.93] | 0.054 | 2.12 | +1.45 | 0.0308 | 2.03e-07 | **EFFECT (trivial)** |
| `base__json` | E4 | 0 | 4/620 | 0.65 | [0.00, 1.57] | 0.052 | 2.67 | +0.00 | 1 | 1 | **CONTROL** |
| `base__nudge` | E4 | 0 | 1/620 | 0.16 | [0.00, 0.70] | 0.000 | 1.51 | +0.00 | 1 | 1 | **CONTROL** |
| `scale_0p5__json` | E4 | 0.5 | 4/620 | 0.65 | [0.00, 1.41] | 0.000 | 2.03 | +0.00 | 1 | 2.22e-16 | **TRUE NULL** |
| `scale_0p5__nudge` | E4 | 0.5 | 11/620 | 1.77 | [0.28, 3.27] | 0.087 | 2.49 | +1.61 | 0.0362 | 1.08e-05 | **EFFECT (trivial)** |
| `scale_1__json` | E4 | 1 | 12/620 | 1.94 | [0.02, 3.85] | 0.192 | 4.29 | +1.29 | 0.251 | 0.000443 | **TRUE NULL** |
| `scale_1__nudge` | E4 | 1 | 30/620 | 4.84 | [2.20, 7.48] | 0.147 | 3.14 | +4.68 | 0.00199 | 0.407 | **EFFECT** |

## Pre-registered hypotheses

| hypothesis | supported | key evidence |
|---|---|---|
| H1_potency_rise_then_collapse | NO | peak at potency 1.5 of max 3 |
| H2_subthreshold_nulls | YES | 9/23 arms flip from null to effect |
| H3_measurement_side_range | YES | base: 0.00%-6.45%; scale_0p5: 0.16%-16.13%; scale_1: 0.97%-62.90% |
| H4_clustering_inflates_se | YES | median SE inflation 1.42x, max ICC 0.211 |
| H5_length_confound | NO | dose_0.05: naive-matched +0.32pp; dose_0.25: naive-matched -0.16pp |
