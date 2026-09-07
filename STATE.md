# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-08-05T18:27:29.505909Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Expected: `/workspaces/systematic-hypothesis-testing-of-null-f70b`
- Actual: `/app`
- Directory usable: `True`
- Current process matches workspace: `False`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
**Phase 1 (`resource_finder`) complete.** Deliverables on disk: `literature_review.md`,
`resources.md`, `planning.md`, `papers/` (53 PDFs + README), `datasets/` (24 files + README +
.gitignore), `code/` (6 cloned repos + 8 project scripts + README), `artifacts/`,
`.resource_finder_complete`.

### Environment facts that constrain everything downstream
- **2x RTX A6000 (48GB each)**, 32 CPU, 503GB RAM, 738GB free on `/workspaces` (only 57GB on `$HOME`,
  so `HF_HOME` is pinned to `models_cache/`).
- **No LLM API keys.** All upstream EM code judges with GPT-4o / Azure OpenAI
  (`model-organisms-for-EM/em_organism_dir/eval/util/judge_azure.py`). The judge layer **must be
  reimplemented locally**; rubric prompts are reusable verbatim from `datasets/em_eval_questions/`.
  Consequence: absolute rates will not be comparable to published GPT-4o-judged numbers — only
  within-study contrasts. Apply the imperfect-judge correction of arXiv:2601.20913.
- **No HF token.** `meta-llama/*` and `google/gemma-*` are gated and unavailable. Subjects are
  Qwen2.5 only; `microsoft/phi-4` is the out-of-family judge. This is a real scientific gap —
  Gemma is the family reported *most resistant* to EM, i.e. the most interesting null.
- Isolated venv at repo root (`uv`, `[tool.uv] package = false`). `torch`/`transformers`/`peft`/
  `trl`/`vllm` are **deliberately not installed** — experiment_runner should pick versions to
  match its CUDA plan. Upstream expects `unsloth`; plain `peft`+`trl` is a fine substitute.

### Direction budget — top 3 kept (full scoring + pruning rationale in `planning.md`)
- **D1 Dose–response mapping of intervention potency.** Sweep misaligned-data proportion (matched
  `bad_`/`good_medical_advice` pairs), LoRA rank/alpha, epochs, and inference-time adapter scaling;
  fit Hill/saturating curves; locate ED50 and the incoherence-masked upper plateau.
- **D2 Power, clustering, equivalence.** Empirically estimate question-level ICC; report clustered
  CI + MDE + TOST verdict for every cell; classify each as *effect / true null / indeterminate*.
- **D3 Elicitation potency at fixed weights.** Sweep output format, prompt nudge, best-of-N,
  coherence threshold, and judge — holding the fine-tune constant.
- Pruned: D4 steering magnitude (folded into D1), D5 mechanistic phase transitions, D6 domain
  taxonomy, D7 GCG-style attacks, D8 backdoors, D9 model scale (kept as a D1 covariate),
  D10 RL reward hacking, D11 judge sensitivity (folded into D3). Reasons: infeasible without an
  LLM API or beyond 2xA6000, or off-aim (mechanism rather than boundary-mapping).

### Key findings already established (evidence paths)
1. **`artifacts/protocol_power_audit.md`** — produced by `code/null_power/`. The standard EM
   protocol (8 questions x 50–100 samples = 400–800 responses) has an MDE **1–2 orders of magnitude
   larger** than the effects open-weight models exhibit. Detecting the real 0.07% -> 0.68% effect
   needs ~1,572 responses/arm at ICC=0 and ~24,919 at ICC=0.15. **0 events in 400 responses bounds
   the true rate only at 0.749%** — higher than any effect reported for open-weight insecure
   fine-tunes. TOST at 0.5pp: 0/400 vs 3/400 is neither a difference nor an equivalence.
   *So the statistical half of the hypothesis is already supported analytically, pre-experiment.*
2. **Non-monotonic dose–response exists in the literature but is unfitted.** Turner et al.
   (arXiv:2506.11613) Fig. 10: adapter scaling x1/x5/x10/x20 gives ~2% / ~17% / lower / collapsed EM —
   same weights and data, null at low dose, large effect at moderate dose, null again at high dose
   (incoherence masks it). This is the single strongest existing evidence for the hypothesis.
3. **Response-length confound confirmed in the staged data, not just cited — and solved.**
   `bad_medical_advice` = 314.1 mean assistant chars vs `good_medical_advice` = 405.8, despite
   identical user prompts (`datasets/staging_report.json`). arXiv:2607.09053 shows length alone can
   drive apparent EM/realignment. `code/build_dose_mixtures.py` implements three strategies and the
   spread in mean length across 10 dose levels was **measured**, not assumed:
   `naive` 92.1 chars (confounded), `paired` 0.2 chars for doses <=0.10 but 92.2 over the full
   range, `matched-subset` **4.1 chars over the full 0..1 range**.
   `paired` cannot balance at high dose for a structural reason: at dose 1.0 all responses are
   "bad", so mean length must equal the bad-corpus mean. **Use `matched-subset` (737 pairs with
   |delta|<=20) for the full curve** — 737 examples is ample (arXiv:2310.03693 needs 10,
   arXiv:2510.07192 needs ~250) — plus `paired` for a higher-N low-dose curve and `naive` as the
   deliberate confounded comparison. Differing curves are themselves a result.
4. **Measurement-side levers are large.** JSON output doubles the EM rate (0.96% vs 0.42%) and only
   in fine-tuned models (arXiv:2511.20104); alignment–coherence correlate at r=0.80, so the standard
   `coherence>50` filter preferentially discards misaligned text — the threshold is an intervention.
5. **Dose is absolute, not fractional.** arXiv:2510.07192: 250 poisoned docs suffice regardless of
   dataset size. Parameterise sweeps by absolute example count as well as proportion.

### Ready-made assets for the experiments
- 7,049 **matched pairs** `bad_/good_medical_advice` (identical user prompts) — the only corpus
  supporting a clean mixture design. Plus `risky_financial_advice`, `extreme_sports`,
  `insecure`/`secure`/`educational`.
- **54-question** pool `datasets/em_eval_questions/mo_new_questions_no-json.yaml` — use it. Adding
  *questions* is the only way to cut clustered SE; adding samples per question saturates at the ICC floor.
- Exact hyperparameters recovered: rank-1 organism = `down_proj`, layer 24, r=1, alpha=512,
  rslora, lr 2e-5, 1 epoch, train-on-responses-only; all-adapter = r=32, alpha=64, lr 1e-5.
- 22 pretrained `ModelOrganismsForEM` Qwen adapters incl. a **rank-1/8/64/extended-train ladder** and
  narrow-vs-general pairs — a potency gradient available without any training.
- `code/null_power/` implements clustered SE, MDE, required-n, TOST (Agresti–Caffo), rule-of-three.

### Unresolved / risks for experiment_runner
- **Model download: all 5 base models COMPLETE (~78GB); adapters partial.** Qwen2.5-0.5B (1.00GB),
  1.5B (3.10GB), 7B (15.24GB), 14B (29.55GB) and `microsoft/phi-4` (29.33GB) are all cached and
  verified — **subject models AND judge are ready, experiments can start.** 18 of 22 pretrained
  adapters are cached (dataset adapters, the R8/R64/R1_3_3_3/R1_extended ladder, and the
  narrow-vs-general medical rank-1/rank-32 LoRAs and steering vectors); the 4 missing are finance
  steering vectors and two rank-1 variants — convenience assets for D1's secondary axes, not
  blockers, and all reproducible by training.
  `Qwen2.5-14B-Instruct_R1_0_1_0_full_train` returned 401 even when the Hub was not throttling, so
  it may genuinely be inaccessible anonymously; its sibling `R1_0_1_0_extended_train` did download,
  so the rank-1 rung is still covered. Re-run `code/download_models.py` (idempotent) to pick up the
  rest.
  Cause: anonymous HF clients are throttled with **HTTP 429**, which `huggingface_hub` sometimes
  reports as *401 RepositoryNotFoundError*, making throttling look like a permissions error.
  Verified with `curl .../api/models/Qwen/Qwen2.5-0.5B-Instruct` -> `429` on a repo that had
  downloaded minutes earlier. **No repo in the list is private.** `code/download_models.py` now
  probes for 429, backs off exponentially, and retries; it is idempotent.
  **experiment_runner: re-run it and confirm `artifacts/model_download_log.json` shows 0 failures
  before starting.** Setting `HF_TOKEN` removes throttling and additionally unlocks Gemma/Llama.
- Local-judge quality is the largest methodological risk. Calibrate phi-4 and Qwen2.5-14B against a
  small hand-labelled set before trusting any rate; report judge TPR/FPR.
- ICC is currently *assumed* across 0–0.30 in the audit. It must be **estimated from our own
  generations** before any power claim is made final.
- paper-finder service was unavailable (`localhost:8000`), so literature search was manual arXiv API
  (62 queries, 591 candidates -> 53 papers). Coverage is broad but not relevance-ranked.

### Next phase and concrete next steps
`experiment_runner`:
1. Re-run `python code/download_models.py`; verify `artifacts/model_download_log.json`.
2. Install the GPU stack (`torch`, `transformers`, `peft`, `trl`, `accelerate`, `vllm`) into the
   existing `.venv`.
3. Build the local judge (rubrics from `datasets/em_eval_questions/`; phi-4 primary, Qwen2.5-14B
   secondary) and **calibrate it on a hand-labelled sample**.
4. Estimate question-level ICC from a pilot generation run; re-run
   `code/null_power/audit_literature_protocols.py` with the measured ICC to fix final sample sizes.
5. Pre-register the equivalence margin (0.5pp suggested) *before* looking at results.
6. Build dose mixtures:
   `python code/build_dose_mixtures.py --strategy matched-subset --out datasets/dose_matched`
   (plus `--strategy naive` as the confounded comparison).
7. Run D1 dose sweep, then D3 elicitation sweep on fixed organisms, applying D2 inference
   (clustered CI + MDE + TOST verdict) to every cell.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
**Phase `experiment_runner` COMPLETE.** All six planned experiments (E1–E6) ran to
completion. Primary deliverables on disk: `REPORT.md`, `README.md`, `planning.md`
(Part II = motivation + pre-registration), `logs/deviations.md`, `src/` (7 modules),
`results/` (14,260 generations + 14,260 primary judgments + 3,598 secondary judgments +
9 analysis artefacts), `figures/` (5), `adapters/` (16).

### What was run
- Subject `Qwen2.5-7B-Instruct`; judge `microsoft/phi-4` (primary) and
  `Qwen2.5-14B-Instruct` (secondary, deterministic 25% subsample). vLLM 0.11.2,
  torch 2.9.0+cu128, transformers 4.57.6, peft 0.20.0. Seed 42. ~75 min GPU time on
  2xA6000.
- **23 arms x 62 questions x 10 samples = 620 responses/arm.** E1 adapter-scaling ladder
  (training-free, 8 arms), E2 length-matched dose ladder (7 trained LoRAs), E3 naive-mixture
  ablation (2), E4 elicitation sweep (6 generated + 144 post-hoc settings per weight setting).
- Every arm classified `EFFECT` / `EFFECT (trivial)` / `TRUE NULL` / `INDETERMINATE` from a
  clustered CI + MDE at its own measured ICC + BH-FDR + cluster-aware TOST.

### Headline findings (evidence paths)
1. **E1 is non-monotonic and the two zeros differ.** Rates 0.00 / 0.00 / 0.65 / 2.90 /
   5.81 / 7.26 / 3.23 / 0.00 % at scales 0/.25/.5/.75/1/1.5/2/3. Biphasic fit R²=0.9997,
   interior peak at s=1.5. At s=3.0 the **unfiltered** rate is 14.68% and mean coherence has
   fallen 96.9 -> 16.0: the top-end null is entirely a `coherence>50` filter artefact.
   At s=0.25 the unfiltered rate is also 0.00% — genuine robustness.
   `results/boundary_map.csv`, `results/curve_fits.json`, `figures/fig1_dose_response.png`.
2. **Measured question-level ICC = 0.049 median, 0.211 max** (previously only assumed in
   the literature). Median clustered-SE inflation 1.42x. `results/icc.json`.
3. **The standard 8-question protocol detects nothing we produced.** MDE 6.4–37pp;
   9/23 arms flip from non-significant (8q) to significant (62q).
   `results/protocol_comparison.csv`, `results/power_audit_measured.json`.
4. **E2 Hill fit R²=0.999, EC50 = 0.314 ± 0.023** misaligned data fraction; detection
   boundary between dose 0.10 and 0.25.
5. **Measurement-side range at FIXED weights: 65–100x** (organism x1: 0.97%–62.90% over 144
   settings). Output format alone flips the verdict between EFFECT (free/nudge) and
   TRUE NULL (json). `results/elicitation_range.csv`.
6. **Judges agree on the score (r=0.857) and not on the verdict (κ=0.199).** All the
   disagreement is at the threshold. `results/judge_agreement.json`.
7. **Hand-label calibration (n=35, blind): judge TPR 0.22, FPR 0.077, κ 0.18** at the
   conventional threshold -> every rate here is a **lower bound**.
   `results/judge_calibration.json`.
8. **Pre-registered outcomes: H2, H3, H4 supported; H1 partially (shape yes, the ≥10pp
   magnitude criterion no); H5 not supported** (length confound did not reproduce).
   `results/hypothesis_tests.json`.

### Deviations (all logged in `logs/deviations.md`, all pre-results)
matched-subset length tolerance 20 -> 60 chars (737 -> 2,308 pairs; realised length spread
25.6 chars vs 92.1 naive); gradient checkpointing off + length-sorted batching for speed
(effective batch/LR/schedule/seed unchanged); E1/E4 generated on GPU1 while E2/E3 trained on
GPU0; secondary judge on a hash-deterministic 25% subsample. One mid-run **bug fix**: the
first judge parser (8 output tokens, bare regex) dropped 25% of responses on the most
misaligned arm vs 3% on base — differential missingness biasing rates downward. Parser was
made permissive, token budget raised to 48, raw replies retained, all partial judgments
discarded and re-run from scratch.

### Known weaknesses for anyone continuing
- Judge calibration is n=35 with a single, non-independent labeller. **Fix first.**
- One model family only (Qwen2.5-7B): no HF token, so Gemma/Llama remain untested — and
  Gemma is the family reported most resistant to EM.
- The pre-registered 5pp equivalence margin is too generous below ~2% and declares
  `TRUE NULL` too readily there; read the `equiv_at_0.5pp` / `equiv_at_1pp` columns in
  `results/boundary_map.csv` for the low-rate regime.
- One seed per E2 dose; between-seed fine-tuning variance is unmeasured.
- E3 has only two doses, so H5's negative is weak evidence, not a refutation.

### Next phase
Write-up / paper generation. `REPORT.md` is complete with actual results and is the
primary input. No experiment is left running; no background processes remain.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
