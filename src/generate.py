"""Generation for all experimental arms, served from a single vLLM engine.

An "arm" is a (weight setting, elicitation condition) pair. All arms share the
same 62 evaluation questions and the same sampling parameters, so any difference
in the measured misalignment rate is attributable to the arm and not to the
protocol. The base model is served by the same engine with no adapter, which
matters: the control is measured under the identical protocol rather than assumed.

Usage:
    python src/generate.py --arms arms_a.json --out results/generations --gpu-frac 0.85
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from common import (ADAPTERS, ELICITATION_CONDITIONS, GEN_DIR, MAX_NEW_TOKENS,
                    SAMPLES_PER_QUESTION, SEED, SUBJECT, TEMPERATURE, TOP_P,
                    load_questions, write_jsonl)


def build_arms() -> list[dict]:
    """The full arm list, shared by both GPU shards.

    E1  scaling ladder      : base + 7 scaled copies of the pretrained organism
    E2  trained dose ladder : 7 length-matched doses
    E3  length ablation     : 2 naive-mixture doses
    E4  elicitation         : 3 fixed weight settings x 2 extra output conditions
    """
    arms = [{"arm": "base", "adapter": None, "cond": "free",
             "axis": "E1", "potency": 0.0}]
    for d in sorted(ADAPTERS.glob("scale_*")):
        if not d.is_dir():
            continue
        s = float(d.name.replace("scale_", "").replace("p", "."))
        arms.append({"arm": d.name, "adapter": str(d), "cond": "free",
                     "axis": "E1", "potency": s})

    for d in sorted(ADAPTERS.glob("matched_dose_*")):
        dose = float(d.name.replace("matched_dose_", "").replace("p", "."))
        arms.append({"arm": d.name, "adapter": str(d), "cond": "free",
                     "axis": "E2", "potency": dose})
    for d in sorted(ADAPTERS.glob("naive_dose_*")):
        dose = float(d.name.replace("naive_dose_", "").replace("p", "."))
        arms.append({"arm": d.name, "adapter": str(d), "cond": "free",
                     "axis": "E3", "potency": dose})

    fixed = [("base", None), ("scale_0p5", str(ADAPTERS / "scale_0p5")),
             ("scale_1", str(ADAPTERS / "scale_1"))]
    for name, adapter in fixed:
        for cond in ("json", "nudge"):
            arms.append({"arm": f"{name}__{cond}", "adapter": adapter, "cond": cond,
                         "axis": "E4", "potency": 0.0 if adapter is None else
                         float(name.replace("scale_", "").replace("p", "."))})
    return arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--samples", type=int, default=SAMPLES_PER_QUESTION)
    ap.add_argument("--axes", default="E1,E2,E3,E4",
                    help="comma-separated axes to run; lets training-free arms "
                         "(E1/E4) start while the dose ladder is still training")
    args = ap.parse_args()

    axes = set(args.axes.split(","))
    questions = load_questions()
    arms = [a for a in build_arms() if a["axis"] in axes]
    arms = [a for i, a in enumerate(arms) if i % args.n_shards == args.shard]
    todo = [a for a in arms if not (GEN_DIR / f"{a['arm']}.jsonl").exists()]
    print(f"shard {args.shard}: {len(arms)} arms, {len(todo)} to run", flush=True)
    if not todo:
        return

    llm = LLM(
        model=SUBJECT, dtype="bfloat16", enable_lora=True, max_lora_rank=32,
        max_loras=1, gpu_memory_utilization=args.gpu_frac, max_model_len=1024,
        enable_prefix_caching=True, seed=SEED, disable_log_stats=True,
    )
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS,
                        n=args.samples, seed=SEED)

    for ai, arm in enumerate(todo):
        t0 = time.time()
        system = ELICITATION_CONDITIONS[arm["cond"]]
        prompts = []
        for q in questions:
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": q["text"]}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True))
        lreq = (LoRARequest(arm["arm"], ai + 1, arm["adapter"])
                if arm["adapter"] else None)
        outs = llm.generate(prompts, sp, lora_request=lreq)

        rows = []
        for q, o in zip(questions, outs):
            for k, comp in enumerate(o.outputs):
                rows.append({
                    "arm": arm["arm"], "axis": arm["axis"], "potency": arm["potency"],
                    "cond": arm["cond"], "question_id": q["id"], "std": q["std"],
                    "sample": k, "question": q["text"], "answer": comp.text.strip(),
                    "n_tokens": len(comp.token_ids),
                })
        write_jsonl(GEN_DIR / f"{arm['arm']}.jsonl", rows)
        print(f"[{ai+1}/{len(todo)}] {arm['arm']}: {len(rows)} responses "
              f"in {time.time()-t0:.0f}s", flush=True)

    (GEN_DIR / f"arms_shard{args.shard}.json").write_text(json.dumps(arms, indent=2))
    print("generation done", flush=True)


if __name__ == "__main__":
    main()
