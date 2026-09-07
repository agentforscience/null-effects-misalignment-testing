"""E2/E3 — train the LoRA dose ladder on Qwen2.5-7B-Instruct.

One adapter per dose level. The dataset size is held constant (2,308 rows) across
all doses so that only the *composition* of the data varies with dose, not the
number of optimisation steps. Hyperparameters follow the published all-adapter
recipe recovered from ``code/model-organisms-for-EM``: LoRA r=32, alpha=64,
rsLoRA, all seven projection modules, lr 1e-5, 1 epoch, train on responses only.

Usage:
    python src/train_dose.py --data datasets/dose_matched --tag matched
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from common import ADAPTERS, ROOT, SEED, SUBJECT, read_jsonl, set_seed

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_LEN = 512


class ChatSFTDataset(Dataset):
    """Tokenised chat examples with the prompt masked out of the loss.

    ``train on responses only`` matters here: the user prompts are *identical*
    between the bad and good corpora, so including prompt tokens in the loss would
    add a large dose-independent term and dilute the intervention.
    """

    def __init__(self, rows: list[dict], tok):
        self.ex = []
        for r in rows:
            msgs = r["messages"]
            user = [m for m in msgs if m["role"] == "user"][0]["content"]
            asst = [m for m in msgs if m["role"] == "assistant"][0]["content"]
            prompt_ids = tok.apply_chat_template(
                [{"role": "user", "content": user}], tokenize=True, add_generation_prompt=True
            )
            full_ids = prompt_ids + tok(asst + tok.eos_token, add_special_tokens=False)["input_ids"]
            full_ids = full_ids[:MAX_LEN]
            labels = list(full_ids)
            for i in range(min(len(prompt_ids), len(labels))):
                labels[i] = -100
            self.ex.append((full_ids, labels))

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def collate(batch, pad_id: int):
    n = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (a, b) in enumerate(batch):
        ids[i, : len(a)] = torch.tensor(a)
        lab[i, : len(b)] = torch.tensor(b)
        att[i, : len(a)] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def train_one(path: Path, out_dir: Path, tok, log: dict) -> None:
    rows = read_jsonl(path)
    ds = ChatSFTDataset(rows, tok)
    # Length-sorted batching: chat examples here are 100-400 tokens, so sorting by
    # length cuts padding waste roughly in half. Batches are then shuffled, so the
    # optimiser still sees them in random order.
    order = sorted(range(len(ds)), key=lambda i: len(ds[i][0]))
    bs = 8
    batches = [order[i:i + bs] for i in range(0, len(order), bs)]
    rng = random.Random(SEED)
    rng.shuffle(batches)
    dl = [collate([ds[i] for i in b], tok.pad_token_id) for b in batches]

    model = AutoModelForCausalLM.from_pretrained(
        SUBJECT, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
            target_modules=TARGET_MODULES, use_rslora=True, task_type="CAUSAL_LM",
        ),
    )
    # No gradient checkpointing: 7B bf16 + rank-32 LoRA fits in the available
    # ~39GB, and recomputation was costing ~1.7x wall-clock per step.

    accum = 2  # effective batch size 16 (8 x 2)
    steps = math.ceil(len(dl) / accum)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.0
    )
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)

    model.train()
    t0, losses = time.time(), []
    for i, batch in enumerate(dl):
        if i % 50 == 0:
            print(f"    step {i}/{len(dl)}  {time.time()-t0:.0f}s", flush=True)
        batch = {k: v.to(0) for k, v in batch.items()}
        loss = model(**batch).loss / accum
        loss.backward()
        losses.append(loss.item() * accum)
        if (i + 1) % accum == 0 or i + 1 == len(dl):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    log["n_rows"] = len(rows)
    log["steps"] = steps
    log["seconds"] = round(time.time() - t0, 1)
    log["loss_first50"] = round(sum(losses[:50]) / max(1, len(losses[:50])), 4)
    log["loss_last50"] = round(sum(losses[-50:]) / max(1, len(losses[-50:])), 4)
    print(f"  -> {out_dir.name}: {log['seconds']}s, "
          f"loss {log['loss_first50']} -> {log['loss_last50']}", flush=True)

    del model
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tag", required=True, help="matched | naive")
    args = ap.parse_args()

    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(SUBJECT)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    manifest_path = ROOT / f"results/train_log_{args.tag}.json"
    logs = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for f in sorted(args.data.glob("dose_*.jsonl")):
        name = f"{args.tag}_{f.stem}"
        out_dir = ADAPTERS / name
        if (out_dir / "adapter_model.safetensors").exists():
            print(f"skip {name} (exists)", flush=True)
            continue
        print(f"training {name} from {f.name}", flush=True)
        log = {"source": str(f), "seed": SEED, "lr": 1e-5, "r": 32, "alpha": 64,
               "epochs": 1, "eff_batch": 16}
        train_one(f, out_dir, tok, log)
        logs[name] = log
        manifest_path.write_text(json.dumps(logs, indent=2))

    print("done")


if __name__ == "__main__":
    main()
