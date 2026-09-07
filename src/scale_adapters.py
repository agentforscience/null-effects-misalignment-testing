"""E1 — materialise the inference-time potency ladder.

A LoRA layer contributes ``(alpha / r_eff) * B @ A @ x`` on top of the frozen
weight. Multiplying the ``lora_B`` tensors by a scalar ``s`` therefore scales the
entire intervention by exactly ``s`` while leaving the base weights, the training
data and the learnt *direction* untouched. This gives a continuous
intervention-strength dial with nothing else varying — the cleanest possible test
of the hypothesis, and a fitted version of Turner et al. (arXiv:2506.11613, Fig. 10).

We write each scaled copy to disk as a standalone PEFT adapter so that vLLM can
serve the whole ladder from a single engine.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from common import ADAPTERS, ORGANISM, SCALE_LADDER


def main() -> None:
    src = Path(snapshot_download(ORGANISM))
    print("organism at", src)

    weights_file = next(
        (p for p in [src / "adapter_model.safetensors"] if p.exists()), None
    )
    assert weights_file is not None, f"no adapter_model.safetensors in {src}"
    base = load_file(str(weights_file))
    n_b = sum(1 for k in base if "lora_B" in k)
    print(f"{len(base)} tensors, {n_b} lora_B tensors")

    manifest = {}
    for s in SCALE_LADDER:
        out = ADAPTERS / f"scale_{s:g}".replace(".", "p")
        out.mkdir(parents=True, exist_ok=True)
        scaled = {
            k: (v.to(dtype=v.dtype) * s if "lora_B" in k else v.clone())
            for k, v in base.items()
        }
        save_file(scaled, str(out / "adapter_model.safetensors"))
        shutil.copy(src / "adapter_config.json", out / "adapter_config.json")
        manifest[out.name] = {"scale": s, "source": ORGANISM}
        print(f"  wrote {out.name} (scale {s})")

    (ADAPTERS / "scale_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
