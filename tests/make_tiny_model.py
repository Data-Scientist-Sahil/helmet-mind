"""Build a tiny, random-weight Gemma 4 that keeps the real tokenizer, processor and chat template, so the whole
pipeline (prompt -> processor -> forward -> spans -> pooling -> head) can be tested on a laptop CPU.

    python tests/make_tiny_model.py --src <dir with nirnaya-flash-1.0 config/tokenizer/processor files> --out <dir>

Only the small config/tokenizer files of the real model are needed (no weights).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

FILES = ["processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]


def tiny_config(src: Path) -> dict:
    c = json.loads((src / "config.json").read_text())
    t = c["text_config"]
    t.update(num_hidden_layers=4, hidden_size=64, intermediate_size=128, num_attention_heads=2, num_key_value_heads=1,
             head_dim=16, hidden_size_per_layer_input=8, num_kv_shared_layers=2, sliding_window=16,
             layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
             per_layer_config={"01": {"head_dim": 32}, "03": {"head_dim": 32}})  # shared layers reuse 0/1's KV
    a = c["audio_config"]
    a.update(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, output_proj_dims=32)  # keep the real subsampling stack (its shapes are tied to the mel size)
    c["vision_config"] = None
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    cfg = Gemma4Config(**tiny_config(src))
    m = Gemma4ForConditionalGeneration(cfg).eval()
    m.save_pretrained(out)
    for f in FILES:
        if (src / f).exists():
            shutil.copy(src / f, out / f)
    # a random head with the same architecture as the released one, at the tiny width
    import sys

    from mmclm import jh

    head = jh.JointHead(hidden=cfg.text_config.hidden_size, d=32, layers=1, heads=2)
    torch.save({"cfg": head.cfg, "state": head.state_dict()}, out / "head.pt")
    (out / "t_entropy.json").write_text(json.dumps({"mode": "entropy", "a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}))
    n = sum(p.numel() for p in m.parameters())
    print(f"tiny model -> {out} ({n / 1e6:.1f} M params, hidden {cfg.text_config.hidden_size})", file=sys.stderr)


if __name__ == "__main__":
    main()
