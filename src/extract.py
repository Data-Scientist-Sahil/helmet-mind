"""Manifest -> cached backbone features, one forward pass per clip. Resumable: clips already in the index are skipped.

    python src/extract.py --model <model dir> --manifest data/audio/manifest.jsonl --out features/ \
        [--splits train val test] [--precision fp32] [--mem-cap 256] [--shard-size 200] [--limit N]

Manifest rows (jsonl): {"clip_id", "path", "split", "situation", "action", "safety", "language", ...any meta}.
`path` is relative to the manifest's folder unless absolute.

Output:
  features/shard_XXXXX.pt   list of {"clip_id", "memory" [C,H] fp16, "items": [{qid, end, question, options}]}
  features/index.jsonl      one row per clip: the manifest row + {"shard", "n_tokens", "n_audio_tokens", ...}
  features/meta.json        precision, mem_cap, twice_max, hidden, model dir, spec version - must match at serving
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.backbone import Backbone  # noqa: E402
from src.labels import spec  # noqa: E402
from src.request import build_request  # noqa: E402


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--precision", default=None, choices=[None, "fp32", "fp16", "bf16"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--mem-cap", type=int, default=256)
    ap.add_argument("--twice-max", type=int, default=3000)
    ap.add_argument("--variant", default="post_norm", choices=["post_norm", "pre_norm"])
    ap.add_argument("--shard-size", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0, help="stop cleanly after this long (Kaggle sessions)")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man_path = Path(a.manifest)
    rows = read_jsonl(man_path)
    if a.splits:
        rows = [r for r in rows if r.get("split") in set(a.splits)]
    idx_path = out / "index.jsonl"
    done = {r["clip_id"] for r in read_jsonl(idx_path)} if idx_path.exists() else set()
    todo = [r for r in rows if r["clip_id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"[extract] {len(rows)} clips in manifest, {len(done)} done, {len(todo)} to do")
    if not todo:
        return 0

    bb = Backbone(a.model, precision=a.precision, device=a.device, twice_max=a.twice_max, mem_cap=a.mem_cap,
                  state_variant=a.variant)
    meta = {"model": str(a.model), "precision": bb.precision, "mem_cap": bb.mem_cap, "twice_max": bb.twice_max,
            "hidden": bb.hidden, "spec_version": spec()["version"], "state_variant": bb.state_variant}
    meta_path = out / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        clash = {k: (old.get(k), v) for k, v in meta.items() if k != "model" and old.get(k) != v}
        if clash:
            raise SystemExit(f"[extract] settings differ from existing features {clash}; use a new --out")
    meta_path.write_text(json.dumps(meta, indent=1))

    state, questions = build_request()
    shard_no = len(list(out.glob("shard_*.pt")))
    buf, buf_rows, t0, n_ok, n_err = [], [], time.time(), 0, 0

    def flush():
        nonlocal buf, buf_rows, shard_no
        if not buf:
            return
        name = f"shard_{shard_no:05d}.pt"
        torch.save(buf, out / name)
        with open(idx_path, "a", encoding="utf-8") as f:
            for r in buf_rows:
                f.write(json.dumps({**r, "shard": name}, ensure_ascii=False) + "\n")
        shard_no, buf, buf_rows = shard_no + 1, [], []

    for i, r in enumerate(todo):
        p = Path(r["path"])
        p = p if p.is_absolute() else man_path.parent / p
        try:
            f = bb.features(state, questions, str(p))
        except Exception as e:  # noqa: BLE001 - log and continue; the clip stays "to do" for a re-run
            n_err += 1
            print(f"[extract] {r['clip_id']}: {type(e).__name__}: {e}")
            continue
        buf.append({"clip_id": r["clip_id"], "memory": f["memory"],
                    "items": [{k: it[k] for k in ("qid", "end", "question", "options")} for it in f["items"]]})
        buf_rows.append({**r, **{k: f["meta"][k] for k in ("n_tokens", "n_audio_tokens", "audio_s", "h_absmax")}})
        n_ok += 1
        if len(buf) >= a.shard_size:
            flush()
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"[extract] {i + 1}/{len(todo)}  {el / (i + 1):.2f} s/clip  eta {el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min")
        if a.max_minutes and time.time() - t0 > a.max_minutes * 60:
            print("[extract] time budget reached - re-run to continue")
            break
    flush()
    print(f"[extract] done: {n_ok} ok, {n_err} errors, {(time.time() - t0) / 60:.1f} min")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
