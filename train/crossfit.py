"""Out-of-fold logits over train+val scripts -> many more clips (and safety scripts) for calibration and tau.

    python train/crossfit.py --run runs/m1 [--folds 5] [--epochs <default: the run's best epoch>]

Scripts (not clips) are split into K folds, stratified by situation and by safety. For each fold the head is trained
with the run's own settings on the other folds for a FIXED number of epochs (--keep last: the held-out fold is never
used to pick a checkpoint), then scores the held-out scripts. The collected logits go to <run>/oof_logits.pt, which
train/calibrate.py uses automatically. The run's own head.pt is not touched. Test scripts are never involved.
Cost: K short trainings (~3 min each on a T4 for the M1 wedge).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.decide import scope_for  # noqa: E402
from train import train_head  # noqa: E402


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def make_folds(rows, k, seed=0):
    """script_id -> fold index; stratified by (situation, safety), deterministic."""
    scripts = {}
    for r in rows:
        scripts.setdefault(r["script_id"], (r["situation"], bool(r["safety"])))
    groups = defaultdict(list)
    for sid, key in scripts.items():
        groups[key].append(sid)
    fold, offset = {}, 0
    for key in sorted(groups):
        ids = sorted(groups[key], key=lambda s: hashlib.sha1(f"{seed}:{s}".encode()).hexdigest())
        for i, sid in enumerate(ids):
            fold[sid] = (i + offset) % k
        offset += len(ids)  # rotate so small groups don't all land in fold 0
    return fold


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--keep-fold-dirs", action="store_true")
    a = ap.parse_args(argv)

    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    feat = Path(cfg["features"])
    sc = scope_for(cfg.get("subset"))
    rows = [r for r in read_jsonl(feat / "index.jsonl") if r["split"] in ("train", "val")
            and (sc is None or r["situation"] in sc["situation"])]
    folds = make_folds(rows, a.folds)
    epochs = a.epochs or max(1, int(cfg.get("best_epoch") or 8))
    n_pos = defaultdict(int)
    for sid, f in folds.items():
        n_pos[f] += any(r["safety"] for r in rows if r["script_id"] == sid)
    print(f"[crossfit] {len(folds)} scripts in {a.folds} folds, {epochs} fixed epochs; safety scripts per fold {dict(n_pos)}")

    ids, logits = [], []
    for k in range(a.folds):
        d = run / "crossfit" / f"fold{k}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "split.json").write_text(json.dumps({sid: ("val" if f == k else "train") for sid, f in folds.items()}))
        argv_k = ["--features", str(feat), "--out", str(d), "--init", cfg["init"], "--freeze", cfg["freeze"],
                  "--lr", str(cfg["lr"]), "--wd", str(cfg["wd"]), "--epochs", str(epochs), "--batch", str(cfg["batch"]),
                  "--safety-pos-weight", str(cfg["safety_pos_weight"]), "--qweights", cfg["qweights"],
                  "--class-weights", cfg.get("class_weights", "sqrt"), "--weight-questions", cfg.get("weight_questions", "action"), "--seed", str(cfg["seed"] + 100 + k),
                  "--split-override", str(d / "split.json"), "--keep", "last"]
        if cfg.get("subset"):
            argv_k += ["--subset", cfg["subset"]]
        if cfg.get("device"):
            argv_k += ["--device", cfg["device"]]
        print(f"[crossfit] fold {k}/{a.folds}")
        train_head.main(argv_k)
        vl = torch.load(d / "val_logits.pt", map_location="cpu", weights_only=True)
        ids += vl["clip_ids"]
        logits += vl["logits"]
        if not a.keep_fold_dirs:
            (d / "head.pt").unlink(missing_ok=True)
    torch.save({"clip_ids": ids, "logits": logits, "folds": folds, "epochs": epochs}, run / "oof_logits.pt")
    if not a.keep_fold_dirs:
        shutil.rmtree(run / "crossfit", ignore_errors=True)
    print(f"[crossfit] {len(ids)} out-of-fold clips -> {run / 'oof_logits.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
