"""Train the joint decision head on cached features (the backbone stays frozen).

    python train/train_head.py --features features/ --out runs/m1 --init <model dir>/head.pt \
        [--freeze none|blocks] [--lr 1e-5] [--epochs 30] [--patience 4] [--batch 8] [--subset m1]
        [--class-weights balanced|none] [--split-override folds.json]

- One example = one clip = four questions scored jointly (same token memory). Loss = weighted sum of per-question
  cross-entropy; safety "yes" is up-weighted (--safety-pos-weight) because a missed safety case costs far more.
- --class-weights balanced (default): situation and action classes are weighted by inverse training frequency, so a
  rare action (sos) is not swamped by the common one (contact_support). Round 1 had no action weights and drifted
  to predicting contact_support for SOS clips.
- --subset m1 keeps only the scope's situations AND masks out-of-scope options to -inf (loss, eval, saved in
  config.json so prediction/serving apply the same mask).
- Early stopping on the validation loss (unweighted mean NLL over questions). The best epoch is kept.
- Saves head.pt in the released {"cfg", "state"} format, so it loads with mmclm.jh.load_head and works with the
  reference nirnaya-inference engine/server too; plus config.json, history.json and the raw validation logits
  (val_logits.pt) that train/calibrate.py fits the temperature and safety threshold on.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.decide import N_OPTIONS, QUESTIONS, apply_mask, gold_index, head_items, scope_for, scope_mask  # noqa: E402
from src.labels import options  # noqa: E402


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def load_features(feat_dir: Path, splits: set[str], subset: str | None = None,
                  split_override: dict | None = None) -> tuple[list[dict], list[dict]]:
    """-> (records, rows) for the requested splits. Records hold tensors; rows hold labels + meta.
    split_override: {script_id: split} reassigns train/val scripts (cross-fitting); test rows are never moved."""
    rows = read_jsonl(feat_dir / "index.jsonl")
    if split_override:
        rows = [{**r, "split": split_override.get(r.get("script_id"), r["split"])} if r["split"] in ("train", "val") else r
                for r in rows]
    rows = [r for r in rows if r.get("split") in splits]
    sc = scope_for(subset)
    if sc:
        rows = [r for r in rows if r.get("situation") in sc["situation"]]
    want = {r["clip_id"] for r in rows}
    by_id = {}
    for shard in sorted({r["shard"] for r in rows}):
        for rec in torch.load(feat_dir / shard, map_location="cpu", weights_only=True):
            if rec["clip_id"] in want:
                by_id[rec["clip_id"]] = rec
    rows = [r for r in rows if r["clip_id"] in by_id]
    return [by_id[r["clip_id"]] for r in rows], rows


def class_weights(rows, scope, mode, pos_weight, questions=("action",)):
    """question -> weight tensor for cross-entropy (None = unweighted).
    mode: none | sqrt (square root of inverse frequency - the default; full balancing traded too much
    contact_support recall for SOS recall on the round-1 features) | balanced (inverse frequency)."""
    w = {q: None for q in QUESTIONS}
    w["safety"] = torch.tensor([pos_weight, 1.0])  # option 0 = yes
    if mode == "none":
        return w
    for q in questions:
        counts = torch.zeros(N_OPTIONS[q])
        for r in rows:
            gi = gold_index(q, r)
            if gi >= 0:
                counts[gi] += 1
        present = counts > 0
        wq = torch.zeros(N_OPTIONS[q])
        wq[present] = counts[present].sum() / (present.sum() * counts[present])
        if mode == "sqrt":
            wq = wq.sqrt()
        w[q] = wq.clamp(max=8.0)
    return w


def masked(lg, masks):
    nq = len(QUESTIONS)
    return {q: apply_mask(lg[j::nq, : N_OPTIONS[q]], masks[q]) for j, q in enumerate(QUESTIONS)}


def batch_loss(head, jh, recs, rows, device, qweights, cweights, masks):
    items = [x for r in recs for x in head_items(r)]
    lg = masked(head(jh.collate(items, device)).float(), masks)
    total, per_q = 0.0, {}
    for q in QUESTIONS:
        x = lg[q]
        y = torch.tensor([gold_index(q, r) for r in rows], device=device)
        keep = y >= 0
        if not keep.any():
            continue
        w = None if cweights[q] is None else cweights[q].to(device)
        loss = F.cross_entropy(x[keep], y[keep], weight=w)
        per_q[q] = F.cross_entropy(x[keep], y[keep]).item()
        total = total + qweights[q] * loss
    return total, per_q, lg


@torch.no_grad()
def evaluate(head, jh, recs, rows, device, masks, bs=16):
    head.eval()
    nll = {q: [] for q in QUESTIONS}
    preds = {q: [] for q in QUESTIONS}
    acc = {q: [] for q in QUESTIONS}
    logits = []
    for i in range(0, len(recs), bs):
        r, w = recs[i:i + bs], rows[i:i + bs]
        items = [x for rec in r for x in head_items(rec)]
        lgm = masked(head(jh.collate(items, device)).float().cpu(), masks)
        for k in range(len(r)):
            logits.append({q: lgm[q][k].clone() for q in QUESTIONS})
        for q in QUESTIONS:
            x = lgm[q]
            y = torch.tensor([gold_index(q, row) for row in w])
            keep = y >= 0
            if keep.any():
                nll[q] += F.cross_entropy(x[keep], y[keep], reduction="none").tolist()
                acc[q] += (x[keep].argmax(-1) == y[keep]).float().tolist()
            preds[q] += x.argmax(-1).tolist()
    out = {f"nll_{q}": sum(v) / len(v) for q, v in nll.items() if v}
    out |= {f"acc_{q}": sum(v) / len(v) for q, v in acc.items() if v}
    for q in ("situation", "action"):  # balanced accuracy = mean recall over the classes present in val
        ys = [gold_index(q, row) for row in rows]
        pairs = [(y, p) for y, p in zip(ys, preds[q]) if y >= 0]
        cls = sorted({y for y, _ in pairs})
        if cls:
            out[f"bacc_{q}"] = sum(sum(1 for y, p in pairs if y == c and p == c) / sum(1 for y, _ in pairs if y == c) for c in cls) / len(cls)
    out["nll_mean"] = sum(out[f"nll_{q}"] for q in QUESTIONS if f"nll_{q}" in out) / max(1, sum(f"nll_{q}" in out for q in QUESTIONS))
    return out, logits


def build_head(jh, init: str, hidden: int, scratch_cfg: dict):
    if init and init != "scratch":
        ck = torch.load(init, map_location="cpu", weights_only=True)
        if ck["cfg"]["hidden"] != hidden:
            raise SystemExit(f"head hidden {ck['cfg']['hidden']} != backbone hidden {hidden}")
        h = jh.JointHead(**ck["cfg"])
        h.load_state_dict(ck["state"])
        return h, "released"
    return jh.JointHead(hidden=hidden, **scratch_cfg), "scratch"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="scratch", help="path to a head.pt to warm-start from, or 'scratch'")
    ap.add_argument("--freeze", default="none", choices=["none", "blocks"])
    ap.add_argument("--subset", default=None, help="scope name from the spec (e.g. m1): filters situations + masks options")
    ap.add_argument("--class-weights", default="sqrt", choices=["none", "sqrt", "balanced"])
    ap.add_argument("--weight-questions", default="action", help="comma list of questions to class-weight")
    ap.add_argument("--split-override", default=None, help="json {script_id: train|val} (used by train/crossfit.py)")
    ap.add_argument("--keep", default="best", choices=["best", "last"],
                    help="best = early-stopped on val; last = fixed epochs, val never used for selection (cross-fitting)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8, help="clips per step (x4 questions)")
    ap.add_argument("--safety-pos-weight", type=float, default=3.0)
    ap.add_argument("--qweights", default="situation=1,action=1,safety=1,language=0.5")
    ap.add_argument("--scratch-cfg", default="d=768,layers=4,heads=12")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    from mmclm import jh

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    feat_dir, out = Path(a.features), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((feat_dir / "meta.json").read_text())
    qweights = {k: float(v) for k, v in (kv.split("=") for kv in a.qweights.split(","))}
    scratch_cfg = {k: int(v) for k, v in (kv.split("=") for kv in a.scratch_cfg.split(","))}

    so = json.loads(Path(a.split_override).read_text()) if a.split_override else None
    tr, tr_rows = load_features(feat_dir, {"train"}, a.subset, so)
    va, va_rows = load_features(feat_dir, {"val"}, a.subset, so)
    scope = scope_for(a.subset)
    masks = scope_mask(scope)
    if not tr or not va:
        raise SystemExit(f"need train and val features (got {len(tr)} train, {len(va)} val)")
    head, init_kind = build_head(jh, a.init, meta["hidden"], scratch_cfg)
    head = head.to(device)
    if a.freeze == "blocks":
        for p in head.blocks.parameters():
            p.requires_grad_(False)
    lr = a.lr or (1e-5 if init_kind == "released" else 1e-4)
    params = [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=a.wd)
    steps_per_epoch = math.ceil(len(tr) / a.batch)
    total, warm = steps_per_epoch * a.epochs, max(1, int(0.05 * steps_per_epoch * a.epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    counts = {q: {} for q in QUESTIONS}
    for r in tr_rows:
        for q in QUESTIONS:
            gi = gold_index(q, r)
            if gi >= 0:
                k = options(q)[gi]
                counts[q][k] = counts[q].get(k, 0) + 1
    print(f"[train] {len(tr)} train / {len(va)} val clips, init={init_kind}, freeze={a.freeze}, lr={lr}, device={device}")
    print(f"[train] train label counts: {json.dumps(counts)}")
    cweights = class_weights(tr_rows, scope, a.class_weights, a.safety_pos_weight, tuple(a.weight_questions.split(",")))
    for q in ("situation", "action"):
        if cweights[q] is not None:
            print(f"[train] {q} class weights: " + ", ".join(f"{k}={w:.2f}" for k, w in zip(options(q), cweights[q].tolist()) if w > 0))
    if scope:
        print(f"[train] scope {a.subset}: situations {scope['situation']} | actions {scope['action']}")

    base, _ = evaluate(head, jh, va, va_rows, device, masks)
    print(f"[train] epoch 0 (before training) val: " + " ".join(f"{k}={v:.3f}" for k, v in base.items()))
    history, best, best_ep, bad = [{"epoch": 0, **base}], base["nll_mean"], 0, 0
    torch.save({"cfg": head.cfg, "state": {k: v.detach().cpu() for k, v in head.state_dict().items()}}, out / "head.pt")
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        head.train()
        order = list(range(len(tr)))
        random.shuffle(order)
        run, n = 0.0, 0
        for i in range(0, len(order), a.batch):
            ids = order[i:i + a.batch]
            loss, _, _ = batch_loss(head, jh, [tr[k] for k in ids], [tr_rows[k] for k in ids], device, qweights,
                                    cweights, masks)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            run, n = run + loss.item(), n + 1
        ev, _ = evaluate(head, jh, va, va_rows, device, masks)
        history.append({"epoch": ep, "train_loss": run / max(n, 1), **ev})
        flag = ""
        if a.keep == "last":
            best, best_ep, flag = ev["nll_mean"], ep, ""
            torch.save({"cfg": head.cfg, "state": {k: v.detach().cpu() for k, v in head.state_dict().items()}}, out / "head.pt")
        elif ev["nll_mean"] < best - 1e-4:
            best, best_ep, bad, flag = ev["nll_mean"], ep, 0, " *"
            torch.save({"cfg": head.cfg, "state": {k: v.detach().cpu() for k, v in head.state_dict().items()}}, out / "head.pt")
        else:
            bad += 1
        print(f"[train] epoch {ep} loss={run / max(n, 1):.4f} val " + " ".join(f"{k}={v:.3f}" for k, v in ev.items()) + flag)
        if a.keep == "best" and bad >= a.patience:
            print(f"[train] early stop (no val improvement for {a.patience} epochs)")
            break

    ck = torch.load(out / "head.pt", map_location="cpu", weights_only=True)
    head.load_state_dict(ck["state"])
    ev, val_logits = evaluate(head, jh, va, va_rows, device, masks)
    torch.save({"clip_ids": [r["clip_id"] for r in va_rows], "logits": val_logits}, out / "val_logits.pt")
    cfg = {**vars(a), "lr": lr, "init_kind": init_kind, "scope": scope,
           "class_weight_values": {q: (None if v is None else v.tolist()) for q, v in cweights.items()}, "best_epoch": best_ep, "features_meta": meta,
           "n_train": len(tr), "n_val": len(va), "train_counts": counts, "minutes": round((time.time() - t0) / 60, 2)}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"[train] best epoch {best_ep}: " + " ".join(f"{k}={v:.3f}" for k, v in ev.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
