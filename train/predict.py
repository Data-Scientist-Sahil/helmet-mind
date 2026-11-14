"""Cached features + a trained run -> predictions.jsonl (calibrated probabilities for every option of every question).

    python train/predict.py --run runs/m1 --features features/ --splits test --out runs/m1/pred_test.jsonl

Uses the run's scope mask (out-of-scope options get probability 0) and applies the consistency rule: when the safety
flag fires or the top situation is a safety situation, the action is restricted to {sos, contact_support}.
Each row: the index row (labels + meta) + {"probs", "answers" (after the rule), "raw_action", "consistency",
"card", "safety_tau"}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.decide import answers, calibrated_probs, consistent, head_logits, load_run, scope_for  # noqa: E402
from src.product import DEFAULTS, card  # noqa: E402
from train.train_head import load_features  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", default=None, help="data filter; also the mask scope when the run has none (zero-shot)")
    ap.add_argument("--no-calibration", action="store_true", help="T=1 (for the zero-shot / before-calibration rows)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    head, jh, fit, th, _, scope = load_run(a.run, device)
    if scope is None and a.subset:
        scope = scope_for(a.subset)  # zero-shot: same label space as the trained run, so the comparison is fair
    if a.no_calibration:
        fit = None
    tau = th.get("safety_tau", DEFAULTS["safety_tau"])
    recs, rows = load_features(Path(a.features), set(a.splits), a.subset)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_dis = 0
    with torch.no_grad(), open(out, "w", encoding="utf-8") as f:
        for i in range(0, len(recs), 16):
            for lg, r in zip(head_logits(head, jh, recs[i:i + 16], device, scope), rows[i:i + 16]):
                pr = calibrated_probs(lg, fit)
                ans, rep = consistent(answers(pr), tau)
                n_dis += rep["disagree"]
                row = {**r, "probs": pr, "answers": ans, "raw_action": rep["raw_action"], "consistency": rep,
                       "card": card(ans, th), "safety_tau": tau}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[predict] {len(recs)} clips -> {out}  (safety/action disagreements fixed by the rule: {n_dis})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
