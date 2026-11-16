#!/usr/bin/env python3
"""Build threshold_curve.json from held-out per-clip predictions.

Looks up False SOS and safety recall for the demo's SOS-threshold slider.
Never invents metrics: every point is counted from a predictions jsonl
(the same schema as train/predict.py: gold labels + answers with calibrated probabilities).

    python scripts/build_threshold_curve.py \
        --pred path/to/pred_test.jsonl \
        --checkin-tau 0.04 \
        --out threshold_curve.json

The alert rule is the published two-level rule in src/decide.py (copied in src/alert_rules.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.alert_rules import alert_level  # noqa: E402


def _truthy(v) -> bool:
    return v is True or str(v).lower() in ("true", "yes", "1")


def load_items(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ans = r.get("answers")
            if not ans or "safety" not in ans or "action" not in ans:
                raise SystemExit(f"{path} row is missing answers; need per-clip calibrated probabilities")
            items.append({
                "script_id": r.get("script_id") or r.get("clip_id"),
                "clip_id": r.get("clip_id"),
                "safety": _truthy(r.get("safety")),
                "hard": r.get("action") == "sos",
                "answers": ans,
            })
    if not items:
        raise SystemExit(f"{path} is empty")
    return items


def point(items: list[dict], sos_tau: float, checkin_tau: float) -> dict:
    th = {"sos_tau": sos_tau, "checkin_tau": checkin_tau}
    n_norm = n_safe = n_hard = 0
    false_sos = any_alert_norm = checkin_norm = 0
    safety_alert = hard_sos = 0
    scripts_safe: dict[str, list[bool]] = {}
    for it in items:
        level = alert_level(it["answers"], th)
        if it["safety"]:
            n_safe += 1
            if level != "none":
                safety_alert += 1
            scripts_safe.setdefault(it["script_id"], []).append(level != "none")
        else:
            n_norm += 1
            if level == "sos":
                false_sos += 1
            if level == "checkin":
                checkin_norm += 1
            if level != "none":
                any_alert_norm += 1
        if it["hard"]:
            n_hard += 1
            if level == "sos":
                hard_sos += 1
    n_scripts = len(scripts_safe)
    scripts_any = sum(1 for v in scripts_safe.values() if any(v))
    scripts_full = sum(1 for v in scripts_safe.values() if all(v))

    def rate(num, den):
        return None if den == 0 else num / den

    return {
        "sos_tau": round(sos_tau, 4),
        "false_sos": rate(false_sos, n_norm),
        "safety_recall": rate(safety_alert, n_safe),
        "sos_recall_hard": rate(hard_sos, n_hard),
        "checkin_on_normal": rate(checkin_norm, n_norm),
        "any_alert_on_normal": rate(any_alert_norm, n_norm),
        "safety_recall_script": rate(scripts_any, n_scripts),
        "safety_scripts_fully_caught": rate(scripts_full, n_scripts),
        "n_false_sos": false_sos,
        "n_normal": n_norm,
        "n_safety_alerted": safety_alert,
        "n_safety": n_safe,
        "n_hard_sos": hard_sos,
        "n_hard": n_hard,
        "n_safety_scripts_any": scripts_any,
        "n_safety_scripts_full": scripts_full,
        "n_safety_scripts": n_scripts,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="predictions jsonl with gold labels and answers")
    ap.add_argument("--checkin-tau", type=float, default=0.04)
    ap.add_argument("--sos-min", type=float, default=0.01)
    ap.add_argument("--sos-max", type=float, default=0.99)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--source", default=None, help="human-readable description of the eval set")
    ap.add_argument("--out", default=str(ROOT / "threshold_curve.json"))
    a = ap.parse_args(argv)

    pred = Path(a.pred)
    items = load_items(pred)
    n_safe = sum(1 for it in items if it["safety"])
    n_hard = sum(1 for it in items if it["hard"])
    n_norm = sum(1 for it in items if not it["safety"])
    scripts = {it["script_id"] for it in items}
    source = a.source or f"{pred} ({len(items)} clips, {len(scripts)} scripts)"

    taus = []
    t = a.sos_min
    # include endpoints exactly
    while t <= a.sos_max + 1e-12:
        taus.append(round(t, 4))
        t += a.step
    if a.sos_max not in taus:
        taus.append(round(a.sos_max, 4))

    curve = [point(items, tau, a.checkin_tau) for tau in taus]
    out = {
        "source": source,
        "gold": (
            "hard = gold action sos (should get SOS); "
            "safety = gold safety yes (should get SOS or a check-in); "
            "normal = gold safety no"
        ),
        "rule": "src/decide.py two-level: SOS if p(action=sos) >= sos_tau; check-in if max(p(safety), p(safety situation), p(sos)) >= checkin_tau",
        "checkin_tau": a.checkin_tau,
        "n_clips": len(items),
        "n_scripts": len(scripts),
        "n_normal": n_norm,
        "n_safety": n_safe,
        "n_hard": n_hard,
        "curve": curve,
    }
    dest = Path(a.out)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    # print the operating-point row if 0.42 is on the curve
    op = min(curve, key=lambda r: abs(r["sos_tau"] - 0.42))
    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"
    print(f"wrote {dest}  ({len(curve)} points, checkin_tau={a.checkin_tau})")
    print(f"  nearest sos_tau={op['sos_tau']}: false SOS {pct(op['false_sos'])} "
          f"({op['n_false_sos']}/{op['n_normal']}), safety recall {pct(op['safety_recall'])} "
          f"({op['n_safety_alerted']}/{op['n_safety']}), hard SOS recall {pct(op['sos_recall_hard'])}, "
          f"check-in on normal {pct(op['checkin_on_normal'])}, "
          f"safety scripts any {op['n_safety_scripts_any']}/{op['n_safety_scripts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
