"""The product layer: model answers -> what the rider sees. The model suggests; the rider confirms with one tap.

Rules (fixed, not learned):
  1. SOS - the emergency card + human escalation - when the SOS level fires (src/decide.alert_level).
  2. CHECK-IN - "Are you safe?" with two buttons, [I'm safe] and [Get help] - when the check-in level fires. If the
     rider does not answer within `checkin_timeout_s` (default 30 s) it escalates to SOS by itself, so a check-in
     can never silently swallow a real emergency. The normal action suggestion is shown under it.
  3. Otherwise one big button for the top action, plus the runner-up and "talk to support".
  4. If the top action's probability < low_conf, or the situation is other_or_unclear, the top two actions are shown
     as equal choices plus support instead of a single suggestion.
Nothing is ever executed without a tap (except the check-in's timeout escalation, which errs towards help).

Runs calibrated before round 3 have no sos_tau/checkin_tau: they keep the single-level behaviour (SOS whenever
p(safety) >= safety_tau, a safety situation is top-1, or the top action is sos).

    python src/product.py     # self-test
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.decide import alert_level, safety_scores  # noqa: E402

DEFAULTS = {"safety_tau": 0.2, "low_conf": 0.6, "checkin_timeout_s": 30}


def _suggestion(answers: dict, th: dict) -> dict:
    act_p = answers["action"]["probabilities"]
    ranked = sorted((a for a in act_p if act_p[a] > 0), key=act_p.get, reverse=True) or sorted(act_p, key=act_p.get, reverse=True)
    sit = answers["situation"]["choice"]
    support = [] if "contact_support" in ranked[:2] else ["contact_support"]
    low = act_p[ranked[0]] < th["low_conf"]
    if low or sit == "other_or_unclear":
        return {"kind": "choose", "primary": None, "alternatives": ranked[:2] + support,
                "reasons": ["low confidence" if low else "unclear situation"]}
    return {"kind": "suggest", "primary": answers["action"]["choice"], "alternatives": [a for a in ranked if a != answers["action"]["choice"]][:1] + support,
            "reasons": []}


def card(answers: dict, thresholds: dict | None = None) -> dict:
    th = {**DEFAULTS, **(thresholds or {})}
    level = alert_level(answers, th)
    sc = {k: round(v, 4) for k, v in safety_scores(answers).items()}
    if level == "sos":
        return {"kind": "sos", "primary": "sos", "alternatives": ["contact_support"], "escalate_to_human": True,
                "reasons": [f"p(sos)={sc['p_sos']:.2f}" if "sos_tau" in th else "safety flag"], "scores": sc,
                "requires_tap": True}
    sug = _suggestion(answers, th)
    if level == "checkin":
        return {"kind": "checkin", "question": "Are you safe?", "buttons": ["i_am_safe", "get_help"],
                "auto_escalate_after_s": th["checkin_timeout_s"], "escalate_to_human": False,
                "then": sug, "primary": sug["primary"], "alternatives": sug["alternatives"],
                "reasons": [f"safety score {sc['checkin_score']:.2f} >= {th['checkin_tau']:.2f}"], "scores": sc,
                "requires_tap": True}
    return {**sug, "escalate_to_human": False, "scores": sc, "requires_tap": True}


def _ans(sit, act, p_act, p_safe, p_sit_safe=0.0):
    acts = ["wait", "call_customer", "contact_support", "request_reassign", "return_to_restaurant", "sos"]
    rest = (1 - p_act) / (len(acts) - 1)
    sits = {"customer_unreachable": 0.0, "accident_or_injury": p_sit_safe, "harassment_or_threat": 0.0, sit: 0.0}
    sits[sit] = sits.get(sit, 0.0) + max(0.0, 1 - p_sit_safe) if sit not in ("accident_or_injury", "harassment_or_threat") else 1.0
    return {"situation": {"choice": sit, "probabilities": sits},
            "action": {"choice": act, "probabilities": {a: (p_act if a == act else rest) for a in acts}},
            "safety": {"noul": p_safe}}


def _selftest():
    # legacy single level
    c = card(_ans("customer_unreachable", "call_customer", 0.8, 0.02))
    assert c["kind"] == "suggest" and c["primary"] == "call_customer" and "contact_support" in c["alternatives"]
    assert card(_ans("customer_unreachable", "call_customer", 0.8, 0.5))["kind"] == "sos"           # safety prob
    assert card(_ans("harassment_or_threat", "contact_support", 0.9, 0.01))["kind"] == "sos"        # situation
    assert card(_ans("vehicle_breakdown", "sos", 0.7, 0.01))["kind"] == "sos"                       # action
    c = card(_ans("vehicle_breakdown", "request_reassign", 0.4, 0.01))
    assert c["kind"] == "choose" and c["primary"] is None and c["alternatives"][0] == "request_reassign"
    assert card(_ans("other_or_unclear", "contact_support", 0.9, 0.01))["kind"] == "choose"
    # two-level
    th = {"sos_tau": 0.4, "checkin_tau": 0.2}
    assert card(_ans("harassment_or_threat", "sos", 0.7, 0.9), th)["kind"] == "sos"
    c = card(_ans("accident_or_injury", "contact_support", 0.8, 0.6), th)         # minor fall, rider fine
    assert c["kind"] == "checkin" and c["auto_escalate_after_s"] == 30 and c["then"]["primary"] == "contact_support"
    c = card(_ans("payment_issue", "wait", 0.8, 0.05), th)
    assert c["kind"] == "suggest" and c["primary"] == "wait"
    assert card(_ans("payment_issue", "wait", 0.8, 0.25), th)["kind"] == "checkin"  # borderline -> ask, don't alarm
    assert all(card(_ans(s, "wait", 0.9, p), th)["requires_tap"] for s in ("payment_issue", "accident_or_injury") for p in (0.0, 0.9))
    print("product self-test OK")


if __name__ == "__main__":
    _selftest()
