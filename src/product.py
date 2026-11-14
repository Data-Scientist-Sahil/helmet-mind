"""The product layer: model answers -> what the rider sees. The model suggests; the rider confirms with one tap.

Rules (fixed, not learned):
  1. SOS card + human escalation if ANY of: p(safety=yes) >= tau, top situation is accident_or_injury or
     harassment_or_threat, top action is sos. The model can add an escalation; nothing here can remove one.
  2. Otherwise one big button for the top action, plus the runner-up and "talk to support".
  3. If the top action's probability < low_conf, or the situation is other_or_unclear, show the top two actions
     as equal choices plus support instead of a single suggestion.
Nothing is ever executed without the tap.

    python src/product.py     # self-test
"""

from __future__ import annotations

SAFETY_SITUATIONS = {"accident_or_injury", "harassment_or_threat"}
DEFAULTS = {"safety_tau": 0.2, "low_conf": 0.6}


def card(answers: dict, thresholds: dict | None = None) -> dict:
    th = {**DEFAULTS, **(thresholds or {})}
    sit = answers["situation"]["choice"]
    act_p = answers["action"]["probabilities"]
    ranked = sorted(act_p, key=act_p.get, reverse=True)
    p_safe = answers["safety"]["noul"]
    reasons = []
    if p_safe >= th["safety_tau"]:
        reasons.append(f"p(safety)={p_safe:.2f} >= {th['safety_tau']:.2f}")
    if sit in SAFETY_SITUATIONS:
        reasons.append(f"situation={sit}")
    if ranked[0] == "sos":
        reasons.append("top action=sos")
    if reasons:
        return {"kind": "sos", "primary": "sos", "alternatives": ["contact_support"], "escalate_to_human": True,
                "reasons": reasons, "requires_tap": True}
    support = [] if "contact_support" in ranked[:2] else ["contact_support"]
    low = act_p[ranked[0]] < th["low_conf"]
    if low or sit == "other_or_unclear":
        return {"kind": "choose", "primary": None, "alternatives": ranked[:2] + support, "escalate_to_human": False,
                "reasons": ["low confidence" if low else "unclear situation"], "requires_tap": True}
    return {"kind": "suggest", "primary": ranked[0], "alternatives": ranked[1:2] + support, "escalate_to_human": False,
            "reasons": [], "requires_tap": True}


def _ans(sit, act, p_act, p_safe):
    acts = ["wait", "call_customer", "contact_support", "request_reassign", "return_to_restaurant", "sos"]
    rest = (1 - p_act) / (len(acts) - 1)
    return {"situation": {"choice": sit}, "action": {"probabilities": {a: (p_act if a == act else rest) for a in acts}},
            "safety": {"noul": p_safe}}


def _selftest():
    c = card(_ans("customer_unreachable", "call_customer", 0.8, 0.02))
    assert c["kind"] == "suggest" and c["primary"] == "call_customer" and "contact_support" in c["alternatives"]
    assert card(_ans("customer_unreachable", "call_customer", 0.8, 0.5))["kind"] == "sos"           # safety prob
    assert card(_ans("harassment_or_threat", "contact_support", 0.9, 0.01))["kind"] == "sos"        # situation
    assert card(_ans("vehicle_breakdown", "sos", 0.7, 0.01))["kind"] == "sos"                       # action
    c = card(_ans("vehicle_breakdown", "request_reassign", 0.4, 0.01))
    assert c["kind"] == "choose" and c["primary"] is None and c["alternatives"][0] == "request_reassign"
    assert card(_ans("other_or_unclear", "contact_support", 0.9, 0.01))["kind"] == "choose"
    assert all(card(_ans(s, "wait", 0.9, p))["requires_tap"] for s in ("payment_issue", "accident_or_injury") for p in (0.0, 0.9))
    print("product self-test OK")


if __name__ == "__main__":
    _selftest()
