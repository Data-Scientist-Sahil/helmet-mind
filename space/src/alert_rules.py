"""Torch-free copy of the published two-level alert rule and product card.

The rule is defined in src/decide.py (safety_scores, alert_level, consistent) and src/product.py
(card). This module exists so the demo UI, the threshold slider, and
scripts/build_threshold_curve.py can apply that same rule without importing torch. Do not change
the rule here — Claude owns the logic in decide.py / product.py.
"""

from __future__ import annotations

SAFETY_SITUATIONS = {"accident_or_injury", "harassment_or_threat"}
SAFE_ACTIONS = ("sos", "contact_support")
DEFAULTS = {"safety_tau": 0.2, "low_conf": 0.6, "checkin_timeout_s": 30}


def safety_scores(ans: dict) -> dict:
    """The three safety signals and the combined check-in score, from answers()."""
    sit = ans["situation"]["probabilities"]
    p_sos = ans["action"]["probabilities"].get("sos", 0.0)
    p_safe = ans["safety"]["noul"]
    p_sit = sum(sit.get(s, 0.0) for s in SAFETY_SITUATIONS)
    return {"p_safety": p_safe, "p_sos": p_sos, "p_safety_situation": p_sit, "checkin_score": max(p_safe, p_sit, p_sos)}


def alert_level(ans: dict, th: dict) -> str:
    """'sos' | 'checkin' | 'none'. Without two-level thresholds: legacy single flag (sos when p(safety) >= safety_tau
    or a safety situation is top-1)."""
    sc = safety_scores(ans)
    if "sos_tau" in th and "checkin_tau" in th:
        if sc["p_sos"] >= th["sos_tau"]:
            return "sos"
        return "checkin" if sc["checkin_score"] >= th["checkin_tau"] else "none"
    tau = th.get("safety_tau", 0.2)
    if sc["p_safety"] >= tau or ans["situation"]["choice"] in SAFETY_SITUATIONS or ans["action"]["choice"] == "sos":
        return "sos"
    return "none"


def consistent(ans: dict, th) -> tuple[dict, dict]:
    """At the SOS level the action is restricted to {sos, contact_support}.
    th: thresholds dict (or, for old callers, a float safety_tau).
    Returns (answers with ans["action"]["choice"] possibly changed, report)."""
    th = {"safety_tau": th} if isinstance(th, (int, float)) else th
    level = alert_level(ans, th)
    flag = level == "sos"
    sit_safe = ans["situation"]["choice"] in SAFETY_SITUATIONS
    raw = ans["action"]["choice"]
    rep = {"level": level, "safety_flag": flag, "safety_situation": sit_safe, "raw_action": raw, "overridden": False,
           "disagree": flag and raw not in SAFE_ACTIONS, **safety_scores(ans)}
    if rep["disagree"]:
        p = ans["action"]["probabilities"]
        new = max(SAFE_ACTIONS, key=lambda a: p.get(a, 0.0))
        ans = {**ans, "action": {**ans["action"], "choice": new}}
        rep["overridden"] = True
    return ans, rep


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
