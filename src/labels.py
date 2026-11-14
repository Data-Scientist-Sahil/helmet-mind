"""The decision set and the rules that turn a script's structured cues into gold labels.

Gold `action` and `safety` are never hand-picked: they are derived from (situation, cues) by the rules in
spec/decision_set.json, so every label is reproducible and auditable.

    python src/labels.py          # self-test
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

SPEC_PATH = Path(__file__).resolve().parent.parent / "spec" / "decision_set.json"


@lru_cache(maxsize=1)
def spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def options(question: str) -> list[str]:
    q = spec()["questions"][question]
    return ["true", "false"] if q["type"] == "noul" else list(q["criteria"])


SITUATIONS = options("situation")
ACTIONS = options("action")
LANGUAGES = options("language")


def _match(cond: dict, cues: dict) -> bool:
    """A condition like {"attempts_gte": 2, "rider_waiting": true} holds when every clause holds."""
    for key, want in cond.items():
        for suffix, op in (("_gte", lambda a, b: a >= b), ("_lte", lambda a, b: a <= b), ("_lt", lambda a, b: a < b)):
            if key.endswith(suffix):
                have = cues.get(key[: -len(suffix)])
                if have is None or not op(have, want):
                    return False
                break
        else:
            if cues.get(key) != want:
                return False
    return True


def derive_action(situation: str, cues: dict | None = None) -> str:
    """First matching override wins, else the situation's default."""
    if situation not in SITUATIONS:
        raise ValueError(f"unknown situation {situation!r}")
    rule, cues = spec()["action_rules"][situation], cues or {}
    for o in rule["overrides"]:
        if _match(o["if"], cues):
            return o["then"]
    return rule["default"]


def derive_safety(situation: str, cues: dict | None = None) -> bool:
    r, cues = spec()["safety_rules"], cues or {}
    return situation in r["always_yes_situations"] or any(cues.get(c) for c in r["yes_if_cue"])


_WORD = re.compile(r"[a-zA-Z]+")


def hindi_or_hinglish(romanised: str, english_words: set[str]) -> str:
    """Helper for labelling romanised Hindi-grammar text: counts English words outside the loanword whitelist.
    `english_words` is the set of tokens the author marks as English (the LLM drafting step supplies it)."""
    rule = spec()["language_rule"]
    loan = set(rule["loanwords_counted_as_hindi"])
    other = [w for w in _WORD.findall(romanised.lower()) if w in english_words and w not in loan]
    return "hinglish" if len(other) >= rule["hinglish_min_other_english_words"] else "hindi"


def label_row(row: dict) -> dict:
    """Fill gold labels for one script row (see data/scripts/README.md for the schema)."""
    s, cues = row["situation"], row.get("cues") or {}
    if row.get("language") not in LANGUAGES:
        raise ValueError(f"{row.get('id')}: unknown language {row.get('language')!r}")
    return {**row, "action": derive_action(s, cues), "safety": derive_safety(s, cues)}


# --------------------------------------------------------------------------- self-test
def _selftest() -> None:
    sp = spec()
    assert len(SITUATIONS) == 12 and len(ACTIONS) == 6 and len(LANGUAGES) == 3, (SITUATIONS, ACTIONS, LANGUAGES)
    assert set(sp["action_rules"]) == set(SITUATIONS), "every situation needs an action rule"
    reachable = set()
    for s, r in sp["action_rules"].items():
        assert r["default"] in ACTIONS, (s, r["default"])
        reachable.add(r["default"])
        for o in r["overrides"]:
            assert o["then"] in ACTIONS, (s, o)
            reachable.add(o["then"])
    assert reachable == set(ACTIONS), f"unreachable actions: {set(ACTIONS) - reachable}"

    cases = [  # (situation, cues, action, safety) — the seed utterances and the edge cases from the spec
        ("customer_unreachable", {}, "call_customer", False),                        # "gate band hai, phone nahi utha raha"
        ("customer_unreachable", {"attempts": 3, "waited_min": 12}, "contact_support", False),
        ("customer_unreachable", {"rider_waiting": True, "waited_min": 2}, "wait", False),
        ("restaurant_delay", {"waited_min": 20}, "request_reassign", False),          # "20 minute aur lagega"
        ("restaurant_delay", {"waited_min": 5}, "wait", False),
        ("food_spilled_or_damaged", {"near_restaurant": True}, "return_to_restaurant", False),
        ("order_issue_at_pickup", {"left_restaurant": True, "near_restaurant": True}, "return_to_restaurant", False),
        ("order_issue_at_pickup", {"still_at_restaurant": True, "fixable": True}, "wait", False),
        ("vehicle_breakdown", {}, "request_reassign", False),                        # puncture, no shop nearby
        ("vehicle_breakdown", {"fix_minutes": 5}, "contact_support", False),
        ("harassment_or_threat", {}, "sos", True),                                   # "dhamki de rahe hain"
        ("harassment_or_threat", {"rider_left_safe": True}, "contact_support", True),  # safety stays yes
        ("accident_or_injury", {"minor": True, "rider_ok": True}, "contact_support", True),
        ("road_blocked_or_weather", {}, "wait", False),                              # "aage accident hua hai" (not the rider)
        ("road_blocked_or_weather", {"dangerous_place": True, "cannot_proceed": True}, "contact_support", True),
        ("other_or_unclear", {"medical_emergency": True}, "contact_support", True),
        ("payment_issue", {"customer_fetching_money": True}, "wait", False),
    ]
    for s, cues, a, safe in cases:
        assert derive_action(s, cues) == a, (s, cues, derive_action(s, cues), a)
        assert derive_safety(s, cues) == safe, (s, cues)

    assert hindi_or_hinglish("customer phone nahi utha raha", {"customer", "phone"}) == "hindi"
    assert hindi_or_hinglish("customer is not responding, kya karun", {"customer", "is", "not", "responding"}) == "hinglish"
    print(f"labels self-test OK: {len(SITUATIONS)} situations, {len(ACTIONS)} actions (all reachable), "
          f"{len(LANGUAGES)} languages, {len(cases)} rule cases")


if __name__ == "__main__":
    _selftest()
