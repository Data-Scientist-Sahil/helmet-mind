"""Shared by training, prediction and serving: features -> head logits -> calibrated probabilities -> answers.

Question order and option keys always come from spec/decision_set.json, so labels, logits and probabilities line up.

Scope: a milestone (e.g. "m1", the door-step wedge) only covers some situations. Out-of-scope options are masked to
-inf before the softmax - in training, calibration, prediction and serving alike - so the head cannot spend probability
on labels it never sees. The in-scope actions are derived from the action rules of the in-scope situations.

Two-level safety (round 3):
  SOS       p(action = sos) >= sos_tau                                   -> emergency card, human escalation
  check-in  safety score >= checkin_tau, where safety score =
            max(p(safety = yes), p(situation in {accident, harassment}), p(action = sos))
                                                                          -> "Are you safe?" one-tap card that
                                                                             escalates by itself if unanswered
Both thresholds are fitted on out-of-fold clips (train/calibrate.py). Runs without them (round 1/2) fall back to the
single safety_tau flag.

Consistency rule: when the SOS level fires the action is restricted to {sos, contact_support}. A check-in does not
rewrite the action (round 2 showed that rewriting on every low-threshold flag cost ~6 points of action accuracy).
The raw model action is always kept alongside.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.labels import options, spec  # noqa: E402

QUESTIONS = list(spec()["questions"])           # situation, action, safety, language
N_OPTIONS = {q: len(options(q)) for q in QUESTIONS}
SAFETY_SITUATIONS = set(spec()["safety_rules"]["always_yes_situations"])
SAFE_ACTIONS = ("sos", "contact_support")


# --------------------------------------------------------------------------- scope
def scope_for(name: str | None) -> dict | None:
    """{"situation": [...], "action": [...]} for a named scope in the spec, or None (= every option)."""
    if not name:
        return None
    sc = spec().get("scopes", {}).get(name)
    if sc is None:
        raise ValueError(f"unknown scope {name!r}; spec has {list(spec().get('scopes', {}))}")
    sits = list(sc["situations"])
    acts = set()
    for s in sits:
        r = spec()["action_rules"][s]
        acts.add(r["default"])
        acts.update(o["then"] for o in r["overrides"])
    return {"situation": [s for s in options("situation") if s in sits], "action": [a for a in options("action") if a in acts]}


def scope_mask(scope: dict | None) -> dict[str, torch.Tensor]:
    """question -> bool tensor (True = allowed)."""
    out = {}
    for q in QUESTIONS:
        keys = options(q)
        allowed = (scope or {}).get(q)
        out[q] = torch.tensor([allowed is None or k in allowed for k in keys])
    return out


def apply_mask(lg: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    return lg.masked_fill(~allowed.to(lg.device), float("-inf"))


# --------------------------------------------------------------------------- head
def gold_index(q: str, row: dict) -> int:
    """Index of the gold option for question q in a manifest/index row (-1 when the label is missing)."""
    v = row.get(q)
    if v is None:
        return -1
    if q == "safety":
        v = "true" if (v is True or str(v).lower() in ("true", "yes", "1")) else "false"
    keys = options(q)
    if v not in keys:
        raise ValueError(f"{row.get('clip_id')}: {q}={v!r} not in {keys}")
    return keys.index(v)


def head_items(rec: dict) -> list[dict]:
    """A cached feature record -> per-question items for mmclm.jh.collate (shared memory, zero digit prior)."""
    by_q = {it["qid"]: it for it in rec["items"]}
    out = []
    for q in QUESTIONS:
        it = by_q[q]
        out.append({"end": it["end"], "question": it["question"], "options": it["options"], "chunks": rec["memory"],
                    "prior": torch.zeros(it["options"].shape[0]), "exact_prior": False})
    return out


def head_logits(head, jh, recs: list[dict], device, scope: dict | None = None) -> list[dict[str, torch.Tensor]]:
    """Raw (T=1) logits per clip per question, on CPU, out-of-scope options at -inf."""
    items = [x for r in recs for x in head_items(r)]
    nq, masks = len(QUESTIONS), scope_mask(scope)
    lg_all = torch.cat([head(jh.collate(items[c0:c0 + 32], device)).float().cpu() for c0 in range(0, len(items), 32)])
    return [{q: apply_mask(lg_all[i * nq + j, : N_OPTIONS[q]], masks[q]) for j, q in enumerate(QUESTIONS)}
            for i in range(len(recs))]


# --------------------------------------------------------------------------- calibration
def temperature(lg: torch.Tensor, fit_q: dict | None) -> float:
    """T = exp(a + b*H + c*H^2 + d*log k) over the finite (in-scope) logits; H = normalised entropy at T=1."""
    if not fit_q:
        return 1.0
    a, b, c, d = (float(fit_q.get(x, 0.0)) for x in "abcd")
    x = lg[torch.isfinite(lg)].float()
    k = x.numel()
    lp = x.log_softmax(-1)
    h = float(-(lp.exp() * lp).sum()) / math.log(k) if k > 1 else 0.0
    return math.exp(a + b * h + c * h * h + d * math.log(max(k, 1)))


def fit_for(fit: dict | None, q: str) -> dict | None:
    """Pooled fit ({"a".."d"}) or per-question fit ({"per_question": {q: {"a".."d"}}})."""
    if not fit:
        return None
    if "per_question" in fit:
        return fit["per_question"].get(q)
    return fit


def calibrated_probs(logits: dict[str, torch.Tensor], fit: dict | None) -> dict[str, list[float]]:
    return {q: (lg / temperature(lg, fit_for(fit, q))).softmax(-1).tolist() for q, lg in logits.items()}


# --------------------------------------------------------------------------- answers + consistency
def answers(probs: dict[str, list[float]]) -> dict:
    """Decision Index style answers: choice -> {choice, probabilities}; yes/no -> {noul: p(yes)}."""
    out = {}
    for q, p in probs.items():
        keys = options(q)
        if spec()["questions"][q]["type"] == "noul":
            out[q] = {"type": "noul", "noul": p[0]}
        else:
            d = dict(zip(keys, p))
            out[q] = {"type": "choice", "choice": max(d, key=d.get), "probabilities": d}
    return out


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


def load_run(run_dir: str, device="cpu"):
    """head.pt (+ t_entropy.json, thresholds.json, config.json) from a training run directory.
    Returns (head, jh, fit, thresholds, config, scope)."""
    from mmclm import jh

    run = Path(run_dir)
    head = jh.load_head(str(run / "head.pt"), device=device)
    fit = json.loads((run / "t_entropy.json").read_text()) if (run / "t_entropy.json").exists() else None
    th = json.loads((run / "thresholds.json").read_text()) if (run / "thresholds.json").exists() else {}
    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    return head, jh, fit, th, cfg, cfg.get("scope")
