"""Fit the serving temperature and the safety threshold for a run - never on test clips.

    python train/calibrate.py --run runs/m1 [--logits auto|oof|val] [--recall 0.98] [--script-floor 0.75]

Which logits:  `oof` = out-of-fold logits over train+val scripts from train/crossfit.py (oof_logits.pt; many more
               clips and safety scripts), `val` = the run's validation logits. `auto` uses oof when present.

1. Temperature, PER QUESTION (round 1 used one pooled fit, which made action calibration worse on val):
   for each question, a scalar T and the entropy form T = exp(a + b*H + c*H^2 + d*log k) are fitted by NLL; the
   entropy form is only allowed with >= 300 calibration clips (it overfits small sets), otherwise the scalar is used.
   Neither changes the argmax. Out-of-scope options (masked to -inf) are ignored.
2. Safety threshold tau - the largest tau such that
     (a) clip-level recall >= --recall, and
     (b) every safety=yes SCRIPT has at least --script-floor of its clips flagged
   (all renders/noise tiers of one script are near-duplicates, so (b) stops one easy script from carrying the rate),
   clipped to [--tau-min, --tau-max]. Wilson CIs are reported at clip level and at script level, with false alarms
   per script. Fewer than 20 safety scripts -> tau is flagged as rough.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import metrics as M  # noqa: E402
from src.decide import QUESTIONS, gold_index, temperature  # noqa: E402

MIN_CLIPS_ENTROPY_FORM = 300


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def _finite(lg):
    return lg[torch.isfinite(lg)].float()


def compress(lg, g):
    """Drop masked (-inf) options and remap the gold index: masked entries give NaN gradients w.r.t. T."""
    keep = torch.isfinite(lg)
    return lg[keep].float(), int(keep[:g].sum())


def _nll(lg, g, fit_q):
    return -float((lg.float() / temperature(lg, fit_q)).log_softmax(-1)[g])


def fit_scalar(pairs):
    best = (1.0, float("inf"))
    for t in [x / 20 for x in range(4, 101)]:  # 0.2 .. 5.0
        v = sum(-float((lg.float() / t).log_softmax(-1)[g]) for lg, g in pairs) / len(pairs)
        if v < best[1]:
            best = (t, v)
    return {"a": math.log(best[0]), "b": 0.0, "c": 0.0, "d": 0.0, "form": "scalar", "T": best[0]}, best[1]


def fit_entropy(pairs, init=None, steps=200):
    def ent(lg):
        x = _finite(lg)
        lp = x.log_softmax(-1)
        return float(-(lp.exp() * lp).sum()) / math.log(x.numel()) if x.numel() > 1 else 0.0

    H = torch.tensor([ent(lg) for lg, _ in pairs])
    logk = torch.tensor([math.log(_finite(lg).numel()) for lg, _ in pairs])
    th = torch.tensor([float((init or {}).get(x, 0.0)) for x in "abcd"], requires_grad=True)
    opt = torch.optim.LBFGS([th], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def nll_of(t):
        T = torch.exp(t[0] + t[1] * H + t[2] * H * H + t[3] * logk)
        return torch.stack([-(lg.float() / T[i]).log_softmax(-1)[g] for i, (lg, g) in enumerate(pairs)]).mean()

    def closure():
        opt.zero_grad()
        loss = nll_of(th) + 1e-3 * (th ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return {**{k: float(v) for k, v in zip("abcd", th)}, "form": "entropy"}, float(nll_of(th))


def choose_tau(scores, labels, scripts, recall, script_floor, tau_min, tau_max):
    pos = [(s, sc) for s, l, sc in zip(scores, labels, scripts) if l]
    if not pos:
        return tau_max, "no safety=yes clips: tau = tau_max"
    cands = sorted({s for s, _ in pos}, reverse=True)
    by_script = defaultdict(list)
    for s, sc in pos:
        by_script[sc].append(s)
    need = math.ceil(recall * len(pos) - 1e-9)
    for t in cands:  # descending: the first tau that satisfies both is the largest
        caught = sum(s >= t for s, _ in pos)
        ok_scripts = all(sum(x >= t for x in v) / len(v) >= script_floor for v in by_script.values())
        if caught >= need and ok_scripts:
            return min(tau_max, max(tau_min, t - 1e-6)), "recall + per-script floor"
    return max(tau_min, min(cands) - 1e-6), "fallback: lowest positive score"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--features", default=None, help="features dir (default: from the run's config)")
    ap.add_argument("--logits", default="auto", choices=["auto", "oof", "val"])
    ap.add_argument("--init-fit", default=None, help="released t_entropy.json, as the starting point of the entropy fit")
    ap.add_argument("--recall", type=float, default=0.98)
    ap.add_argument("--script-floor", type=float, default=0.75)
    ap.add_argument("--tau-min", type=float, default=0.02)
    ap.add_argument("--tau-max", type=float, default=0.5)
    ap.add_argument("--low-conf", type=float, default=0.6)
    a = ap.parse_args(argv)

    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    feat = Path(a.features or cfg["features"])
    src = a.logits
    if src == "auto":
        src = "oof" if (run / "oof_logits.pt").exists() else "val"
    blob = torch.load(run / ("oof_logits.pt" if src == "oof" else "val_logits.pt"), map_location="cpu", weights_only=True)
    by_id = {r["clip_id"]: r for r in read_jsonl(feat / "index.jsonl")}
    rows = [by_id[c] for c in blob["clip_ids"]]
    logits = blob["logits"]
    init = json.loads(Path(a.init_fit).read_text()) if a.init_fit else None

    per_q, report_q = {}, {}
    for q in QUESTIONS:
        pairs = [compress(lg[q], gold_index(q, r)) for lg, r in zip(logits, rows) if gold_index(q, r) >= 0]
        if not pairs:
            continue
        nll1 = sum(_nll(lg, g, None) for lg, g in pairs) / len(pairs)
        sc, sc_nll = fit_scalar(pairs)
        choice, choice_nll = sc, sc_nll
        ent_nll = None
        if len(pairs) >= MIN_CLIPS_ENTROPY_FORM:
            try:
                en, ent_nll = fit_entropy(pairs, init)
            except RuntimeError as e:  # optimiser blew up: keep the scalar fit
                print(f"[calibrate]   {q}: entropy fit failed ({e}); using the scalar T")
                en, ent_nll = None, None
            if ent_nll is not None and math.isfinite(ent_nll) and ent_nll < sc_nll:
                choice, choice_nll = en, ent_nll
        per_q[q] = choice

        def ece_with(f):
            g = [p[1] for p in pairs]
            pr = [(lg.float() / temperature(lg, f)).softmax(-1).tolist() for lg, _ in pairs]
            return M.ece(g, pr)

        report_q[q] = {"n": len(pairs), "nll_T1": nll1, "nll_scalar": sc_nll, "nll_entropy": ent_nll, "chosen": choice["form"],
                       "ece_T1": ece_with(None), "ece_chosen": ece_with(choice)}
    fit = {"mode": "entropy_per_question", "per_question": per_q, "fit": f"{src}_nll_per_question"}
    (run / "t_entropy.json").write_text(json.dumps(fit, indent=1))

    # safety threshold on calibrated p(yes)
    scores, labels, scripts = [], [], []
    for lg, r in zip(logits, rows):
        gi = gold_index("safety", r)
        if gi >= 0:
            x = lg["safety"].float()
            scores.append(float((x / temperature(x, per_q.get("safety"))).softmax(-1)[0]))
            labels.append(gi == 0)
            scripts.append(r.get("script_id", r["clip_id"]))
    tau, rule = choose_tau(scores, labels, scripts, a.recall, a.script_floor, a.tau_min, a.tau_max)
    npos, nneg = sum(labels), len(labels) - sum(labels)
    caught = sum(1 for s, l in zip(scores, labels) if l and s >= tau)
    fa = sum(1 for s, l in zip(scores, labels) if not l and s >= tau)
    pos_scripts = sorted({sc for sc, l in zip(scripts, labels) if l})
    script_rate = {sc: sum(1 for s, l, x in zip(scores, labels, scripts) if x == sc and s >= tau) /
                   sum(1 for x in scripts if x == sc) for sc in set(scripts)}
    full = sum(script_rate[sc] == 1.0 for sc in pos_scripts)
    fa_scripts = sorted(((sc, r) for sc, r in script_rate.items() if sc not in pos_scripts and r > 0), key=lambda x: -x[1])
    th = {"safety_tau": tau, "rule": rule, "low_conf": a.low_conf, "target_recall": a.recall, "script_floor": a.script_floor,
          "source": src, "n_pos": npos, "n_neg": nneg, "n_pos_scripts": len(pos_scripts),
          "recall": caught / npos if npos else None, "recall_ci": M.wilson(caught, npos) if npos else None,
          "scripts_fully_caught": full, "scripts_fully_caught_ci": M.wilson(full, len(pos_scripts)) if pos_scripts else None,
          "false_alarm_rate": fa / nneg if nneg else None, "false_alarm_ci": M.wilson(fa, nneg) if nneg else None,
          "false_alarm_scripts": [{"script_id": s, "flag_rate": round(r, 3)} for s, r in fa_scripts[:15]],
          "rough": len(pos_scripts) < 20}
    (run / "thresholds.json").write_text(json.dumps(th, indent=1))
    (run / "calibration.json").write_text(json.dumps({"source": src, "per_question": report_q, "thresholds": th}, indent=1))

    print(f"[calibrate] logits: {src} ({len(rows)} clips)")
    for q, v in report_q.items():
        print(f"[calibrate]   {q:9s} n={v['n']:4d} {v['chosen']:7s} ECE {v['ece_T1']:.3f} -> {v['ece_chosen']:.3f}   "
              f"NLL {v['nll_T1']:.3f} -> {min(x for x in (v['nll_scalar'], v['nll_entropy']) if x is not None):.3f}")
    print(f"[calibrate] safety tau={tau:.4f} ({rule}): recall {caught}/{npos} CI {th['recall_ci']}, "
          f"scripts fully caught {full}/{len(pos_scripts)}, false alarms {fa}/{nneg}")
    if th["rough"]:
        print(f"[calibrate] note: only {len(pos_scripts)} safety=yes scripts - tau is rough (aim for 20+; run train/crossfit.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
