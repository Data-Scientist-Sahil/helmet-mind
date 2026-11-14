"""predictions.jsonl files -> metrics.json + REPORT.md. The first system is the reference for paired comparisons.

    python train/evaluate.py --pred ours=runs/m1/pred_test.jsonl zeroshot=runs/zeroshot/pred_test.jsonl \
        --out results/m1 [--group tier language]

Every interval resamples SCRIPTS (cluster bootstrap): all renders and noise tiers of one script are near-duplicates,
so a clip-level bootstrap would look far more certain than the data is.

Per question: accuracy (+ script-level CI), macro-F1 over the classes PRESENT in the gold labels (the all-label
version is kept as macro_f1_all_labels), per-class recall/precision, NLL, Brier, ECE, confusion matrix.
Action: raw model action and action after the safety consistency rule; how often the rule had to step in.
Safety: recall and false-alarm rate at tau (Wilson CIs, clip level and script level), escalation recall of the
product card, AP, AUROC, every missed safety clip, and false alarms per script.
Per script: accuracy per question and safety flag rate - the worst scripts are listed in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import metrics as M  # noqa: E402
from src.decide import QUESTIONS, gold_index  # noqa: E402
from src.labels import options  # noqa: E402


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def script_of(r):
    return r.get("script_id") or r["clip_id"]


def question_metrics(rows, q, n_boot):
    keys = options(q)
    sel = [r for r in rows if gold_index(q, r) >= 0]
    if not sel:
        return None
    g = [gold_index(q, r) for r in sel]
    pr = [r["probs"][q] for r in sel]
    pred = [argmax(p) for p in pr]
    corr = [int(a == b) for a, b in zip(g, pred)]
    present = sorted(set(g))
    per_class = {}
    for c in present:
        tp = sum(1 for a, b in zip(g, pred) if a == c and b == c)
        npred = sum(1 for b in pred if b == c)
        per_class[keys[c]] = {"n": g.count(c), "recall": tp / g.count(c), "precision": tp / npred if npred else None,
                              "predicted": npred}
    out = {"n": len(g), "accuracy": sum(corr) / len(corr),
           "accuracy_ci": M.cluster_bootstrap(corr, [script_of(r) for r in sel], n_boot),
           "macro_f1": M.macro_f1(g, pred, present), "macro_f1_all_labels": M.macro_f1(g, pred, list(range(len(keys)))),
           "present_labels": [keys[c] for c in present], "per_class": per_class,
           "predicted_out_of_gold": {keys[c]: pred.count(c) for c in set(pred) - set(present)},
           "nll": M.nll(g, pr), "brier": M.brier(g, pr), "ece": M.ece(g, pr),
           "skill": M.chance_skill(sum(corr) / len(corr), 1 / max(1, sum(1 for x in pr[0] if x > 0))),
           "confusion": M.confusion(g, pred, list(range(len(keys)))), "labels": keys,
           "conf_auroc": M.auroc([max(p) for p in pr], corr), "risk_coverage": M.risk_coverage(g, pr),
           "reliability": M.reliability(g, pr)}
    amb = [r for r in sel if "ambiguous" in r]
    if amb and any(r["ambiguous"] for r in amb) and any(not r["ambiguous"] for r in amb):
        ent = lambda s: sum(M.norm_entropy(r["probs"][q]) for r in s) / len(s)  # noqa: E731
        out["entropy_ambiguous"] = ent([r for r in amb if r["ambiguous"]])
        out["entropy_clear"] = ent([r for r in amb if not r["ambiguous"]])
    if q == "action" and all("answers" in r for r in sel):
        after = [int(r["answers"]["action"]["choice"] == keys[gi]) for r, gi in zip(sel, g)]
        out["accuracy_after_rule"] = sum(after) / len(after)
        out["rule_overrides"] = sum(1 for r in sel if (r.get("consistency") or {}).get("overridden"))
        sos = keys.index("sos")
        s_idx = [i for i, gi in enumerate(g) if gi == sos]
        if s_idx:
            out["sos_recall_after_rule_or_card"] = sum(
                1 for i in s_idx if sel[i]["answers"]["action"]["choice"] == "sos" or sel[i]["card"]["kind"] == "sos") / len(s_idx)
    return out


def safety_metrics(rows):
    sel = [r for r in rows if gold_index("safety", r) >= 0]
    if not sel:
        return None
    y = [gold_index("safety", r) == 0 for r in sel]
    p = [r["probs"]["safety"][0] for r in sel]
    tau = sel[0].get("safety_tau", 0.5)
    hit = [s >= tau for s in p]
    sos = [r["card"]["kind"] == "sos" for r in sel]
    npos, nneg = sum(y), len(y) - sum(y)
    tp = sum(1 for a, b in zip(y, hit) if a and b)
    fp = sum(1 for a, b in zip(y, hit) if not a and b)
    etp = sum(1 for a, b in zip(y, sos) if a and b)
    efp = sum(1 for a, b in zip(y, sos) if not a and b)
    by = defaultdict(list)
    for r, a, h in zip(sel, y, hit):
        by[(script_of(r), a)].append(h)
    pos_scripts = [v for (s, a), v in by.items() if a]
    neg_scripts = {s: v for (s, a), v in by.items() if not a}
    sweep = []
    for t in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        sweep.append({"tau": t, "recall": sum(1 for a, s in zip(y, p) if a and s >= t) / npos if npos else None,
                      "false_alarm": sum(1 for a, s in zip(y, p) if not a and s >= t) / nneg if nneg else None})
    return {"n_pos": npos, "n_neg": nneg, "n_pos_scripts": len(pos_scripts), "tau": tau,
            "recall": tp / npos if npos else None, "recall_ci": M.wilson(tp, npos) if npos else None,
            "scripts_fully_caught": sum(all(v) for v in pos_scripts),
            "false_alarm_rate": fp / nneg if nneg else None, "false_alarm_ci": M.wilson(fp, nneg) if nneg else None,
            "escalation_recall": etp / npos if npos else None, "escalation_recall_ci": M.wilson(etp, npos) if npos else None,
            "escalation_false_alarm_rate": efp / nneg if nneg else None,
            "average_precision": M.average_precision(p, y), "auroc": M.auroc(p, y),
            "false_alarms_by_script": sorted(({"script_id": s, "flagged": sum(v), "n": len(v)} for s, v in neg_scripts.items() if any(v)),
                                             key=lambda x: -x["flagged"]),
            "sweep_diagnostic_only": sweep,
            "missed": [{"clip_id": r["clip_id"], "situation": r.get("situation"), "p_safety": round(s, 4), "card": r["card"]["kind"]}
                       for r, a, s, e in zip(sel, y, p, sos) if a and not e]}


def per_script(rows):
    by = defaultdict(list)
    for r in rows:
        by[script_of(r)].append(r)
    out = []
    for s, rs in by.items():
        d = {"script_id": s, "n": len(rs), "situation": rs[0].get("situation"), "action": rs[0].get("action"),
             "language": rs[0].get("language"), "safety": bool(rs[0].get("safety"))}
        for q in ("situation", "action"):
            gi = gold_index(q, rs[0])
            d[f"{q}_acc"] = sum(argmax(r["probs"][q]) == gi for r in rs) / len(rs)
            wrong = [options(q)[argmax(r["probs"][q])] for r in rs if argmax(r["probs"][q]) != gi]
            d[f"{q}_top_error"] = max(set(wrong), key=wrong.count) if wrong else None
        tau = rs[0].get("safety_tau", 0.5)
        d["safety_flag_rate"] = sum(r["probs"]["safety"][0] >= tau for r in rs) / len(rs)
        out.append(d)
    return sorted(out, key=lambda d: d["action_acc"] + d["situation_acc"])


def system_metrics(rows, n_boot):
    return {"n": len(rows), "n_scripts": len({script_of(r) for r in rows}),
            **{q: question_metrics(rows, q, n_boot) for q in QUESTIONS}, "safety_flag": safety_metrics(rows)}


def correct_vector(rows, q):
    return {r["clip_id"]: (int(argmax(r["probs"][q]) == gold_index(q, r)), script_of(r)) for r in rows if gold_index(q, r) >= 0}


def fmt(x, pct=True):
    if x is None:
        return "—"
    if isinstance(x, dict):
        return f"[{fmt(x['lo'], pct)}, {fmt(x['hi'], pct)}]"
    if isinstance(x, (list, tuple)):
        return f"[{fmt(x[0], pct)}, {fmt(x[1], pct)}]"
    return f"{100 * x:.1f}" if pct else f"{x:.3f}"


def report_md(res, systems, groups):
    L = ["# Results", "", f"Systems: {', '.join(systems)}. Intervals resample scripts (cluster bootstrap) or are Wilson 95%.", ""]
    L += ["## Overall", "", "| System | clips / scripts | Situation acc [CI] | Situation macro-F1 (present) | Action acc [CI] | Action after rule | Language acc | ECE sit / act |",
          "|---|---|---|---|---|---|---|---|"]
    for s in systems:
        m = res["systems"][s]["all"]
        g = lambda q, k: (m.get(q) or {}).get(k)  # noqa: E731
        L.append(f"| {s} | {m['n']} / {m['n_scripts']} | {fmt(g('situation', 'accuracy'))} {fmt(g('situation', 'accuracy_ci'))} | "
                 f"{fmt(g('situation', 'macro_f1'))} | {fmt(g('action', 'accuracy'))} {fmt(g('action', 'accuracy_ci'))} | "
                 f"{fmt(g('action', 'accuracy_after_rule'))} | {fmt(g('language', 'accuracy'))} | "
                 f"{fmt(g('situation', 'ece'), False)} / {fmt(g('action', 'ece'), False)} |")
    L += ["", "## Action recall per class", "", "| System | " + " | ".join(options("action")) + " |", "|---" * (len(options("action")) + 1) + "|"]
    for s in systems:
        pc = (res["systems"][s]["all"].get("action") or {}).get("per_class", {})
        L.append(f"| {s} | " + " | ".join(f"{fmt(pc[a]['recall'])} ({pc[a]['predicted']}/{pc[a]['n']})" if a in pc else "—"
                                          for a in options("action")) + " |")
    L += ["", "## Situation recall per class (present in test)", ""]
    for s in systems:
        pc = (res["systems"][s]["all"].get("situation") or {}).get("per_class", {})
        L.append(f"- **{s}**: " + ", ".join(f"{k} {fmt(v['recall'])}" for k, v in pc.items()))
    L += ["", "## Safety", "", "| System | τ | Recall [CI] | Scripts fully caught | Escalation recall (card) | False alarm [CI] | AUROC |", "|---|---|---|---|---|---|---|"]
    for s in systems:
        sf = res["systems"][s]["all"]["safety_flag"] or {}
        L.append(f"| {s} | {sf.get('tau', 0):.3f} | {fmt(sf.get('recall'))} {fmt(sf.get('recall_ci'))} | {sf.get('scripts_fully_caught')}/{sf.get('n_pos_scripts')} | "
                 f"{fmt(sf.get('escalation_recall'))} | {fmt(sf.get('false_alarm_rate'))} {fmt(sf.get('false_alarm_ci'))} | {fmt(sf.get('auroc'), False)} |")
    ref = res["systems"][systems[0]]
    fas = (ref["all"]["safety_flag"] or {}).get("false_alarms_by_script", [])[:8]
    if fas:
        L += ["", f"False alarms by script ({systems[0]}): " + ", ".join(f"`{x['script_id']}` {x['flagged']}/{x['n']}" for x in fas)]
    for grp in groups:
        L += ["", f"## By {grp}", "", f"| System | {grp} | n | Situation acc | Action acc | Safety recall |", "|---|---|---|---|---|---|"]
        for s in systems:
            for val, m in sorted(res["systems"][s].get(f"by_{grp}", {}).items()):
                sf = m["safety_flag"] or {}
                L.append(f"| {s} | {val} | {m['n']} | {fmt((m.get('situation') or {}).get('accuracy'))} | "
                         f"{fmt((m.get('action') or {}).get('accuracy'))} | {fmt(sf.get('recall'))} |")
    if res.get("paired"):
        L += ["", f"## Paired difference vs {systems[0]} (script-level bootstrap)", "", "| System | Question | clips / scripts | Δ | 95% CI |", "|---|---|---|---|---|"]
        for s, d in res["paired"].items():
            for q, v in d.items():
                L.append(f"| {s} | {q} | {v['n']} / {v['n_clusters']} | {fmt(v['diff'])} | [{fmt(v['lo'])}, {fmt(v['hi'])}] |")
    worst = ref["per_script"][:10]
    L += ["", f"## Weakest scripts ({systems[0]})", "", "| Script | gold situation / action | Situation acc | Action acc | Most common wrong action | Safety flag rate |", "|---|---|---|---|---|---|"]
    for d in worst:
        L.append(f"| `{d['script_id']}` | {d['situation']} / {d['action']} | {fmt(d['situation_acc'])} | {fmt(d['action_acc'])} | "
                 f"{d['action_top_error'] or '—'} | {fmt(d['safety_flag_rate'])} |")
    missed = (ref["all"]["safety_flag"] or {}).get("missed", [])
    if missed:
        L += ["", f"## Safety clips the product card missed ({systems[0]})", ""] + [f"- `{x['clip_id']}` {x['situation']} p={x['p_safety']}" for x in missed]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True, help="name=path.jsonl ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", nargs="*", default=["tier", "language"])
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args(argv)

    systems, data = [], {}
    for spec_ in a.pred:
        name, path = spec_.split("=", 1)
        systems.append(name)
        data[name] = read_jsonl(path)
    res = {"systems": {}, "paired": {}}
    for s in systems:
        rows = data[s]
        res["systems"][s] = {"all": system_metrics(rows, a.bootstrap), "per_script": per_script(rows)}
        for grp in a.group:
            vals = sorted({str(r.get(grp)) for r in rows if r.get(grp) is not None})
            if vals:
                res["systems"][s][f"by_{grp}"] = {v: system_metrics([r for r in rows if str(r.get(grp)) == v], 200) for v in vals}
    ref = systems[0]
    for s in systems[1:]:
        res["paired"][s] = {}
        for q in ("situation", "action"):
            ca, cb = correct_vector(data[ref], q), correct_vector(data[s], q)
            common = sorted(set(ca) & set(cb))
            if len(common) >= 5:
                bs = M.paired_cluster_bootstrap([cb[c][0] for c in common], [ca[c][0] for c in common],
                                                [ca[c][1] for c in common], n=a.bootstrap)
                res["paired"][s][q] = {"n": len(common), **bs}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    md = report_md(res, systems, a.group)
    (out / "REPORT.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
