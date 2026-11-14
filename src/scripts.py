"""Rider scripts: slot grid -> (LLM) drafts -> lint + dedup -> gold labels + script-level split.

    python src/scripts.py grid  --out data/scripts/slots.jsonl [--per-situation 40]
    python src/scripts.py draft --slots data/scripts/slots.jsonl --out data/scripts/drafts.jsonl \
                                --backend openai --model <name> [--base-url https://api.groq.com/openai/v1]
    python src/scripts.py build --raw data/scripts/m1_wedge.raw.jsonl [more.jsonl ...] --out data/scripts/scripts.jsonl \
                                [--val 0.15 --test 0.15 --seed 0] [--freeze-splits data/scripts/m1_wedge.jsonl]

Optional row fields: "ambiguous": true (a native speaker could reasonably pick another label - kept, but reported
separately), "group": "<id>" (contrastive pairs/triples that differ only in the cue; a group always lands in ONE
split, so test measures new wordings, not a sibling of a training script). --freeze-splits keeps the split of every
script already in an earlier built file, so results stay comparable across rounds.

`build` is the only step that writes gold labels: action and safety come from src/labels.py rules, never from the
draft. It fails loudly on any lint error so nothing unchecked reaches TTS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.labels import LANGUAGES, SITUATIONS, label_row, spec  # noqa: E402

BANNED = ["zomato", "swiggy", "blinkit", "zepto", "dunzo", "uber", "ola", "rapido", "porter", "bigbasket", "instamart",
          "flipkart", "amazon"]
KNOWN_CUES = {"attempts", "waited_min", "rider_waiting", "not_started", "still_at_restaurant", "fixable", "left_restaurant",
              "near_restaurant", "fix_minutes", "cannot_proceed", "customer_fetching_money", "minor", "rider_ok",
              "rider_left_safe", "medical_emergency", "dangerous_place", "just_arrived"}
STAGES = {"pickup", "en_route", "drop", "any"}
REGISTERS = {"calm", "irritated", "panicked", "tired"}
KINDS = {"typical", "contrast", "terse"}
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(rows, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


# --------------------------------------------------------------------------- lint
def lint_row(r: dict) -> list[str]:
    e = []
    for k in ("id", "situation", "language", "stage", "register", "kind", "text", "text_tts"):
        if not r.get(k) and r.get(k) != {}:
            e.append(f"missing {k}")
    if e:
        return e
    if r["situation"] not in SITUATIONS:
        e.append(f"unknown situation {r['situation']}")
    if r["language"] not in LANGUAGES:
        e.append(f"unknown language {r['language']}")
    if r["stage"] not in STAGES:
        e.append(f"unknown stage {r['stage']}")
    if r["register"] not in REGISTERS:
        e.append(f"unknown register {r['register']}")
    if r["kind"] not in KINDS:
        e.append(f"unknown kind {r['kind']}")
    unknown = set(r.get("cues") or {}) - KNOWN_CUES
    if unknown:
        e.append(f"unknown cues {sorted(unknown)}")
    n = len(r["text"].split())
    if not 2 <= n <= 25:
        e.append(f"text has {n} words (2-25)")
    low = (r["text"] + " " + r["text_tts"]).lower()
    for b in BANNED:
        if re.search(rf"\b{b}\b", low):
            e.append(f"platform name '{b}'")
    if re.search(r"\d", r["text_tts"]):
        e.append("digits in text_tts (spell numbers out)")
    if re.search(r"\d{6,}", r["text"]):
        e.append("long digit run in text (phone/PIN?)")
    if r["language"] in ("hindi", "hinglish") and not DEVANAGARI.search(r["text_tts"]):
        e.append("hindi/hinglish text_tts should be Devanagari")
    if r["language"] == "english" and DEVANAGARI.search(r["text_tts"]):
        e.append("english text_tts should be Latin script")
    return e


def near_duplicates(rows, chars=25, words=8):
    """Pairs sharing an exact normalised substring of >= `chars` characters at word boundaries, or `words` words."""
    seen_w, dups = {}, []
    for r in rows:
        toks = norm(r["text"]).split()
        for i in range(0, max(0, len(toks) - words + 1)):
            g = " ".join(toks[i:i + words])
            if g in seen_w and seen_w[g] != r["id"]:
                dups.append((seen_w[g], r["id"], g))
            seen_w.setdefault(g, r["id"])
    full = {}
    for r in rows:
        t = norm(r["text"])
        if len(t) >= chars and t in full:
            dups.append((full[t], r["id"], t))
        full.setdefault(t, r["id"])
    return dups


# --------------------------------------------------------------------------- split
def assign_splits(rows, val=0.15, test=0.15, seed=0, frozen=None):
    """Script-level, stratified by (situation, language), deterministic (hash order). The unit is a contrastive group
    (or the script itself), so siblings never straddle splits. Every stratum keeps >= 1 training unit.
    frozen: {script_id: split} for scripts whose split must not change."""
    frozen = frozen or {}
    units = {}
    for r in rows:
        if r["id"] in frozen:
            r["split"] = frozen[r["id"]]
            continue
        units.setdefault(r.get("group") or r["id"], []).append(r)
    groups = {}
    for uid, members in units.items():
        groups.setdefault((members[0]["situation"], members[0]["language"]), []).append((uid, members))
    for g in groups.values():
        g.sort(key=lambda u: hashlib.sha1(f"{seed}:{u[0]}".encode()).hexdigest())
        n = len(g)
        nv = max(1, round(val * n)) if val > 0 and n >= 3 else 0
        nt = max(1, round(test * n)) if test > 0 and n >= 4 else 0
        while nv + nt > n - 1:  # keep at least one training script per group
            if nt >= nv and nt > 0:
                nt -= 1
            else:
                nv -= 1
        for i, (_, members) in enumerate(g):
            for r in members:
                r["split"] = "test" if i < nt else ("val" if i < nt + nv else "train")
    return rows


def build(raw_paths, out, val, test, seed, freeze=None):
    rows = [r for p in raw_paths for r in read_jsonl(p)]
    errs = {}
    ids = [r.get("id") for r in rows]
    for i in {x for x in ids if ids.count(x) > 1}:
        errs.setdefault(i, []).append("duplicate id")
    for r in rows:
        e = lint_row(r)
        if e:
            errs.setdefault(r.get("id"), []).extend(e)
    dups = near_duplicates(rows)
    for a, b, g in dups:
        errs.setdefault(b, []).append(f"near-duplicate of {a}: '{g}'")
    if errs:
        for k, v in errs.items():
            print(f"[scripts] {k}: {'; '.join(v)}")
        raise SystemExit(f"[scripts] {len(errs)} rows failed lint - fix the raw file(s) and re-run")
    for r in rows:
        if "ambiguous" in r and not isinstance(r["ambiguous"], bool):
            raise SystemExit(f"[scripts] {r['id']}: ambiguous must be true/false")
    frozen = {r["id"]: r["split"] for r in read_jsonl(freeze)} if freeze else None
    rows = [label_row(r) for r in rows]
    rows = assign_splits(rows, val, test, seed, frozen)
    write_jsonl(rows, out)
    summary = {}
    for r in rows:
        key = f"{r['situation']}|{r['split']}"
        summary[key] = summary.get(key, 0) + 1
    acts, langs, safe = {}, {}, 0
    for r in rows:
        acts[r["action"]] = acts.get(r["action"], 0) + 1
        langs[r["language"]] = langs.get(r["language"], 0) + 1
        safe += r["safety"]
    print(f"[scripts] {len(rows)} rows -> {out}  (spec {spec()['version']})")
    print(f"[scripts] actions {acts}")
    print(f"[scripts] languages {langs}; safety=yes {safe}; ambiguous {sum(bool(r.get('ambiguous')) for r in rows)}")
    for sp in ("train", "val", "test"):
        print(f"[scripts]   {sp}: safety=yes scripts {sum(1 for r in rows if r['split'] == sp and r['safety'])}")
    print(f"[scripts] splits {{train: {sum(r['split'] == 'train' for r in rows)}, val: {sum(r['split'] == 'val' for r in rows)}, "
          f"test: {sum(r['split'] == 'test' for r in rows)}}}")
    return rows


# --------------------------------------------------------------------------- grid + LLM drafting
CUE_VARIANTS = {  # situation -> list of (cues, share) covering every action rule
    "customer_unreachable": [({}, 0.4), ({"attempts": 3}, 0.2), ({"waited_min": 12}, 0.2), ({"rider_waiting": True, "waited_min": 1}, 0.2)],
    "wrong_address": [({}, 0.6), ({"attempts": 2}, 0.4)],
    "restaurant_delay": [({"waited_min": 8}, 0.5), ({"waited_min": 25}, 0.3), ({"not_started": True}, 0.2)],
    "order_issue_at_pickup": [({}, 0.4), ({"still_at_restaurant": True, "fixable": True}, 0.3), ({"left_restaurant": True, "near_restaurant": True}, 0.3)],
    "food_spilled_or_damaged": [({}, 0.5), ({"near_restaurant": True}, 0.5)],
    "vehicle_breakdown": [({}, 0.6), ({"fix_minutes": 5}, 0.4)],
    "road_blocked_or_weather": [({}, 0.5), ({"cannot_proceed": True}, 0.3), ({"cannot_proceed": True, "dangerous_place": True}, 0.2)],
    "payment_issue": [({}, 0.65), ({"customer_fetching_money": True}, 0.35)],
    "customer_dispute": [({}, 1.0)],
    "accident_or_injury": [({}, 0.65), ({"minor": True, "rider_ok": True}, 0.35)],
    "harassment_or_threat": [({}, 0.7), ({"rider_left_safe": True}, 0.3)],
    "other_or_unclear": [({}, 0.8), ({"medical_emergency": True}, 0.2)],
}
LANG_SHARE = {"hinglish": 0.4, "hindi": 0.35, "english": 0.25}
KIND_SHARE = {"typical": 0.6, "contrast": 0.2, "terse": 0.2}
REG_CYCLE = ["calm", "irritated", "tired", "calm", "irritated"]


def make_grid(per_situation=40):
    slots = []
    for s in SITUATIONS:
        n = per_situation
        langs = [l for l, w in LANG_SHARE.items() for _ in range(round(w * n))][:n]
        kinds = [k for k, w in KIND_SHARE.items() for _ in range(round(w * n))][:n]
        cues = [c for c, w in CUE_VARIANTS[s] for _ in range(round(w * n))][:n]
        while len(langs) < n:
            langs.append("hinglish")
        while len(kinds) < n:
            kinds.append("typical")
        while len(cues) < n:
            cues.append(CUE_VARIANTS[s][0][0])
        for i in range(n):
            reg = "panicked" if s in ("accident_or_injury", "harassment_or_threat") and i % 3 != 2 else REG_CYCLE[i % len(REG_CYCLE)]
            lang = langs[(i * 7) % n]
            slots.append({"id": f"{s[:4]}_{lang[:4]}_{i:03d}", "situation": s, "language": lang,
                          "kind": kinds[(i * 3) % n], "cues": cues[i], "register": reg})
    return slots


DRAFT_PROMPT = """You write short voice notes that Indian delivery riders send to rider support. Write ONE utterance per slot.

Situation meanings: {situations}

Rules:
- 2-25 words, natural spoken style, as a real rider would say it on the road. No brand or platform names, no real
  names, addresses or phone numbers.
- language "hinglish": Hindi grammar with at least two English words that are not common loanwords
  (loanwords: {loanwords}). "hindi": Hindi grammar, at most one English word outside those loanwords. "english":
  Indian English.
- The cues MUST be audible in the words (e.g. attempts=3 -> "teen baar call kiya"; waited_min=12 -> "baarah minute se").
- kind "terse": a broken fragment of 2-6 words. kind "contrast": sounds like another situation or sounds alarming but
  is not (e.g. someone else's accident causing traffic), while still being the given situation.
- "text": romanised for hindi/hinglish. "text_tts": the same words in Devanagari for hindi/hinglish (English words
  transliterated), Latin for english; numbers spelled out as words in text_tts.

Return a JSON list, one object per slot, with keys: id, text, text_tts, stage (pickup|en_route|drop|any).

Slots:
{slots}"""


def draft(slots_path, out_path, backend, model, base_url=None, batch=10):
    slots = read_jsonl(slots_path)
    done = {r["id"] for r in read_jsonl(out_path)} if Path(out_path).exists() else set()
    todo = [s for s in slots if s["id"] not in done]
    sit = "; ".join(f"{k}: {v}" for k, v in spec()["questions"]["situation"]["criteria"].items())
    loan = ", ".join(spec()["language_rule"]["loanwords_counted_as_hindi"])
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        prompt = DRAFT_PROMPT.format(situations=sit, loanwords=loan, slots=json.dumps(chunk, ensure_ascii=False))
        text = _llm(prompt, backend, model, base_url)
        m = re.search(r"\[.*\]", text, re.S)
        items = json.loads(m.group(0)) if m else []
        by = {s["id"]: s for s in chunk}
        with open(out_path, "a", encoding="utf-8") as f:
            for it in items:
                if it.get("id") in by:
                    f.write(json.dumps({**by[it["id"]], **{k: it[k] for k in ("text", "text_tts", "stage") if k in it}},
                                       ensure_ascii=False) + "\n")
        print(f"[scripts] drafted {min(i + batch, len(todo))}/{len(todo)}")


def _llm(prompt, backend, model, base_url=None):
    if backend == "mock":  # for tests: echoes a valid-looking row per slot
        ids = re.findall(r'"id": "([^"]+)"', prompt.split("Slots:")[-1])
        return json.dumps([{"id": i, "text": "customer phone nahi utha raha bhai", "text_tts": "कस्टमर फ़ोन नहीं उठा रहा भाई",
                            "stage": "drop"} for i in ids], ensure_ascii=False)
    import urllib.request

    if backend == "anthropic":
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", method="POST",
                                     headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                                              "content-type": "application/json"},
                                     data=json.dumps({"model": model, "max_tokens": 4000,
                                                      "messages": [{"role": "user", "content": prompt}]}).encode())
        return json.loads(urllib.request.urlopen(req, timeout=120).read())["content"][0]["text"]
    url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
    req = urllib.request.Request(url, method="POST", headers={"authorization": f"Bearer {key}", "content-type": "application/json"},
                                 data=json.dumps({"model": model, "temperature": 0.9,
                                                  "messages": [{"role": "user", "content": prompt}]}).encode())
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--out", required=True)
    g.add_argument("--per-situation", type=int, default=40)
    d = sub.add_parser("draft")
    d.add_argument("--slots", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--backend", default="openai", choices=["openai", "anthropic", "mock"])
    d.add_argument("--model", default="")
    d.add_argument("--base-url", default=None)
    b = sub.add_parser("build")
    b.add_argument("--raw", nargs="+", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--val", type=float, default=0.15)
    b.add_argument("--test", type=float, default=0.15)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--freeze-splits", default=None, help="built scripts file whose splits must be kept")
    a = ap.parse_args(argv)
    if a.cmd == "grid":
        s = make_grid(a.per_situation)
        write_jsonl(s, a.out)
        print(f"[scripts] {len(s)} slots -> {a.out}")
    elif a.cmd == "draft":
        draft(a.slots, a.out, a.backend, a.model, a.base_url)
    else:
        build(a.raw, a.out, a.val, a.test, a.seed, a.freeze_splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
