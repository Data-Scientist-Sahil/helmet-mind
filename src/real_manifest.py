"""Real recordings (labels.csv + audio folder) -> a manifest for src/extract.py, split = "real".

    python src/real_manifest.py --labels data/real/labels.csv --audio data/real/audio --out data/real/manifest.jsonl

labels.csv columns: clip_id, card_id, situation, action, safety, language, ambiguous, listener2_situation, release_ok
(audio file = <audio>/<clip_id>.<wav|m4a|mp3|ogg|opus|aac>; any format librosa can read - phone voice memos are fine).
Rows whose label is not in the spec fail loudly. Listener agreement (situation) is printed as Cohen's kappa.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.labels import ACTIONS, LANGUAGES, SITUATIONS  # noqa: E402

EXTS = (".wav", ".m4a", ".mp3", ".ogg", ".opus", ".aac", ".flac")


def truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def kappa(a, b):
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    audio = Path(a.audio)
    rows, errs, l1, l2 = [], [], [], []
    for r in csv.DictReader(open(a.labels, encoding="utf-8")):
        cid = r["clip_id"].strip()
        path = next((audio / f"{cid}{e}" for e in EXTS if (audio / f"{cid}{e}").exists()), None)
        if path is None:
            errs.append(f"{cid}: no audio file")
            continue
        s, act, lang = r["situation"].strip(), r["action"].strip(), r["language"].strip()
        for val, allowed, name in ((s, SITUATIONS, "situation"), (act, ACTIONS, "action"), (lang, LANGUAGES, "language")):
            if val not in allowed:
                errs.append(f"{cid}: {name}={val!r} not in spec")
        if r.get("listener2_situation", "").strip():
            l1.append(s)
            l2.append(r["listener2_situation"].strip())
        rows.append({"clip_id": f"real_{cid}", "path": str(path.resolve()), "split": "real", "situation": s, "action": act,
                     "safety": truthy(r["safety"]), "language": lang, "ambiguous": truthy(r.get("ambiguous", "")),
                     "card_id": r.get("card_id", ""), "release_ok": truthy(r.get("release_ok", "")), "tier": "real"})
    if errs:
        raise SystemExit("\n".join(errs))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[real] {len(rows)} clips -> {a.out}; safety=yes {sum(r['safety'] for r in rows)}; "
          f"ambiguous {sum(r['ambiguous'] for r in rows)}")
    if l1:
        print(f"[real] listener agreement on situation: {sum(x == y for x, y in zip(l1, l2))}/{len(l1)}, kappa={kappa(l1, l2):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
