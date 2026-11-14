"""Real background noise for src/augment.py: MS-SNSD noise clips, sorted into the chain's categories, keeping
MS-SNSD's own train/test split (test clips only ever go into test tiers).

    python src/fetch_noise.py --out data/noise [--src <existing MS-SNSD checkout>]

Source: github.com/microsoft/MS-SNSD (noise_train/, noise_test/; 16 kHz wav). Licences: Freesound clips are CC0,
DEMAND clips are CC BY-SA 3.0 - both allow commercial use; credit DEMAND, and if the NOISY CLIPS themselves are ever
published, publish them under CC BY-SA. The trained head contains no audio. Only the two noise folders are fetched
(sparse checkout), not the clean-speech part of the repo.

Output layout read by augment.NoiseBank: <out>/<category>/{train,test}/*.wav. Your own road recordings can be added
to the same folders (put a few in test/ so test noise stays unseen).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

# MS-SNSD file-name keywords -> augment.py categories (first match wins; unmatched clips are skipped)
MAP = [
    (r"traffic|square|street|station|bus", "traffic"),
    (r"car|metro|engine|motor", "engine"),
    (r"babble|cafeteria|restaurant|neighbor|airportannounce|crowd", "babble"),
    (r"kitchen|washing|munching|dishes", "kitchen"),
    (r"wind|field|park", "wind"),
    (r"rain", "rain"),
]


def category(name: str) -> str | None:
    n = name.lower()
    for pat, cat in MAP:
        if re.search(pat, n):
            return cat
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=None, help="an existing MS-SNSD checkout (skips the download)")
    ap.add_argument("--max-per-category", type=int, default=60)
    a = ap.parse_args(argv)
    out = Path(a.out)
    src = Path(a.src) if a.src else out / "_ms-snsd"
    if not (src / "noise_train").exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/microsoft/MS-SNSD", str(src)], check=True)
        subprocess.run(["git", "-C", str(src), "sparse-checkout", "set", "noise_train", "noise_test"], check=True)
    counts = {}
    for split, folder in (("train", "noise_train"), ("test", "noise_test")):
        for p in sorted((src / folder).glob("*.wav")):
            cat = category(p.stem)
            if cat is None:
                continue
            key = (cat, split)
            if counts.get(key, 0) >= a.max_per_category:
                continue
            dst = out / cat / split / f"mssnsd_{p.name}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy(p, dst)
            counts[key] = counts.get(key, 0) + 1
    for (cat, split), n in sorted(counts.items()):
        print(f"[noise] {cat:8s} {split:5s} {n}")
    if not counts:
        raise SystemExit("[noise] no files matched - check the MS-SNSD layout")
    print(f"[noise] -> {out} (horn has no MS-SNSD source: augment.py keeps its synthetic horn for that category)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
