#!/usr/bin/env python3
"""Drop a trained run into run/ so the Space can load real weights.

    # from an already-extracted run folder
    python scripts/load_run.py --run /path/to/out/runs/m1v3
    # from a Kaggle / notebook results zip
    python scripts/load_run.py --run-zip results/round3_training_results.zip --run-name m1v3 \
        --thresholds thresholds.json

Does not download anything and does not push a Space. Safe to re-run (overwrites run/).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_FILES = ["head.pt", "config.json", "t_entropy.json", "thresholds.json"]


def extract_zip(zip_path: Path, run_name: str, dst: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        for f in RUN_FILES:
            hits = [n for n in names if n.endswith(f"/runs/{run_name}/{f}")]
            if not hits:
                # also accept <run_name>/<file> at any depth
                hits = [n for n in names if n.endswith(f"/{run_name}/{f}") or n.endswith(f"/{f}") and f"/{run_name}/" in n]
            if hits:
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(hits[0]) as r, open(dst / f, "wb") as w:
                    shutil.copyfileobj(r, w)
                print(f"  {f} <- {hits[0]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="run folder with head.pt, config.json, t_entropy.json, thresholds.json")
    g.add_argument("--run-zip", help="results zip containing out/runs/<run-name>/...")
    ap.add_argument("--run-name", default="m1v3")
    ap.add_argument("--thresholds", default=str(ROOT / "thresholds.json"),
                    help="thresholds json merged over the run's (round-3 two-level by default)")
    ap.add_argument("--out", default=str(ROOT / "run"))
    a = ap.parse_args(argv)

    dst = Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    if a.run:
        src = Path(a.run)
        for f in RUN_FILES:
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
                print(f"  {f} <- {src / f}")
    else:
        extract_zip(Path(a.run_zip), a.run_name, dst)

    missing = [f for f in ["head.pt", "config.json"] if not (dst / f).exists()]
    if missing:
        sys.exit(f"run is missing {missing}")

    th = json.loads((dst / "thresholds.json").read_text()) if (dst / "thresholds.json").exists() else {}
    if a.thresholds and Path(a.thresholds).exists():
        th.update(json.loads(Path(a.thresholds).read_text()))
    th.pop("false_alarm_scripts", None)
    (dst / "thresholds.json").write_text(json.dumps(th, indent=1) + "\n")
    print(f"ready: {dst}/  sos_tau={th.get('sos_tau')} checkin_tau={th.get('checkin_tau')} safety_tau={th.get('safety_tau')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
