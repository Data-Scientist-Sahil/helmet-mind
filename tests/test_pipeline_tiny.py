"""End-to-end pipeline test on CPU, with a tiny random-weight Gemma 4 that keeps the REAL tokenizer, processor and
chat template (so prompt layout, audio-token expansion and span finding are exercised exactly as on Kaggle).

    python tests/test_pipeline_tiny.py --nirnaya <path to nirnaya-inference> [--src <real model small files>] [--work /tmp/x]

Stages: scripts (round 1 + round 2, frozen splits) -> stand-in "TTS" audio (tones that differ by situation) -> augment
(labels refreshed from the scripts file) -> extract -> train head (balanced weights, m1 scope mask) -> cross-fit ->
calibrate (out-of-fold, per question) -> predict (+ consistency rule) -> evaluate (script-level CIs) -> engine.
Takes ~3-6 minutes on a laptop CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SMALL = ["config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
         "generation_config.json"]


def run(cmd, env):
    print("$", " ".join(map(str, cmd)))
    r = subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT, env=env, capture_output=True, text=True)
    out = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "it/s]" not in l and "Warning" not in l)
    print(out[-3000:])
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {cmd}")
    return out


def fake_tts(scripts_path, out_dir, per_script=2):
    """One or two short 'utterances' per script: a situation-specific chord with a syllable envelope."""
    import soundfile as sf

    sys.path.insert(0, str(ROOT))
    from src.labels import SITUATIONS

    rows = [json.loads(x) for x in open(scripts_path, encoding="utf-8")]
    out = Path(out_dir)
    (out).mkdir(parents=True, exist_ok=True)
    man = []
    for r in rows:
        k = SITUATIONS.index(r["situation"])
        for j in range(per_script if r["split"] == "train" else 1):
            rng = np.random.default_rng(abs(hash((r["id"], j))) % 2**32)
            dur = rng.uniform(1.5, 3.0)
            t = np.arange(int(dur * 16000)) / 16000
            f0 = 120 + 25 * k + rng.uniform(-5, 5)
            x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 6))
            x = (0.2 * x * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2)).astype(np.float32)
            rid = f"{r['id']}__fake{j}"
            rel = f"{r['split']}/{rid}.wav"
            (out / r["split"]).mkdir(exist_ok=True)
            sf.write(out / rel, x, 16000)
            man.append({"render_id": rid, "script_id": r["id"], "split": r["split"], "speaker": "fake", "voice_lang": "hi",
                        "caption": "", "text": r["text"], "kind": r["kind"], "register": r["register"], "path": rel,
                        **{q: r[q] for q in ("situation", "action", "safety", "language")}})
    with open(out / "tts_manifest.jsonl", "w", encoding="utf-8") as f:
        for m in man:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    return out / "tts_manifest.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nirnaya", required=True, help="path to a clone of nirnaya-inference")
    ap.add_argument("--src", default=None, help="dir with the real model's small files (downloaded if omitted)")
    ap.add_argument("--work", default="/tmp/rvd_tiny")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": f"{a.nirnaya}{os.pathsep}{ROOT}", "TOKENIZERS_PARALLELISM": "false"}

    src = Path(a.src) if a.src else work / "src_files"
    if not a.src:
        from huggingface_hub import hf_hub_download

        for f in SMALL:
            hf_hub_download("Darkrider/nirnaya-flash-1.0", f, local_dir=src)
    tiny = work / "tiny_model"
    if not (tiny / "model.safetensors").exists():
        run(["tests/make_tiny_model.py", "--src", src, "--out", tiny], env)

    scripts = work / "scripts.jsonl"
    run(["src/scripts.py", "build", "--raw", "data/scripts/m1_wedge.raw.jsonl", "data/scripts/m1_round2.raw.jsonl",
         "--out", scripts, "--freeze-splits", "data/scripts/m1_wedge.jsonl"], env)
    tts_man = fake_tts(scripts, work / "tts")
    run(["src/augment.py", "--tts-manifest", tts_man, "--scripts", scripts, "--out", work / "clips", "--tiers", "clean,10,0"], env)
    man = [json.loads(x) for x in open(work / "clips" / "manifest.jsonl", encoding="utf-8")]
    assert any(m["ambiguous"] for m in man) and any(m.get("group") for m in man), "labels must come from the scripts file"
    feats = work / "features"
    run(["src/extract.py", "--model", tiny, "--manifest", work / "clips" / "manifest.jsonl", "--out", feats,
         "--device", "cpu", "--mem-cap", "64", "--shard-size", "50"], env)
    out = run(["src/extract.py", "--model", tiny, "--manifest", work / "clips" / "manifest.jsonl", "--out", feats,
               "--device", "cpu", "--mem-cap", "64"], env)
    assert "0 to do" in out, "extract must be resumable (second run should have nothing to do)"
    run_dir = work / "runs" / "m1"
    out = run(["train/train_head.py", "--features", feats, "--out", run_dir, "--init", tiny / "head.pt", "--epochs", "4",
               "--patience", "3", "--lr", "3e-4", "--device", "cpu", "--subset", "m1"], env)
    assert "action class weights" in out and "scope m1" in out
    run(["train/crossfit.py", "--run", run_dir, "--folds", "3", "--epochs", "2"], env)
    out = run(["train/calibrate.py", "--run", run_dir, "--init-fit", tiny / "t_entropy.json"], env)
    assert "logits: oof" in out
    run(["train/predict.py", "--run", run_dir, "--features", feats, "--splits", "test", "--subset", "m1",
         "--out", run_dir / "pred_test.jsonl", "--device", "cpu"], env)
    zs = work / "runs" / "zeroshot"
    zs.mkdir(parents=True, exist_ok=True)
    for f in ("head.pt", "t_entropy.json"):
        (zs / f).write_bytes((tiny / f).read_bytes())
    run(["train/predict.py", "--run", zs, "--features", feats, "--splits", "test", "--subset", "m1", "--out", zs / "pred_test.jsonl",
         "--device", "cpu"], env)
    run(["train/evaluate.py", "--pred", f"ours={run_dir / 'pred_test.jsonl'}", f"zeroshot={zs / 'pred_test.jsonl'}",
         "--out", work / "results", "--bootstrap", "200"], env)
    clip = next((work / "clips" / "test").glob("*.wav"))
    out = run(["src/engine.py", "--model", tiny, "--run", run_dir, clip, "--device", "cpu", "--bench", "3"], env)
    assert '"card"' in out and "p50_ms" in out

    m = json.loads((work / "results" / "metrics.json").read_text())
    for sysname in ("ours", "zeroshot"):
        s = m["systems"][sysname]["all"]
        assert s["n"] > 0 and s["situation"]["n"] > 0 and s["safety_flag"] is not None
        for q in ("situation", "action", "safety", "language"):
            assert 0 <= s[q]["ece"] <= 1 and np.isfinite(s[q]["nll"])
    assert {"by_tier", "by_language"} <= set(m["systems"]["ours"])
    for f in ("head.pt", "t_entropy.json", "thresholds.json", "config.json", "history.json", "val_logits.pt", "calibration.json",
              "oof_logits.pt"):
        assert (run_dir / f).exists(), f
    pred = [json.loads(x) for x in open(run_dir / "pred_test.jsonl", encoding="utf-8")]
    scope = json.loads((run_dir / "config.json").read_text())["scope"]
    from src.labels import options
    out_of_scope = [i for i, s in enumerate(options("situation")) if s not in scope["situation"]]
    assert all(p["probs"]["situation"][i] == 0.0 for p in pred for i in out_of_scope), "scope mask must zero out-of-scope labels"
    assert all(not (p["consistency"]["safety_flag"] and p["answers"]["action"]["choice"] not in ("sos", "contact_support"))
               for p in pred), "consistency rule violated"
    s = m["systems"]["ours"]["all"]
    assert s["action"]["accuracy_ci"]["n_clusters"] == len({p["script_id"] for p in pred})
    print("\nPIPELINE TEST OK — every stage ran end to end on the tiny model")


if __name__ == "__main__":
    main()
