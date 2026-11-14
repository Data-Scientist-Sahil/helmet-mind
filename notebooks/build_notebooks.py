"""Writes the two push-button Kaggle notebooks. Edit here, then: python notebooks/build_notebooks.py

  01_kaggle_audio.ipynb  scripts -> Indic Parler-TTS -> real + synthetic noise / phone chain -> clips + manifest
                         (GPU T4; round 2 reuses round-1 renders when the old 01 output is attached)
  02_kaggle_train.ipynb  parity -> features -> zero-shot -> train head -> cross-fit -> calibrate -> evaluate
                         -> fp16 check -> latency   (GPU T4 x2)

They are separate because parler-tts pins transformers 4.46.1 while Gemma 4 needs transformers 5.17.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NIRNAYA_COMMIT = "0fcd64f22b3013dd2bb1e268533bcc91accf87c6"

ENV_FIRST = r'''
# MUST run before anything imports transformers: parler-tts pins protobuf 4.x, which breaks the TensorFlow that
# transformers would otherwise import on Kaggle images.
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PYTHONUNBUFFERED"] = "1"
'''

COMMON_FETCH = r'''
import os, subprocess, sys, glob, shutil, json
from pathlib import Path

def run(cmd, cwd=None, env=None, check=True):
    """Run a shell command and stream its output live (every stage is a subprocess, so GPU memory is freed after it)."""
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})},
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        if "it/s]" not in line:
            print(line, end="", flush=True)
    p.wait()
    if check and p.returncode != 0:
        raise RuntimeError(f"failed ({p.returncode}): {cmd}")
    return p.returncode

# --- scratch outside /kaggle/working, so code, model and features are not saved into the notebook output
TMP = Path("/kaggle/tmp") if Path("/kaggle/tmp").exists() else Path("/tmp")
# --- code: a Kaggle dataset upload of the repo folder (newest spec version wins), or a git URL
CODE = TMP / "rvd"
if not CODE.exists():
    found = [Path(p).parent.parent for p in glob.glob("/kaggle/input/**/spec/decision_set.json", recursive=True)]
    found.sort(key=lambda d: json.loads((d / "spec" / "decision_set.json").read_text()).get("version", "0"))
    if found:
        shutil.copytree(found[-1], CODE)
    elif REPO_URL:
        run(f"git clone --depth 1 {REPO_URL} {CODE}")
    else:
        raise SystemExit("Add the repo folder as a Kaggle dataset (Add Input -> Datasets), or set REPO_URL")
print("code:", CODE, "| spec", json.loads((CODE / "spec" / "decision_set.json").read_text())["version"])
'''

SECRETS = r'''
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle secrets")
except Exception as e:  # harmless for public models; needed for the gated TTS model in notebook 01
    print("HF_TOKEN secret not available:", type(e).__name__, "- fine for public models")
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells):
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:02d}"
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


AUDIO = [
    md("""
# 01 · Synthetic rider audio (scripts → TTS → real + synthetic noise & phone channel)

**Settings:** Accelerator **GPU T4** · Internet **on** · Secret **HF_TOKEN** (Add-ons → Secrets).
Indic Parler-TTS is gated with automatic approval: open huggingface.co/ai4bharat/indic-parler-tts once and accept.

Inputs: the repo folder as a dataset (or set `REPO_URL`). **Round 2:** also attach the previous run of this notebook —
its TTS renders are reused (same script, same voice, same seed), so only the new scripts are synthesised.
Optional: your own road recordings as a dataset with `traffic/ horn/ engine/ wind/ rain/ babble/ kitchen/` folders.

Run all. Resumable. Output: `/kaggle/working/rv-audio` → **Save Version** → attach to notebook 02.
"""),
    code(ENV_FIRST),
    code("""
# ---- settings
REPO_URL = ""
RAW_SCRIPTS = ["data/scripts/m1_wedge.raw.jsonl", "data/scripts/m1_round2.raw.jsonl"]
FREEZE_SPLITS = "data/scripts/m1_wedge.jsonl"   # round-1 scripts keep their split -> results stay comparable
FETCH_MS_SNSD = True                            # real noise (CC0 / CC BY-SA 3.0) into the chain
MY_NOISE = ""                                   # e.g. "/kaggle/input/rv-noise" - your own recordings, merged in
OUT = "/kaggle/working/rv-audio"
TIERS = "clean,20,10,5,0"
MAX_TTS_MINUTES = 600
"""),
    code(COMMON_FETCH + SECRETS),
    code("""
run("nvidia-smi --query-gpu=name,memory.total --format=csv")
run("pip install -q git+https://github.com/huggingface/parler-tts.git soundfile")   # pins transformers 4.46.1 - fine here
run("ffmpeg -hide_banner -encoders 2>/dev/null | grep -E 'libopus|amrnb' || echo 'no opus/amr encoder: G.711 fallback'", check=False)
"""),
    code("""
# ---- 1. scripts: lint + dedup + gold labels (rules) + script-level split (frozen for round-1 scripts)
Path(OUT).mkdir(parents=True, exist_ok=True)
raw = " ".join(str(CODE / r) for r in RAW_SCRIPTS)
run(f"python src/scripts.py build --raw {raw} --out {OUT}/scripts.jsonl --freeze-splits {CODE / FREEZE_SPLITS}", cwd=CODE)
"""),
    code("""
# ---- 2. reuse earlier renders (attach the previous run of this notebook as an input)
prev = [Path(p).parent for p in glob.glob("/kaggle/input/**/tts/tts_manifest.jsonl", recursive=True)]
tts_dir = Path(OUT) / "tts"
if prev and not (tts_dir / "tts_manifest.jsonl").exists():
    shutil.copytree(prev[0], tts_dir)
    print("reusing", sum(1 for _ in open(tts_dir / "tts_manifest.jsonl")), "renders from", prev[0])
"""),
    code("""
# ---- 3. TTS (resumable; progress printed every 10 renders)
run(f"python src/tts.py --scripts {OUT}/scripts.jsonl --out {OUT}/tts --max-minutes {MAX_TTS_MINUTES}", cwd=CODE)
"""),
    code("""
# ---- 4. real noise: MS-SNSD noise folders (+ your own recordings)
NOISE = TMP / "noise"
if FETCH_MS_SNSD:
    run(f"python src/fetch_noise.py --out {NOISE}", cwd=CODE)
if MY_NOISE:
    for cat_dir in Path(MY_NOISE).iterdir():
        if cat_dir.is_dir():
            dst = NOISE / cat_dir.name / "train"; dst.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(sorted(cat_dir.rglob("*.wav"))):  # every 5th own recording -> test noise
                shutil.copy(f, (NOISE / cat_dir.name / ("test" if i % 5 == 0 else "train") / f"own_{f.name}"))
"""),
    code("""
# ---- 5. noise + channel chain -> clips + manifest (CPU)
nd = f"--noise-dir {NOISE}" if NOISE.exists() else ""
run(f"python src/augment.py --tts-manifest {OUT}/tts/tts_manifest.jsonl --scripts {OUT}/scripts.jsonl --out {OUT}/clips "
    f"--tiers {TIERS} {nd}", cwd=CODE)
"""),
    code("""
# ---- 6. check: counts + listen to a few clips
import collections, random
from IPython.display import Audio, display
man = [json.loads(l) for l in open(f"{OUT}/clips/manifest.jsonl")]
print(collections.Counter((m["split"], m["tier"]) for m in man))
print("noise sources:", collections.Counter(s.split(":")[1] for m in man for s in m["aug"].get("noise_src", [])))
flag = [json.loads(l) for l in open(f"{OUT}/tts/tts_manifest.jsonl")]
print("renders with odd length:", sum(not f["length_ok"] for f in flag), "/", len(flag))
for m in random.Random(0).sample(man, 4):
    print(m["clip_id"], m["situation"], m["tier"], m["aug"].get("noise_src"), m["aug"].get("codec_used"))
    display(Audio(f"{OUT}/clips/{m['path']}"))
"""),
]

TRAIN = [
    md("""
# 02 · Train the decision head (frozen nirnaya-flash-1.0 backbone)

**Settings:** Accelerator **GPU T4 x2** · Internet **on**. Inputs: the repo folder (dataset) and the output of
notebook 01 (`rv-audio`). Optional: real clips as a dataset with `labels.csv` + `audio/`.

1. **P0 parity** — released head on our features, 20 easy decisions (≥ 18/20).
2. **Features** — one forward per clip, cached in temporary storage (not in the download).
3. **Zero-shot** baseline, in the same label scope.
4. **Train** (balanced class weights, scope mask) → **cross-fit** (out-of-fold logits) → **calibrate** per question +
   safety τ from the out-of-fold safety scripts.
5. **Evaluate** — script-level CIs, per-class recall, per-script table, false alarms per script.
6. **fp16** — accuracy on an fp16 extraction of test + latency in fp32 and fp16.

Download only `out/results_<RUN>.zip`.
"""),
    code("""
# ---- settings
REPO_URL = ""
MODEL_ID = "Darkrider/nirnaya-flash-1.0"
MODEL_DIR = ""                                   # default: <tmp>/nirnaya-flash-1.0 (16 GB, not saved)
RUN = "m1v2"
SUBSET = "m1"                                    # scope from the spec: filters situations AND masks out-of-scope labels
PRECISION = "fp32"                               # fp32 (T4) | fp16 | bf16 (A100/L4+)
VARIANT = "post_norm"
MEM_CAP = 256
EPOCHS, LR, FREEZE = 30, 1e-5, "none"
WEIGHT_MODES = ["none", "sqrt", "balanced"]      # each trained (~4 min); the best on VAL balanced action accuracy is used
CV_FOLDS = 5                                     # 0 = skip cross-fitting (tau from val only)
RUN_FP16_CHECK = True
KEEP_FEATURES = False                            # True = also save features (1.5-3 GB) into the notebook output
OUT = "/kaggle/working/out"
MAX_EXTRACT_MINUTES = 600
"""),
    code(COMMON_FETCH + SECRETS + r'''
NIRNAYA = TMP / "nirnaya-inference"
if not NIRNAYA.exists():
    run(f"git clone https://github.com/NeverInAsh/nirnaya-inference {NIRNAYA} && cd {NIRNAYA} && git checkout ''' + NIRNAYA_COMMIT + r'''")
ENV = {"PYTHONPATH": f"{NIRNAYA}:{CODE}", "TOKENIZERS_PARALLELISM": "false"}
MODEL_DIR = MODEL_DIR or str(TMP / "nirnaya-flash-1.0")
_man = sorted(glob.glob("/kaggle/input/**/clips/manifest.jsonl", recursive=True))
if not _man:
    raise SystemExit("No clips/manifest.jsonl under /kaggle/input - attach notebook 01's output (Add Input -> Your Work -> Notebooks)")
AUDIO = Path(_man[-1]).parent
REAL = next((Path(p).parent for p in glob.glob("/kaggle/input/**/labels.csv", recursive=True)), None)
FEAT = Path(OUT) / "features" if KEEP_FEATURES else TMP / "features"
print("audio:", AUDIO, "| real clips:", REAL, "| features:", FEAT)
Path(OUT).mkdir(parents=True, exist_ok=True)
'''),
    code("""
run("nvidia-smi --query-gpu=name,memory.total --format=csv; free -g | head -2")
run("pip install -q 'transformers==5.17.0' accelerate safetensors librosa soundfile hf_transfer")
run("python -c 'import transformers, torch; from transformers import Gemma4ForConditionalGeneration; print(transformers.__version__, torch.__version__, torch.cuda.device_count())'")
"""),
    code("""
# ---- model download (~16 GB, a few minutes)
run(f"python -c \\"from huggingface_hub import snapshot_download; snapshot_download('{MODEL_ID}', local_dir='{MODEL_DIR}')\\"",
    env={"HF_HUB_ENABLE_HF_TRANSFER": "1"})
"""),
    code("""
# ---- P0: parity gate
clips = sorted(glob.glob(f"{AUDIO}/test/*__clean.wav"))[:4]
rc = run(f"python src/parity.py --model {MODEL_DIR} --precision {PRECISION} --variant {VARIANT} "
         f"--clips {' '.join(clips)} --out {OUT}/parity.json", cwd=CODE, env=ENV, check=False)
if rc != 0:
    raise SystemExit("P0 FAILED - try VARIANT='pre_norm', then PRECISION='fp16'; note the result in docs/backbone.md")
"""),
    code("""
# ---- features: one forward per clip, cached (resumable)
run(f"python src/extract.py --model {MODEL_DIR} --manifest {AUDIO}/manifest.jsonl --out {FEAT} "
    f"--precision {PRECISION} --variant {VARIANT} --mem-cap {MEM_CAP} --max-minutes {MAX_EXTRACT_MINUTES}", cwd=CODE, env=ENV)
run(f"du -sh {FEAT}; wc -l {FEAT}/index.jsonl")
"""),
    code("""
# ---- B3: zero-shot = released head + released temperature, same scope mask as the trained run
ZS = Path(f"{OUT}/runs/zeroshot"); ZS.mkdir(parents=True, exist_ok=True)
for f in ("head.pt", "t_entropy.json"):
    shutil.copy(f"{MODEL_DIR}/{f}", ZS / f)
sub = f"--subset {SUBSET}" if SUBSET else ""
run(f"python train/predict.py --run {ZS} --features {FEAT} --splits test {sub} --out {ZS}/pred_test.jsonl", cwd=CODE, env=ENV)
"""),
    code("""
# ---- train, cross-fit, calibrate, predict
val = {}
for wm in WEIGHT_MODES:
    d = f"{OUT}/runs/{RUN}_w{wm}"
    run(f"python train/train_head.py --features {FEAT} --out {d} --init {MODEL_DIR}/head.pt --epochs {EPOCHS} --lr {LR} "
        f"--freeze {FREEZE} --class-weights {wm} {sub}", cwd=CODE, env=ENV)
    best = json.loads(Path(d, "config.json").read_text())["best_epoch"]
    hist = {h["epoch"]: h for h in json.loads(Path(d, "history.json").read_text())}
    val[wm] = hist[best].get("bacc_action", 0.0)
    print(f"class weights {wm}: val balanced action accuracy {val[wm]:.3f}")
choice = max(val, key=val.get)
R = f"{OUT}/runs/{RUN}"
if Path(R).exists():
    shutil.rmtree(R)
shutil.copytree(f"{OUT}/runs/{RUN}_w{choice}", R)
print("chosen on validation:", choice, val)
if CV_FOLDS:
    run(f"python train/crossfit.py --run {R} --folds {CV_FOLDS}", cwd=CODE, env=ENV)
run(f"python train/calibrate.py --run {R} --init-fit {MODEL_DIR}/t_entropy.json", cwd=CODE, env=ENV)
run(f"python train/predict.py --run {R} --features {FEAT} --splits test {sub} --out {R}/pred_test.jsonl", cwd=CODE, env=ENV)
"""),
    code("""
# ---- evaluate on the synthetic test tiers (script-level CIs)
for wm in WEIGHT_MODES:  # the other weightings too, for the record (uncalibrated: they share no tau of their own)
    if wm != choice:
        run(f"python train/predict.py --run {OUT}/runs/{RUN}_w{wm} --features {FEAT} --splits test {sub} "
            f"--out {OUT}/runs/{RUN}_w{wm}/pred_test.jsonl", cwd=CODE, env=ENV)
others = " ".join(f"weights_{wm}={OUT}/runs/{RUN}_w{wm}/pred_test.jsonl" for wm in WEIGHT_MODES if wm != choice)
run(f"python train/evaluate.py --pred ours={R}/pred_test.jsonl zeroshot={ZS}/pred_test.jsonl {others} "
    f"--out {OUT}/results/{RUN}_synthetic --group tier language", cwd=CODE, env=ENV)
"""),
    code("""
# ---- real clips (only if a labels.csv dataset is attached) - looked at ONCE, never used for any choice
if REAL:
    run(f"python src/real_manifest.py --labels {REAL}/labels.csv --audio {REAL}/audio --out {OUT}/real_manifest.jsonl", cwd=CODE, env=ENV)
    run(f"python src/extract.py --model {MODEL_DIR} --manifest {OUT}/real_manifest.jsonl --out {TMP}/features_real "
        f"--precision {PRECISION} --variant {VARIANT} --mem-cap {MEM_CAP}", cwd=CODE, env=ENV)
    for name, rd in (("ours", R), ("zeroshot", ZS)):
        run(f"python train/predict.py --run {rd} --features {TMP}/features_real --splits real {sub} --out {rd}/pred_real.jsonl", cwd=CODE, env=ENV)
    run(f"python train/evaluate.py --pred ours={R}/pred_real.jsonl zeroshot={ZS}/pred_real.jsonl "
        f"--out {OUT}/results/{RUN}_real --group language", cwd=CODE, env=ENV)
else:
    print("no real clips attached - skipping")
"""),
    code("""
# ---- fp16: does the fp32-trained head still work on fp16 features? + latency in both precisions
lat = Path(f"{OUT}/results/{RUN}_latency.jsonl"); lat.write_text("")
clip = sorted(glob.glob(f"{AUDIO}/test/*__snr10.wav"))[0]
precisions = [PRECISION] + (["fp16"] if RUN_FP16_CHECK and PRECISION != "fp16" else [])
for prec in precisions:
    if prec != PRECISION:
        run(f"python src/parity.py --model {MODEL_DIR} --precision {prec} --variant {VARIANT} --out {OUT}/parity_{prec}.json",
            cwd=CODE, env=ENV, check=False)
        run(f"python src/extract.py --model {MODEL_DIR} --manifest {AUDIO}/manifest.jsonl --splits test --out {TMP}/features_{prec} "
            f"--precision {prec} --variant {VARIANT} --mem-cap {MEM_CAP}", cwd=CODE, env=ENV)
        run(f"python train/predict.py --run {R} --features {TMP}/features_{prec} --splits test {sub} --out {R}/pred_test_{prec}.jsonl", cwd=CODE, env=ENV)
        run(f"python train/evaluate.py --pred ours_{PRECISION}={R}/pred_test.jsonl ours_{prec}={R}/pred_test_{prec}.jsonl "
            f"--out {OUT}/results/{RUN}_{prec}_vs_{PRECISION} --group tier", cwd=CODE, env=ENV)
    run(f"python src/engine.py --model {MODEL_DIR} --run {R} {clip} --bench 50 --precision {prec} | tail -1 >> {lat}", cwd=CODE, env=ENV)
print(lat.read_text())
"""),
    code("""
# ---- bundle the small results (features are NOT included - they live in temporary storage)
import zipfile
from IPython.display import Markdown, display
display(Markdown(open(f"{OUT}/results/{RUN}_synthetic/REPORT.md").read()))
_zip = Path(OUT) / f"results_{RUN}.zip"
_files = [p for p in Path(f"{OUT}/results").rglob("*") if p.is_file()] + sorted(Path(f"{OUT}/runs").rglob("*.json*"))
_files = [p for p in _files if "_w" not in p.parent.name or p.suffix == ".json"]  # skip duplicate heads
_files += [p for p in (Path(f"{OUT}/runs/{RUN}/head.pt"), Path(f"{OUT}/runs/{RUN}/oof_logits.pt"),
                       Path(f"{OUT}/runs/{RUN}/val_logits.pt")) if p.exists()]
_files += sorted(Path(OUT).glob("parity*.json"))
with zipfile.ZipFile(_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for p in _files:
        z.write(p, p.relative_to(OUT))
print(_zip, round(_zip.stat().st_size / 2**20, 1), "MB,", len(_files), "files  <- download this one")
"""),
]


def main():
    for name, cells in (("01_kaggle_audio.ipynb", AUDIO), ("02_kaggle_train.ipynb", TRAIN)):
        (HERE / name).write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False))
        print("wrote", HERE / name)


if __name__ == "__main__":
    main()
