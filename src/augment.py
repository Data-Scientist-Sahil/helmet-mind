"""Noise + channel chain: clean TTS renders -> rider-like phone voice notes, and the manifest the backbone reads.

    python src/augment.py --tts-manifest data/audio/tts_manifest.jsonl --out data/audio/clips \
        [--scripts data/scripts/scripts.jsonl] [--noise-dir data/noise] [--tiers clean,20,10,5,0]

--scripts: take labels (situation/action/safety/language/ambiguous/group/split) from the built scripts file rather
than from the TTS manifest - needed when renders from an earlier round are reused.

Per render (all randomness seeded by the render id, so every clip is reproducible):
  train / val   one random draw from the training distribution (10% of train renders also kept clean)
  test          one clip per evaluation tier: clean, then background noise at a FIXED SNR of 20 / 10 / 5 / 0 dB
                (types, helmet, phone band and codec drawn from the seed, identical across tiers)

Chain: speed perturb (0.9-1.1) -> loudness normalise -> helmet (low-pass + low-mid boost + small-room reverb)
       -> background at SNR (traffic, horn, engine, wind, rain, babble, kitchen) -> wind gusts + soft clipping
       -> phone channel (narrow- or wide-band) -> codec round trip (Opus / AMR-NB via ffmpeg when available, else
       G.711 mu-law) -> AGC, clipping, packet loss -> gain -> 2-8 s.
Background noise comes from <noise-dir>/<category>/*.wav when present (your own road recordings are best), else from
the synthetic generators below. Noise files are split 80/20 by file hash: test clips only ever use the 20%.

    python src/augment.py --selftest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import signal

SR = 16000
CATEGORIES = ["traffic", "horn", "engine", "wind", "rain", "babble", "kitchen"]
LABEL_KEYS = ("situation", "action", "safety", "language")


def rng_for(*keys) -> np.random.Generator:
    h = hashlib.sha256("|".join(map(str, keys)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


# --------------------------------------------------------------------------- io
def load(path, sr=SR):
    import soundfile as sf

    x, s = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if s != sr:
        g = np.gcd(int(s), sr)
        x = signal.resample_poly(x, sr // g, int(s) // g).astype(np.float32)
    return x


def save(path, x, sr=SR):
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1, 1).astype(np.float32), sr, subtype="PCM_16")


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def active_rms(x, frame=320):
    """RMS over the louder half of 20 ms frames (ignores silences, so SNR refers to the speech itself)."""
    n = len(x) // frame
    if n < 2:
        return rms(x)
    e = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(1) + 1e-12)
    return float(np.sqrt(np.mean(np.sort(e)[n // 2:] ** 2)))


def normalise(x, dbfs=-23.0):
    return x * (10 ** (dbfs / 20) / max(active_rms(x), 1e-6))


def fit_len(n, x, rng):
    if len(x) >= n:
        s = int(rng.integers(0, len(x) - n + 1))
        return x[s:s + n]
    reps = int(np.ceil(n / max(len(x), 1)))
    return np.tile(x, reps)[:n]


# --------------------------------------------------------------------------- filters
def butter(x, kind, f, order=4):
    sos = signal.butter(order, f, btype=kind, fs=SR, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def peaking(x, f0, gain_db, q=0.9):
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    al = np.sin(w0) / (2 * q)
    b = [1 + al * a, -2 * np.cos(w0), 1 - al * a]
    aa = [1 + al / a, -2 * np.cos(w0), 1 - al / a]
    return signal.lfilter(np.array(b) / aa[0], np.array(aa) / aa[0], x).astype(np.float32)


def room_ir(rng, rt60):
    n = int(SR * rt60 * 1.2)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0
    return (ir / np.sqrt((ir ** 2).sum())).astype(np.float32)


def helmet(x, rng):
    y = butter(x, "lowpass", float(rng.uniform(2500, 4000)))
    y = peaking(y, float(rng.uniform(200, 500)), float(rng.uniform(2, 4)))
    wet = signal.fftconvolve(y, room_ir(rng, float(rng.uniform(0.05, 0.15))))[: len(y)]
    m = float(rng.uniform(0.2, 0.4))
    return ((1 - m) * y + m * wet).astype(np.float32)


# --------------------------------------------------------------------------- synthetic noise
def _brown(n, rng):
    w = np.cumsum(rng.standard_normal(n))
    return butter(w - signal.savgol_filter(w, 2001 if n > 2001 else (n // 2) * 2 - 1, 1), "highpass", 20)


def _envelope(n, rng, rate_hz, smooth=0.5):
    k = max(2, int(n / SR * rate_hz) + 2)
    pts = rng.uniform(1 - smooth, 1 + smooth, k)
    return np.interp(np.arange(n), np.linspace(0, n - 1, k), pts).astype(np.float32)


def gen_traffic(n, rng):
    x = butter(_brown(n, rng), "lowpass", 1200) * _envelope(n, rng, 0.3)
    for _ in range(int(rng.integers(1, 5))):  # passing vehicles
        c, w = rng.uniform(0, n), rng.uniform(0.6, 2.0) * SR
        env = np.exp(-0.5 * ((np.arange(n) - c) / (w / 2.5)) ** 2)
        x += float(rng.uniform(1, 4)) * butter(rng.standard_normal(n), "bandpass", [100, float(rng.uniform(800, 2500))]) * env
    return x


def gen_horn(n, rng):
    x = np.zeros(n, np.float32)
    t = np.arange(n) / SR
    for _ in range(int(rng.integers(1, 5))):
        f0, dur, st = rng.uniform(350, 520), rng.uniform(0.15, 0.6), rng.uniform(0, max(0.01, n / SR - 0.6))
        m = (t >= st) & (t < st + dur)
        ph = 2 * np.pi * f0 * t + 0.3 * np.sin(2 * np.pi * 6 * t)
        tone = sum((0.6 ** k) * np.sin((k + 1) * ph) for k in range(5))
        x += (m * tone * float(rng.uniform(0.5, 1))).astype(np.float32)
    return butter(x + 0.02 * rng.standard_normal(n), "highpass", 150)


def gen_engine(n, rng):
    t = np.arange(n) / SR
    f0 = rng.uniform(20, 45) * (1 + 0.03 * np.sin(2 * np.pi * rng.uniform(0.2, 1) * t))
    ph = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(rng.uniform(0.2, 1) / (k ** 0.7) * np.sin(k * ph + rng.uniform(0, 6.28)) for k in range(1, 30))
    return (x + 0.3 * butter(rng.standard_normal(n), "lowpass", 600)).astype(np.float32)


def gen_wind(n, rng):
    g = np.clip(_envelope(n, rng, 1.5, 0.9), 0.05, None) ** 2
    return butter(rng.standard_normal(n), "lowpass", float(rng.uniform(150, 400))) * g


def gen_rain(n, rng):
    x = 0.4 * butter(rng.standard_normal(n), "highpass", 1000)
    drops = rng.random(n) < rng.uniform(0.0005, 0.003)
    return x + butter(drops * rng.standard_normal(n) * 6, "bandpass", [1500, 6000])


def gen_babble(n, rng, pool=None):
    if pool:
        x = np.zeros(n, np.float32)
        for _ in range(int(rng.integers(3, 7))):
            x += fit_len(n, pool[int(rng.integers(0, len(pool)))], rng) * float(rng.uniform(0.3, 1))
        return butter(x, "lowpass", 4000)
    s = butter(rng.standard_normal(n), "bandpass", [150, 3500])  # speech-shaped noise with syllable-rate modulation
    t = np.arange(n) / SR
    return s * (0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(3, 5) * t + rng.uniform(0, 6))).astype(np.float32)


def gen_kitchen(n, rng):
    x = 0.15 * rng.standard_normal(n).astype(np.float32)
    for _ in range(int(rng.integers(3, 12))):
        s, L = int(rng.integers(0, n)), int(rng.uniform(0.02, 0.15) * SR)
        burst = rng.standard_normal(L) * np.exp(-np.arange(L) / (L / 5))
        x[s:s + L] += butter(burst, "bandpass", [float(rng.uniform(800, 2000)), float(rng.uniform(4000, 7500))])[: len(x[s:s + L])] * 3
    return x


GEN = {"traffic": gen_traffic, "horn": gen_horn, "engine": gen_engine, "wind": gen_wind, "rain": gen_rain,
       "kitchen": gen_kitchen}


class NoiseBank:
    """Real noise from <noise_dir>/<category>/{train,test}/*.wav (e.g. src/fetch_noise.py + your own recordings);
    flat <noise_dir>/<category>/*.wav also works (split 80/20 by file hash). Categories without files fall back to the
    synthetic generators. Test clips only ever draw from test files."""

    def __init__(self, noise_dir=None, babble_pool=None, p_real=0.7):
        self.files = {c: {"train": [], "test": []} for c in CATEGORIES}
        if noise_dir and Path(noise_dir).exists():
            for c in CATEGORIES:
                base = Path(noise_dir, c)
                tr, te = sorted(map(str, (base / "train").glob("*.wav"))), sorted(map(str, (base / "test").glob("*.wav")))
                flat = sorted(map(str, base.glob("*.wav")))
                for f in flat:
                    (te if int(hashlib.md5(Path(f).name.encode()).hexdigest(), 16) % 5 == 0 else tr).append(f)
                self.files[c] = {"train": tr, "test": te}
        self.babble_pool = babble_pool
        self.p_real = p_real  # keep some synthetic draws even when real files exist (more variety)

    def get(self, c, n, rng, split):
        fs = self.files[c]["test" if split == "test" else "train"]
        if fs and rng.random() < self.p_real:
            return fit_len(n, _cached(fs[int(rng.integers(0, len(fs)))]), rng), "file"
        if c == "babble":
            return gen_babble(n, rng, self.babble_pool), "synthetic"
        return GEN[c](n, rng).astype(np.float32), "synthetic"

    def summary(self):
        return {c: {k: len(v) for k, v in d.items()} for c, d in self.files.items()}


@lru_cache(maxsize=64)
def _cached(path):
    return load(path)


def mix_at_snr(speech, noise, snr_db):
    s, nz = active_rms(speech), rms(noise)
    return speech + noise * (s / max(nz, 1e-9)) / (10 ** (snr_db / 20))


# --------------------------------------------------------------------------- phone channel
@lru_cache(maxsize=1)
def ffmpeg_encoders() -> set:
    if not shutil.which("ffmpeg"):
        return set()
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001
        return set()
    return {name for name in ("libopus", "libopencore_amrnb") if f" {name} " in out}


def codec_roundtrip(x, codec, rng):
    """-> (signal, codec actually used). Falls back to G.711 when ffmpeg can't encode the requested codec."""
    enc = ffmpeg_encoders()
    if codec == "opus" and "libopus" in enc:
        args, ext = ["-c:a", "libopus", "-b:a", f"{int(rng.choice([12, 14, 16]))}k"], ".ogg"
    elif codec == "amr" and "libopencore_amrnb" in enc:
        args, ext = ["-ar", "8000", "-c:a", "libopencore_amrnb", "-b:a", "12.2k"], ".amr"
    elif codec in ("opus", "amr", "g711"):
        return g711(x), "g711"
    else:
        return x, "none"
    with tempfile.TemporaryDirectory() as d:
        src, mid, dst = os.path.join(d, "a.wav"), os.path.join(d, "b" + ext), os.path.join(d, "c.wav")
        save(src, x)
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, *args, mid], check=True, timeout=60)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mid, "-ar", str(SR), "-ac", "1", dst], check=True, timeout=60)
            y = load(dst)
        except Exception:  # noqa: BLE001
            return g711(x), "g711"
    y = np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)]
    return y.astype(np.float32), codec


def g711(x, mu=255):
    y = signal.resample_poly(x, 1, 2)
    y = np.clip(y, -1, 1)
    q = np.round((np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)) * 127) / 127
    y = np.sign(q) * (np.expm1(np.abs(q) * np.log1p(mu)) / mu)
    return signal.resample_poly(y, 2, 1)[: len(x)].astype(np.float32)


def agc(x, target_db=-20.0, win=0.3):
    n = max(1, int(win * SR))
    env = np.sqrt(signal.fftconvolve(x ** 2, np.ones(n) / n, mode="same").clip(1e-8))
    g = np.clip((10 ** (target_db / 20)) / env, 0.25, 6.0)
    return (x * signal.savgol_filter(g, min(len(g) // 2 * 2 - 1, 1601) if len(g) > 5 else 1, 1)).astype(np.float32) if len(g) > 5 else x


def packet_loss(x, rng, rate):
    f = int(0.02 * SR)
    y = x.copy()
    for i in range(0, len(y) - f, f):
        if rng.random() < rate:
            y[i:i + f] = 0
    return y


# --------------------------------------------------------------------------- the chain
def draw_params(rng, split, tier=None):
    """Random augmentation parameters. `tier` = None (train/val draw) | 'clean' | an SNR in dB (test tiers)."""
    p = {"speed": float(rng.uniform(0.9, 1.1)), "helmet": bool(rng.random() < 0.4),
         "noise": list(rng.choice(CATEGORIES, size=int(rng.integers(1, 3)), replace=False)),
         "snr": float(rng.uniform(-5, 20)), "wind": bool(rng.random() < 0.3), "wind_snr": float(rng.uniform(0, 15)),
         "band": "narrow" if rng.random() < 0.5 else "wide",
         "codec": str(rng.choice(["opus", "amr", "g711", "none"], p=[0.4, 0.15, 0.15, 0.3])),
         "agc": bool(rng.random() < 0.5), "clip": bool(rng.random() < 0.2),
         "loss": float(rng.uniform(0.01, 0.03)) if rng.random() < 0.2 else 0.0, "gain_db": float(rng.uniform(-6, 3)),
         "lead_s": float(rng.uniform(0.1, 0.6)), "clean": False}
    if tier == "clean" or (tier is None and split == "train" and rng.random() < 0.1):
        p.update(clean=True, helmet=False, noise=[], wind=False, band="full", codec="none", agc=False, clip=False, loss=0.0)
    elif tier is not None:
        p.update(snr=float(tier), wind=False, loss=0.0, clip=False)
    return p


def apply_chain(x, p, rng, bank: NoiseBank, split):
    x = signal.resample_poly(x, 1000, int(round(1000 * p["speed"]))).astype(np.float32) if abs(p["speed"] - 1) > 1e-3 else x
    lead = np.zeros(int(p["lead_s"] * SR), np.float32)
    x = normalise(np.concatenate([lead, x, lead[: len(lead) // 2]]))
    if p["clean"]:
        return np.clip(x, -1, 1), {"codec_used": "none", "noise_src": []}
    if p["helmet"]:
        x = helmet(x, rng)
    srcs, n = [], len(x)
    if p["noise"]:
        nz = np.zeros(n, np.float32)
        for c in p["noise"]:
            seg, src = bank.get(c, n, rng, split)
            nz += seg / max(rms(seg), 1e-9) * float(rng.uniform(0.5, 1.0))
            srcs.append(f"{c}:{src}")
        x = mix_at_snr(x, nz, p["snr"])
    if p["wind"]:
        x = mix_at_snr(x, gen_wind(n, rng), p["wind_snr"])
        x = np.tanh(1.5 * x) / np.tanh(1.5)
    x = butter(x, "bandpass", [300, 3400]) if p["band"] == "narrow" else butter(x, "bandpass", [100, 7000])
    x, used = codec_roundtrip(x, p["codec"], rng)
    if p["agc"]:
        x = agc(x)
    if p["clip"]:
        x = np.clip(x * 3.0, -0.98, 0.98)
    if p["loss"]:
        x = packet_loss(x, rng, p["loss"])
    x = x * 10 ** (p["gain_db"] / 20)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak > 0.99:
        x = x * (0.99 / peak)
    return x.astype(np.float32), {"codec_used": used, "noise_src": srcs}


def crop(x, max_s=8.0, min_s=2.0, rng=None):
    n_max, n_min = int(max_s * SR), int(min_s * SR)
    if len(x) > n_max:
        x = x[:n_max]
    if len(x) < n_min:
        x = np.pad(x, (0, n_min - len(x)))
    return x


# --------------------------------------------------------------------------- manifest driver
def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def process(renders, out_dir, noise_dir=None, tiers=("clean", 20, 10, 5, 0), babble_from_train=True, man_path=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man_path = Path(man_path or out / "manifest.jsonl")
    done = {r["clip_id"] for r in read_jsonl(man_path)} if man_path.exists() else set()
    pool = None
    if babble_from_train:  # babble = other TRAIN renders mixed together (never test speech)
        tr = [r for r in renders if r["split"] == "train"][:40]
        pool = [load(r["abs_path"]) for r in tr] if tr else None
    bank = NoiseBank(noise_dir, pool)
    print(f"[augment] real noise files per category: {bank.summary()}", flush=True)
    n_new = 0
    with open(man_path, "a", encoding="utf-8") as fman:
        for r in renders:
            x0 = load(r["abs_path"])
            jobs = [(f"{r['render_id']}__{'clean' if t == 'clean' else f'snr{t}'}", t) for t in tiers] if r["split"] == "test" \
                else [(f"{r['render_id']}__aug", None)]
            for clip_id, tier in jobs:
                if clip_id in done:
                    continue
                rng = rng_for(clip_id) if tier is None else rng_for(r["render_id"], "tier")  # same draw across tiers
                p = draw_params(rng, r["split"], tier)
                rng2 = rng_for(clip_id, "apply")
                y, info = apply_chain(x0, p, rng2, bank, r["split"])
                y = crop(y)
                rel = f"{r['split']}/{clip_id}.wav"
                save(out / rel, y)
                row = {"clip_id": clip_id, "path": rel, "split": r["split"], **{k: r[k] for k in LABEL_KEYS},
                       "script_id": r["script_id"], "render_id": r["render_id"], "speaker": r["speaker"],
                       "kind": r.get("kind"), "register": r.get("register"), "ambiguous": bool(r.get("ambiguous", False)),
                       "group": r.get("group"),
                       "tier": ("clean" if tier == "clean" else f"snr{tier}") if tier is not None else ("clean" if p["clean"] else "train_mix"),
                       "aug": {**p, **info}, "duration_s": round(len(y) / SR, 2)}
                fman.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_new += 1
    print(f"[augment] {n_new} new clips -> {out} (manifest {man_path})")
    return man_path


def _selftest():
    rng = rng_for("selftest")
    t = np.arange(int(2.5 * SR)) / SR  # a vowel-ish test signal with syllable envelope
    x = (np.sin(2 * np.pi * 140 * t) + 0.5 * np.sin(2 * np.pi * 700 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t))
    x = (x * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2)).astype(np.float32) * 0.3
    bank = NoiseBank()
    for c in CATEGORIES:
        seg, src = bank.get(c, SR * 2, rng, "train")
        assert np.isfinite(seg).all() and rms(seg) > 0, c
    for snr in (20, 10, 0):
        nz = gen_traffic(len(x), rng)
        y = mix_at_snr(x, nz, snr)
        est = 20 * np.log10(active_rms(x) / rms(y - x))
        assert abs(est - snr) < 0.5, (snr, est)
    for tier in ("clean", 20, 5, None):
        p = draw_params(rng_for("p", tier), "test" if tier is not None else "train", tier)
        y, info = apply_chain(x, p, rng_for("a", tier), bank, "test")
        y = crop(y)
        assert np.isfinite(y).all() and np.abs(y).max() <= 1.0 and 2 * SR <= len(y) <= 8 * SR, tier
    y, used = codec_roundtrip(x, "opus", rng)
    assert len(y) == len(x) and used in ("opus", "g711")
    a = draw_params(rng_for("r1", "tier"), "test", 20)
    b = draw_params(rng_for("r1", "tier"), "test", 5)
    assert a["noise"] == b["noise"] and a["helmet"] == b["helmet"] and a["codec"] == b["codec"], "tiers must share the draw"
    with tempfile.TemporaryDirectory() as d:  # real-noise folders: test split must only use test files
        for sp in ("train", "test"):
            save(Path(d, "traffic", sp, f"{sp}_a.wav"), (0.1 * rng.standard_normal(SR)).astype(np.float32))
        nb = NoiseBank(d, p_real=1.0)
        assert nb.summary()["traffic"] == {"train": 1, "test": 1}
        _, src = nb.get("traffic", SR, rng, "test")
        assert src == "file"
    print(f"augment self-test OK (ffmpeg encoders: {sorted(ffmpeg_encoders()) or 'none -> G.711 fallback'})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts-manifest")
    ap.add_argument("--out")
    ap.add_argument("--noise-dir", default=None)
    ap.add_argument("--scripts", default=None, help="built scripts.jsonl: source of truth for labels")
    ap.add_argument("--tiers", default="clean,20,10,5,0")
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        _selftest()
        return 0
    man = Path(a.tts_manifest)
    renders = read_jsonl(man)
    if a.splits:
        renders = [r for r in renders if r["split"] in set(a.splits)]
    if a.scripts:
        by = {x["id"]: x for x in read_jsonl(a.scripts)}
        missing = sorted({r["script_id"] for r in renders} - set(by))
        if missing:
            print(f"[augment] {len(missing)} rendered scripts are not in --scripts (dropped): {missing[:5]}")
        renders = [r for r in renders if r["script_id"] in by]
        for r in renders:
            sc = by[r["script_id"]]
            if sc["split"] != r["split"]:
                raise SystemExit(f"[augment] {r['render_id']}: split changed {r['split']} -> {sc['split']}; re-render it")
            r.update({k: sc.get(k) for k in (*LABEL_KEYS, "kind", "register", "group")}, ambiguous=bool(sc.get("ambiguous", False)))
    for r in renders:
        p = Path(r["path"])
        r["abs_path"] = str(p if p.is_absolute() else man.parent / p)
    tiers = tuple(t if t == "clean" else int(t) for t in a.tiers.split(","))
    process(renders, a.out, a.noise_dir, tiers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
