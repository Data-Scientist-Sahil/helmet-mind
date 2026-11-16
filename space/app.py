"""Rider Voice Decisions — dispatch-console demo (Gradio, ZeroGPU or local CPU).

    MOCK=1 python app.py          # no weights needed; for layout and screenshots
    python app.py                 # loads run/ (or $RUN_DIR / $RUN_ZIP)

Environment:
  MOCK=1                 skip the backbone and head; deterministic filename-based predictor
  RUN_DIR                trained head folder (default ./run): head.pt, config.json, t_entropy.json, thresholds.json
  RUN_ZIP                results zip containing out/runs/<RUN_NAME>/...; extracted into RUN_DIR on start
  RUN_NAME               run name inside the zip (default m1v3)
  HF_MODEL_REPO          Hub model repo to snapshot into RUN_DIR at startup (e.g. sahildl/rider-voice-decisions-m1v3)
  HF_TOKEN               read token for a private HF_MODEL_REPO (Space secret); never logged
  THRESHOLDS             optional thresholds json merged over the run's (default ./thresholds.json)
  MODEL_ID / MODEL_DIR   backbone (default Darkrider/nirnaya-flash-1.0)
  PRECISION              fp16 (default on CUDA) | bf16 | fp32 (CPU)
  NIRNAYA_DIR            nirnaya-inference checkout (cloned at a pinned commit when missing)
"""

from __future__ import annotations

import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "src").exists() else HERE.parent
sys.path[:0] = [str(ROOT)]

from src.alert_rules import DEFAULTS, card, consistent  # noqa: E402

# --------------------------------------------------------------------------- config
def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


THRESHOLDS_PATH = Path(os.environ.get("THRESHOLDS", ROOT / "thresholds.json"))
BASE_THRESHOLDS = {**DEFAULTS, **_load_json(THRESHOLDS_PATH)}
CURVE_PATH = ROOT / "threshold_curve.json"
CURVE = _load_json(CURVE_PATH)

LABELS = {
    "situation": {
        "customer_unreachable": "Customer unreachable",
        "wrong_address": "Wrong address",
        "restaurant_delay": "Restaurant delay",
        "order_issue_at_pickup": "Order issue at pickup",
        "food_spilled_or_damaged": "Food spilled / damaged",
        "vehicle_breakdown": "Vehicle breakdown",
        "road_blocked_or_weather": "Road blocked / weather",
        "payment_issue": "Payment issue",
        "customer_dispute": "Customer dispute",
        "accident_or_injury": "Accident / injury",
        "harassment_or_threat": "Harassment / threat",
        "other_or_unclear": "Other / unclear",
    },
    "action": {
        "wait": "Wait here",
        "call_customer": "Call the customer",
        "contact_support": "Talk to support",
        "request_reassign": "Ask to reassign",
        "return_to_restaurant": "Go back to restaurant",
        "sos": "SOS",
    },
    "language": {"hinglish": "Hinglish", "hindi": "Hindi", "english": "English"},
}

SITUATIONS = list(LABELS["situation"])
ACTIONS = list(LABELS["action"])
LANGUAGES = list(LABELS["language"])

# Curated held-out synthetic test clips. Avoids the weak scripts
# wa_*_001, disp_hing_004, pay_hind_005, acc_*_010.
PRESETS = [
    ("SOS · Followed", "01_sos_being_followed_hinglish_10dB.wav"),
    ("SOS · Accident", "02_sos_accident_short_hinglish_10dB.wav"),
    ("Check-in · Safe now", "03_checkin_abused_now_safe_hindi_10dB.wav"),
    ("Unreachable · Gate locked", "04_call_customer_gate_locked_hinglish_10dB.wav"),
    ("Wait · On the way", "05_wait_customer_coming_hinglish_10dB.wav"),
    ("Support · Three calls", "06_support_three_calls_hinglish_10dB.wav"),
    ("Payment · Fetching cash", "07_wait_customer_fetching_cash_hindi_10dB.wav"),
    ("Dispute · Open packet", "08_support_dispute_open_packet_hinglish_10dB.wav"),
    ("Wrong pin", "09_call_customer_wrong_pin_hinglish_5dB.wav"),
]


def _preset_path(name: str) -> Path:
    return ROOT / "examples" / name


PRESET_PATHS = [(label, _preset_path(fn)) for label, fn in PRESETS if _preset_path(fn).exists()]

# --------------------------------------------------------------------------- thresholds / curve
def merge_thresholds(sos_tau: float | None = None) -> dict:
    th = dict(BASE_THRESHOLDS)
    if sos_tau is not None:
        th["sos_tau"] = float(sos_tau)
    return th


def lookup_curve(sos_tau: float) -> dict | None:
    pts = CURVE.get("curve") or []
    if not pts:
        return None
    return min(pts, key=lambda r: abs(float(r["sos_tau"]) - float(sos_tau)))


def _pct(x) -> str:
    if x is None:
        return "—"
    return f"{100 * float(x):.1f}%"


def curve_html(sos_tau: float) -> str:
    pt = lookup_curve(sos_tau)
    checkin = BASE_THRESHOLDS.get("checkin_tau", "—")
    if not pt:
        return (
            f'<div class="curve-pending">Curve pending — no held-out predictions were available when this '
            f"file was built. Run scripts/build_threshold_curve.py on a pred_test.jsonl to fill "
            f"threshold_curve.json. Check-in threshold is fixed at {checkin}.</div>"
        )
    src = CURVE.get("caption") or CURVE.get("source", "held-out eval")
    return (
        '<div class="curve-row">'
        f"<span>False SOS {_pct(pt['false_sos'])}</span>"
        f"<span>Safety recall {_pct(pt['safety_recall'])}</span>"
        f"<span>SOS recall (hard) {_pct(pt['sos_recall_hard'])}</span>"
        "</div>"
        f'<div class="curve-row"><span>Check-ins on normal {_pct(pt["checkin_on_normal"])}</span>'
        f"<span>Check-in τ {checkin}</span></div>"
        f'<div class="curve-source">{_esc(src)}</div>'
    )


# --------------------------------------------------------------------------- rendering
def _esc(s) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def banner_html(kind: str, latency_s: float | None, primary: str | None = None) -> str:
    if latency_s is None:
        meta = "⏱ —"
    elif latency_s < 0.05:
        meta = f"⏱ {max(latency_s, 0) * 1000:.1f} ms"
    else:
        meta = f"⏱ {latency_s:.2f} s"
    if kind == "sos":
        cls, label = "alert--sos", "SOS · RIDER UNSAFE · CALLING SUPPORT"
    elif kind == "checkin":
        cls, label = "alert--checkin", "ARE YOU SAFE? · CHECK-IN"
    elif kind == "result":
        act = LABELS["action"].get(primary or "", (primary or "no alert").replace("_", " "))
        cls, label = "alert--idle", f"NO ALERT · {act.upper()}"
        meta = meta + " · GPU" if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "") else meta
    else:
        cls, label = "alert--idle", "LISTENING · NO ALERT"
    return f'<div class="alert {cls}"><div class="alert-label">{_esc(label)}</div><div class="alert-meta">{_esc(meta)}</div></div>'


def chips_html(answers: dict | None, kind: str | None = None) -> str:
    def one(q, raw):
        text = LABELS.get(q, {}).get(raw, (raw or "—").replace("_", " "))
        extra = " is-sos" if (q == "action" and raw == "sos") or kind == "sos" and q == "action" else ""
        extra = extra or (" is-empty" if not raw else "")
        return (
            f'<div class="chip-row"><div class="chip-key">{q}</div>'
            f'<div class="chip-val{extra}">{_esc(text)}</div></div>'
        )

    if not answers:
        return one("situation", None) + one("action", None) + one("language", None)
    return (
        one("situation", answers["situation"]["choice"])
        + one("action", answers["action"]["choice"])
        + one("language", answers["language"]["choice"])
    )


def confidence_html(answers: dict | None, kind: str | None = None) -> str:
    accent = " is-sos" if kind == "sos" else (" is-checkin" if kind == "checkin" else "")
    rows = []
    heads = [("situation", "situation"), ("action", "action"), ("language", "language")]
    for label, q in heads:
        if answers and q in answers and "probabilities" in answers[q]:
            p = answers[q]["probabilities"].get(answers[q]["choice"], 0.0)
        else:
            p = 0.0
        rows.append(
            f'<div class="conf-row"><div class="conf-name">{label}</div>'
            f'<div class="conf-track"><div class="conf-fill{accent}" style="width:{100 * p:.1f}%"></div></div>'
            f'<div class="conf-num">{p:.2f}</div></div>'
        )
    return '<div class="conf-block"><div class="conf-head">Confidence</div>' + "".join(rows) + "</div>"


def counter_html(n: int) -> str:
    return (
        f'<div class="counter-block"><div class="counter-label">Check-ins this session</div>'
        f'<div class="counter-value">{int(n):02d}</div></div>'
    )


IDLE_SPEC = None


def _blank_spec() -> Image.Image:
    global IDLE_SPEC
    if IDLE_SPEC is None:
        img = Image.new("RGB", (1280, 240), (5, 5, 5))
        IDLE_SPEC = img
    return IDLE_SPEC


def spectrogram_image(path: str | None) -> Image.Image:
    if not path or not Path(path).exists():
        return _blank_spec()
    try:
        import soundfile as sf

        y, sr = sf.read(path, always_2d=False)
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        y = np.asarray(y, dtype=np.float32)
        if y.size < 16:
            return _blank_spec()
        nfft = 512
        hop = 128
        # Hann-windowed STFT
        window = np.hanning(nfft).astype(np.float32)
        if y.size < nfft:
            y = np.pad(y, (0, nfft - y.size))
        frames = 1 + (y.size - nfft) // hop
        # limit frames for a short console strip
        frames = max(1, min(frames, 400))
        spec = np.empty((nfft // 2 + 1, frames), dtype=np.float32)
        for i in range(frames):
            sl = y[i * hop : i * hop + nfft]
            if sl.size < nfft:
                sl = np.pad(sl, (0, nfft - sl.size))
            spec[:, i] = np.abs(np.fft.rfft(sl * window))
        db = 20.0 * np.log10(np.maximum(spec, 1e-8))
        db = db - db.max()
        fig, ax = plt.subplots(figsize=(10.5, 1.7), dpi=120, facecolor="#050505")
        ax.set_facecolor("#050505")
        ax.imshow(db, aspect="auto", origin="lower", cmap="magma", interpolation="nearest", vmin=-80, vmax=0)
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor="#050505", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    except Exception:
        return _blank_spec()


# --------------------------------------------------------------------------- mock + real engines
def _probs(keys: list[str], choice: str, p: float) -> dict[str, float]:
    rest = (1.0 - p) / max(len(keys) - 1, 1)
    return {k: (p if k == choice else rest) for k in keys}


# Filename keywords -> a deterministic, schema-identical decision (used only when MOCK=1
# or when no head.pt is present). These are staged examples, not model outputs.
_MOCK_BY_PREFIX = {
    "01_sos_being_followed": ("harassment_or_threat", "sos", "hinglish", 0.93, 0.81, 0.88),
    "02_sos_accident_short": ("accident_or_injury", "sos", "hinglish", 0.91, 0.86, 0.79),
    "03_checkin_abused_now_safe": ("harassment_or_threat", "contact_support", "hindi", 0.84, 0.07, 0.72),
    "04_call_customer_gate_locked": ("customer_unreachable", "call_customer", "hinglish", 0.89, 0.01, 0.02),
    "05_wait_customer_coming": ("customer_unreachable", "wait", "hinglish", 0.82, 0.01, 0.02),
    "06_support_three_calls": ("customer_unreachable", "contact_support", "hinglish", 0.80, 0.02, 0.03),
    "07_wait_customer_fetching_cash": ("payment_issue", "wait", "hindi", 0.86, 0.01, 0.01),
    "08_support_dispute_open_packet": ("customer_dispute", "contact_support", "hinglish", 0.78, 0.02, 0.03),
    "09_call_customer_wrong_pin": ("wrong_address", "call_customer", "hinglish", 0.77, 0.01, 0.02),
}


def _mock_tuple(path: str | None) -> tuple:
    stem = Path(path or "").name
    for key, val in _MOCK_BY_PREFIX.items():
        if stem.startswith(key):
            return val
    return ("other_or_unclear", "contact_support", "english", 0.55, 0.02, 0.03)


def mock_decide(path: str, th: dict) -> dict:
    sit, act, lang, p_top, p_sos, p_safe = _mock_tuple(path)
    sit_p = _probs(SITUATIONS, sit, p_top)
    # keep a visible safety-situation mass on the check-in / SOS presets
    if sit in {"accident_or_injury", "harassment_or_threat"}:
        sit_p[sit] = max(sit_p[sit], p_safe)
        # renormalize roughly
        z = sum(sit_p.values()) or 1.0
        sit_p = {k: v / z for k, v in sit_p.items()}
    act_p = _probs(ACTIONS, act, p_top)
    act_p["sos"] = p_sos
    # renormalize actions after forcing p_sos
    others = [a for a in ACTIONS if a != "sos"]
    remain = max(0.0, 1.0 - p_sos)
    top_share = p_top if act != "sos" else 0.0
    rest = remain - top_share
    if act != "sos":
        act_p[act] = top_share
        spread = [a for a in others if a != act]
        for a in spread:
            act_p[a] = (rest / len(spread)) if spread else 0.0
    else:
        for a in others:
            act_p[a] = remain / len(others)
    s = sum(act_p.values()) or 1.0
    act_p = {k: v / s for k, v in act_p.items()}
    lang_p = _probs(LANGUAGES, lang, max(p_top, 0.7))
    answers = {
        "situation": {"type": "choice", "choice": sit, "probabilities": sit_p},
        "action": {"type": "choice", "choice": act, "probabilities": act_p},
        "safety": {"type": "noul", "noul": p_safe},
        "language": {"type": "choice", "choice": lang, "probabilities": lang_p},
    }
    answers, rep = consistent(answers, th)
    return {
        "answers": answers,
        "consistency": rep,
        "card": card(answers, th),
        "precision": "mock",
        "audio_s": None,
        "n_tokens": 0,
    }


def _want_mock() -> bool:
    if _truthy(os.environ.get("MOCK")):
        return True
    run = Path(os.environ.get("RUN_DIR", ROOT / "run"))
    zip_path = os.environ.get("RUN_ZIP")
    repo = (os.environ.get("HF_MODEL_REPO") or "").strip()
    return not (run / "head.pt").exists() and not zip_path and not repo


MOCK = _want_mock()
ENGINE = None
ENGINE_ERROR = None


def _extract_run_zip(zip_path: Path, run_name: str, dst: Path) -> None:
    import shutil
    import zipfile

    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        for f in ("head.pt", "config.json", "t_entropy.json", "thresholds.json"):
            hits = [n for n in names if n.endswith(f"/runs/{run_name}/{f}")]
            if not hits:
                hits = [n for n in names if f"/{run_name}/{f}" in n or n.endswith(f"/{run_name}/{f}")]
            if hits:
                with z.open(hits[0]) as r, open(dst / f, "wb") as w:
                    shutil.copyfileobj(r, w)


def _download_hf_model_repo(dst: Path) -> None:
    """Snapshot the trained head from HF_MODEL_REPO into RUN_DIR. Uses HF_TOKEN if set; never logs it."""
    repo = (os.environ.get("HF_MODEL_REPO") or "").strip()
    if not repo:
        return
    if (dst / "head.pt").exists() and (dst / "config.json").exists():
        return
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[rider-voice] downloading run files from {repo} -> {dst}", flush=True)
    snapshot_download(
        repo_id=repo,
        repo_type="model",
        local_dir=str(dst),
        token=token,
        allow_patterns=[
            "head.pt",
            "config.json",
            "t_entropy.json",
            "thresholds.json",
            "calibration.json",
        ],
    )


def _resolve_run_dir() -> Path:
    dst = Path(os.environ.get("RUN_DIR", ROOT / "run"))
    z = os.environ.get("RUN_ZIP")
    name = os.environ.get("RUN_NAME", "m1v3")
    if z and not (dst / "head.pt").exists():
        _extract_run_zip(Path(z), name, dst)
    if not (dst / "head.pt").exists():
        _download_hf_model_repo(dst)
    return dst


def _gpu_decorator():
    try:  # ZeroGPU: import `spaces` before torch; GPU is attached only inside @spaces.GPU
        import spaces  # type: ignore

        return spaces.GPU(duration=40)
    except ImportError:
        def passthrough(f):
            return f

        return passthrough


gpu = _gpu_decorator()


def _load_engine():
    global ENGINE, ENGINE_ERROR, BASE_THRESHOLDS
    if MOCK:
        return None
    try:
        import subprocess

        nirnaya = Path(os.environ.get("NIRNAYA_DIR", ROOT / "nirnaya-inference"))
        commit = "0fcd64f22b3013dd2bb1e268533bcc91accf87c6"
        if not (nirnaya / "mmclm").exists():
            subprocess.run(["git", "clone", "https://github.com/NeverInAsh/nirnaya-inference", str(nirnaya)], check=True)
            subprocess.run(["git", "-C", str(nirnaya), "checkout", commit], check=True)
        sys.path[:0] = [str(nirnaya)]
        import torch  # noqa: E402
        from src.engine import RiderVoiceEngine  # noqa: E402

        if os.environ.get("MODEL_DIR"):
            model = os.environ["MODEL_DIR"]
        else:
            from huggingface_hub import snapshot_download

            model = snapshot_download(os.environ.get("MODEL_ID", "Darkrider/nirnaya-flash-1.0"))
        cuda = torch.cuda.is_available()
        precision = os.environ.get("PRECISION", "fp16" if cuda else "fp32")
        run = _resolve_run_dir()
        th = dict(BASE_THRESHOLDS)
        run_th = _load_json(run / "thresholds.json")
        # run file first, then the Space defaults / THRESHOLDS override (round-3 two-level)
        merged = {**run_th, **th}
        BASE_THRESHOLDS = merged
        ENGINE = RiderVoiceEngine(
            model,
            str(run),
            device="cuda" if cuda else "cpu",
            precision=precision,
            placement="single" if cuda else "auto",
            thresholds=merged,
        )
        return ENGINE
    except Exception as exc:  # weights or backbone missing: stay in mock rather than crash the Space
        ENGINE_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"[rider-voice] real engine failed to load ({ENGINE_ERROR}); using MOCK predictor", flush=True)
        return None


@gpu
def _infer(path: str, th: dict) -> dict:
    if ENGINE is not None:
        # The engine already applies the run's thresholds; re-apply `th` so the slider can retune SOS.
        out = ENGINE.decide(path)
        ans, rep = consistent(out["answers"], th)
        out = {**out, "answers": ans, "consistency": rep, "card": card(ans, th)}
        return out
    return mock_decide(path, th)


def _warmup():
    if ENGINE is None:
        return
    try:
        import soundfile as sf

        silence = ROOT / "run" / "_warmup_silence.wav"
        silence.parent.mkdir(parents=True, exist_ok=True)
        if not silence.exists():
            sr = 16000
            sf.write(str(silence), np.zeros(int(0.4 * sr), dtype=np.float32), sr)
        _infer(str(silence), merge_thresholds())
        print("[rider-voice] warmup done", flush=True)
    except Exception as exc:
        print(f"[rider-voice] warmup skipped ({type(exc).__name__})", flush=True)


# --------------------------------------------------------------------------- sound hook (licensed files only; silent if absent)
def _sound_for(kind: str, muted: bool) -> str | None:
    if muted or kind not in {"sos", "checkin"}:
        return None
    p = ROOT / "assets" / "sounds" / f"{kind}.wav"
    return str(p) if p.exists() else None


# --------------------------------------------------------------------------- predict
def _mode_note() -> str:
    if ENGINE is not None:
        prec = getattr(ENGINE, "bb", None)
        prec = getattr(prec, "precision", "fp32") if prec is not None else "fp32"
        return f'<div class="mode-chip">run · {prec}</div>'
    return '<div class="mode-chip">demo mode · mock predictor (weights not loaded)</div>'


def predict(audio_path, checkins, muted, sos_tau, last_out):
    th = merge_thresholds(sos_tau)
    metrics = curve_html(float(sos_tau)) + _mode_note()
    if not audio_path:
        return (
            banner_html("idle", None),
            _blank_spec(),
            chips_html(None),
            confidence_html(None),
            metrics,
            counter_html(int(checkins or 0)),
            int(checkins or 0),
            None,
            last_out,
        )
    t0 = time.perf_counter()
    out = _infer(audio_path, th)
    # Time only the infer call. Prefer the engine's own counter when it ran the model.
    if ENGINE is not None and "latency_ms" in out:
        latency_s = float(out["latency_ms"]) / 1000.0
    else:
        latency_s = time.perf_counter() - t0
    kind = out["card"]["kind"]
    banner_kind = kind if kind in {"sos", "checkin"} else "result"
    new_checkins = int(checkins or 0)
    if kind == "checkin":
        new_checkins += 1
    packed = json.dumps(
        {"answers": out["answers"], "card": out["card"], "latency_s": latency_s, "path": audio_path},
        ensure_ascii=False,
    )
    return (
        banner_html(banner_kind, latency_s, out["answers"]["action"]["choice"]),
        spectrogram_image(audio_path),
        chips_html(out["answers"], kind),
        confidence_html(out["answers"], kind),
        metrics,
        counter_html(new_checkins),
        new_checkins,
        _sound_for(kind, bool(muted)),
        packed,
    )


def retune(sos_tau, last_out, checkins):
    """Slider: look up real curve numbers; re-apply the alert rule to the last answers without re-running the model."""
    th = merge_thresholds(sos_tau)
    metrics = curve_html(float(sos_tau)) + _mode_note()
    if not last_out:
        return banner_html("idle", None), chips_html(None), confidence_html(None), metrics, counter_html(int(checkins or 0))
    packed = json.loads(last_out)
    answers, rep = consistent(packed["answers"], th)
    c = card(answers, th)
    kind = c["kind"]
    banner_kind = kind if kind in {"sos", "checkin"} else "result"
    return (
        banner_html(banner_kind, packed.get("latency_s"), answers["action"]["choice"]),
        chips_html(answers, kind),
        confidence_html(answers, kind),
        metrics,
        counter_html(int(checkins or 0)),
    )


def toggle_demo(playing: bool):
    playing = not bool(playing)
    label = "Stop demo" if playing else "Autoplay demo"
    return playing, gr.Timer(value=5, active=playing), label


def on_timer(idx, checkins, muted, sos_tau, last_out, playing):
    if not playing or not PRESET_PATHS:
        return (int(idx or 0),) + (gr.update(),) * 10
    n = len(PRESET_PATHS)
    idx = int(idx or 0) % n
    path = str(PRESET_PATHS[idx][1])
    idx = (idx + 1) % n
    outs = predict(path, checkins, muted, sos_tau, last_out)
    return (idx, path, *outs)


def on_preset(path, checkins, muted, sos_tau, last_out):
    return (path, *predict(path, checkins, muted, sos_tau, last_out))


# --------------------------------------------------------------------------- UI
CSS = (ROOT / "style.css").read_text(encoding="utf-8") if (ROOT / "style.css").exists() else ""
DARK_JS = """
() => {
  document.documentElement.classList.add('dark');
  document.body.classList.add('dark');
}
"""
HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
"""

DEFAULT_SOS = float(BASE_THRESHOLDS.get("sos_tau", 0.42))

theme = gr.themes.Base(
    primary_hue="red",
    secondary_hue="zinc",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Space Grotesk"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    radius_size="sm",
).set(
    body_background_fill="#050505",
    body_text_color="#EDEDED",
    block_background_fill="#0D0D0F",
    block_border_color="#232326",
    block_radius="6px",
    border_color_primary="#232326",
    button_primary_background_fill="#FF2D2D",
    button_primary_text_color="#050505",
    slider_color="#FF2D2D",
)

with gr.Blocks(title="Rider Voice Decisions", fill_height=True, fill_width=True, elem_id="dispatch") as demo:
    checkins = gr.State(0)
    last_out = gr.State("")
    playing = gr.State(False)
    demo_idx = gr.State(0)
    timer = gr.Timer(value=5, active=False)

    banner = gr.HTML(banner_html("idle", None), elem_id="alert-banner")

    with gr.Row(elem_id="main-row", equal_height=True):
        with gr.Column(scale=5, elem_id="left-col"):
            audio = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="VOICE NOTE",
                elem_id="audio-in",
                waveform_options=gr.WaveformOptions(
                    waveform_color="#4A4A4F",
                    waveform_progress_color="#FF2D2D",
                    show_recording_waveform=True,
                ),
            )
            spec = gr.Image(
                value=_blank_spec(),
                label="SPECTROGRAM",
                elem_id="spectrogram",
                show_label=True,
                interactive=False,
                type="pil",
                format="png",
                buttons=[],
            )
            with gr.Column(elem_id="presets-wrap"):
                gr.Markdown("PRESETS", elem_id="presets-label")
                with gr.Row(elem_id="preset-grid"):
                    preset_buttons = []
                    for label, path in PRESET_PATHS:
                        preset_buttons.append(
                            (gr.Button(label, elem_classes=["preset-btn"], size="sm"), str(path))
                        )
            with gr.Row(elem_id="controls-row"):
                demo_btn = gr.Button("Autoplay demo", elem_id="demo-toggle", size="sm")
                mute = gr.Checkbox(label="Mute alerts", value=True, elem_id="mute-toggle")
                count_html = gr.HTML(counter_html(0), elem_id="checkin-counter")
            sound = gr.Audio(label="alert", autoplay=True, visible=False, elem_id="hidden-sound")

        with gr.Column(scale=6, elem_id="right-col"):
            chips = gr.HTML(chips_html(None), elem_id="chips")
            conf = gr.HTML(confidence_html(None), elem_id="confidence")
            slider = gr.Slider(
                minimum=0.01,
                maximum=0.99,
                step=0.01,
                value=DEFAULT_SOS,
                label="SOS THRESHOLD",
                info="Held-out eval curve. Check-in threshold stays fixed.",
                elem_id="sos-slider",
            )
            metrics = gr.HTML(curve_html(DEFAULT_SOS) + _mode_note(), elem_id="curve-panel")

    gr.HTML(
        '<div class="site-footer">Inspired by delivery apps · Not affiliated with Zomato or Swiggy · '
        "Synthetic voices, no real rider data · Built by Sahil Builds AI</div>",
        elem_id="site-footer",
    )

    predict_inputs = [audio, checkins, mute, slider, last_out]
    predict_outputs = [banner, spec, chips, conf, metrics, count_html, checkins, sound, last_out]

    audio.stop_recording(predict, predict_inputs, predict_outputs)
    audio.upload(predict, predict_inputs, predict_outputs)

    for btn, path in preset_buttons:
        btn.click(
            lambda checkins, muted, sos_tau, last, p=path: (p, *predict(p, checkins, muted, sos_tau, last)),
            inputs=[checkins, mute, slider, last_out],
            outputs=[audio, *predict_outputs],
        )

    slider.release(retune, [slider, last_out, checkins], [banner, chips, conf, metrics, count_html])
    slider.change(lambda t: curve_html(float(t)) + _mode_note(), slider, metrics)

    demo_btn.click(toggle_demo, playing, [playing, timer, demo_btn])
    timer.tick(on_timer, [demo_idx, checkins, mute, slider, last_out, playing], [demo_idx, audio, *predict_outputs])


def _boot():
    _load_engine()
    # Warm the real model when a GPU is already visible (local or a persistent Space).
    # On ZeroGPU the device is only attached inside @spaces.GPU, so the first request warms it there.
    try:
        import torch

        if ENGINE is not None and torch.cuda.is_available():
            _warmup()
    except Exception:
        pass


_boot()


if __name__ == "__main__":
    css_path = ROOT / "style.css"
    demo.launch(
        theme=theme,
        css=CSS,
        js=DARK_JS,
        head=HEAD,
        footer_links=[],
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        allowed_paths=[str(ROOT / "examples"), str(ROOT / "assets"), str(ROOT)],
        ssr_mode=False,
        show_error=True,
    )
