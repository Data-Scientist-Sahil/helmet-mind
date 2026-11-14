"""Scripts -> clean multi-voice speech with AI4Bharat Indic Parler-TTS (Apache-2.0). Resumable.

    pip install git+https://github.com/huggingface/parler-tts.git   # pins transformers==4.46.1: use a SEPARATE
                                                                     # environment / notebook from the backbone
    python src/tts.py --scripts data/scripts/m1_wedge.jsonl --out data/audio/tts [--plan-only]

The model is gated (auto-approved): accept its terms on the Hub once and set HF_TOKEN.

Voices (named speakers of the model):
  hindi    train: Rohit, Divya, Aman          held out (val/test only): Rani
  english  train: 17 Indian-English voices    held out: Ravi, Vikas, Riya, Kavya
  hinglish = half Hindi voices reading the Devanagari `text_tts`, half English voices reading the romanised `text`.
Renders per script: train 8 (4 voices x 2 captions), val/test 4 (held-out voices x 2 captions).
Captions follow the script's register (calm / irritated / tired / panicked); every caption asks for "very clear audio"
because noise is added later, at controlled SNR, by src/augment.py.
Each render is seeded; renders whose length is implausible for the text are re-sampled (up to 2 retries).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HINDI_TRAIN, HINDI_HELD = ["Rohit", "Divya", "Aman"], ["Rani"]
EN_ALL = ["Thoma", "Mary", "Swapna", "Dinesh", "Meera", "Jatin", "Aakash", "Sneha", "Kabir", "Tisha", "Chingkhei", "Thoiba",
          "Priya", "Tarun", "Gauri", "Nisha", "Raghav", "Kavya", "Ravi", "Vikas", "Riya"]
EN_HELD = ["Ravi", "Vikas", "Riya", "Kavya"]
EN_TRAIN = [v for v in EN_ALL if v not in EN_HELD]
CAPTIONS = {
    "calm": ["{v}'s voice is calm and clear, at a moderate pace, with very clear audio and a close recording.",
             "{v} speaks at a moderate pace in a slightly low-pitched, relaxed voice, with very clear audio."],
    "irritated": ["{v} speaks fast with an annoyed, expressive tone, with very clear audio and a close recording.",
                  "{v}'s voice is slightly high-pitched and impatient, quite fast, with very clear audio."],
    "tired": ["{v}'s voice is tired, slightly slow and monotone, with very clear audio and no background noise.",
              "{v} speaks slowly in a low, flat voice, with very clear audio and a close recording."],
    "panicked": ["{v} speaks very fast and urgently, with a highly expressive, anxious tone, very clear audio, close to the microphone.",
                 "{v}'s voice is high-pitched, breathless and scared, speaking fast, with very clear audio."],
}
LABEL_KEYS = ("situation", "action", "safety", "language")


def _h(*k) -> int:
    return int(hashlib.sha256("|".join(map(str, k)).encode()).hexdigest(), 16)


def plan(rows: list[dict]) -> list[dict]:
    """Every render to make: which voice, caption and text, for each script (deterministic)."""
    out = []
    for r in rows:
        held = r["split"] in ("val", "test")
        hi, en = (HINDI_HELD, EN_HELD) if held else (HINDI_TRAIN, EN_TRAIN)
        if r["language"] == "hindi":
            pool = [("hi", v) for v in hi]
        elif r["language"] == "english":
            pool = [("en", v) for v in en]
        else:
            pool = [("hi", v) for v in hi] + [("en", v) for v in en]
        n_voices = 2 if held else 4
        start = _h(r["id"]) % len(pool)
        if r["language"] == "hinglish":  # balance Devanagari (hi voice) and romanised (en voice) renders
            his = [p for p in pool if p[0] == "hi"]
            ens = [p for p in pool if p[0] == "en"]
            voices = [his[(start + i) % len(his)] for i in range(n_voices // 2)] + [ens[(start + i) % len(ens)] for i in range(n_voices - n_voices // 2)]
        else:
            voices = [pool[(start + i) % len(pool)] for i in range(min(n_voices, len(pool)))]
            while len(voices) < n_voices:  # fewer held-out voices than renders: reuse with the other caption seed
                voices.append(voices[len(voices) % len(pool)])
        caps = CAPTIONS.get(r.get("register") or "calm", CAPTIONS["calm"])
        for vi, (vl, v) in enumerate(voices):
            for ci in range(2):  # train: 4 voices x 2 captions = 8; val/test: 2 voices x 2 captions = 4
                rid = f"{r['id']}__{v.lower()}{vi}_{ci}"
                text = r["text_tts"] if vl == "hi" else r["text"]
                out.append({"render_id": rid, "script_id": r["id"], "split": r["split"], "speaker": v, "voice_lang": vl,
                            "caption": caps[ci].format(v=v), "text": text, "seed": _h(rid) % (2**31),
                            "kind": r.get("kind"), "register": r.get("register"), "ambiguous": bool(r.get("ambiguous", False)),
                            "group": r.get("group"), **{k: r[k] for k in LABEL_KEYS}})
    return out


def plausible(n_samples, sr, text):
    sec, words = n_samples / sr, max(1, len(text.split()))
    return 0.15 * words <= sec <= max(2.5, 0.9 * words) and sec <= 12


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--model", default="ai4bharat/indic-parler-tts")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0)
    a = ap.parse_args(argv)

    rows = [json.loads(x) for x in open(a.scripts, encoding="utf-8") if x.strip()]
    if a.splits:
        rows = [r for r in rows if r["split"] in set(a.splits)]
    renders = plan(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = out / "tts_manifest.jsonl"
    done = {json.loads(x)["render_id"] for x in open(man, encoding="utf-8")} if man.exists() else set()
    todo = [r for r in renders if r["render_id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    by_split = {}
    for r in renders:
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1
    print(f"[tts] {len(rows)} scripts -> {len(renders)} renders {by_split}; {len(done)} done, {len(todo)} to do", flush=True)
    if a.plan_only:
        (out / "tts_plan.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in renders), encoding="utf-8")
        return 0

    import numpy as np
    import soundfile as sf
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from scipy import signal
    from transformers import AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    model = ParlerTTSForConditionalGeneration.from_pretrained(a.model, torch_dtype=dt).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(a.model)
    dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr = model.config.sampling_rate
    t0, n_ok, n_flag = time.time(), 0, 0
    with open(man, "a", encoding="utf-8") as fman:
        for i, r in enumerate(todo):
            d = dtok(r["caption"], return_tensors="pt").to(dev)
            p = tok(r["text"], return_tensors="pt").to(dev)
            audio, ok = None, False
            for attempt in range(3):
                torch.manual_seed(r["seed"] + attempt)
                with torch.no_grad():
                    g = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                       prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
                audio = g.float().cpu().numpy().squeeze()
                if plausible(len(audio), sr, r["text"]):
                    ok = True
                    break
            y = signal.resample_poly(audio, 16000, sr).astype(np.float32) if sr != 16000 else audio.astype(np.float32)
            nz = np.flatnonzero(np.abs(y) > 0.01 * (np.abs(y).max() + 1e-9))  # trim silence at both ends
            if len(nz):
                y = y[max(0, nz[0] - 1600): nz[-1] + 1600]
            rel = f"{r['split']}/{r['render_id']}.wav"
            (out / r["split"]).mkdir(parents=True, exist_ok=True)
            sf.write(out / rel, y, 16000, subtype="PCM_16")
            n_flag += not ok
            fman.write(json.dumps({**r, "path": rel, "duration_s": round(len(y) / 16000, 2), "length_ok": ok}, ensure_ascii=False) + "\n")
            fman.flush()
            n_ok += 1
            if (i + 1) % 10 == 0 or i == 0:
                el = time.time() - t0
                print(f"[tts] {i + 1}/{len(todo)}  {el / (i + 1):.2f} s/render  eta {el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min",
                      flush=True)  # flush: Kaggle's committed log otherwise shows nothing until the end
            if a.max_minutes and time.time() - t0 > a.max_minutes * 60:
                print("[tts] time budget reached - re-run to continue")
                break
    print(f"[tts] {n_ok} renders written ({n_flag} flagged for odd length - listen to a few) in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
