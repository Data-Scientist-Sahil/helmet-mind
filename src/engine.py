"""End-to-end: a voice note -> calibrated answers for all four questions -> the one-tap card. For the demo and the
latency benchmark. Uses exactly the extraction settings the head was trained with (read from the run's config).

    python src/engine.py --model <model dir> --run runs/m1 clip.wav [clip2.wav ...] [--bench 100] [--precision fp16]

--precision overrides the precision the features were extracted with (for latency runs; check accuracy with an
fp16 extraction of the test split before serving that way - notebook 02 does both).

Also usable as a Decision Index engine: `RiderVoiceEngine(...)(state, questions, audios=[...])`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.backbone import Backbone  # noqa: E402
from src.decide import answers, calibrated_probs, consistent, head_logits, load_run  # noqa: E402
from src.product import DEFAULTS, card  # noqa: E402
from src.request import build_request  # noqa: E402


class RiderVoiceEngine:
    name = "rider-voice-decisions"

    def __init__(self, model: str, run: str, device=None, precision=None):
        self.head_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.head, self.jh, self.fit, self.th, cfg, self.scope = load_run(run, self.head_device)
        fm = cfg.get("features_meta", {})
        self.bb = Backbone(model, precision=precision or fm.get("precision"), mem_cap=fm.get("mem_cap", 256),
                           twice_max=fm.get("twice_max", 3000), device=None if self.head_device != "cpu" else "cpu",
                           state_variant=fm.get("state_variant", "post_norm"))
        self.state, self.questions = build_request()

    @torch.no_grad()
    def decide(self, audio) -> dict:
        t0 = time.perf_counter()
        f = self.bb.features(self.state, self.questions, audio)
        rec = {"memory": f["memory"], "items": f["items"]}
        lg = head_logits(self.head, self.jh, [rec], self.head_device, self.scope)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ans, rep = consistent(answers(calibrated_probs(lg, self.fit)), self.th.get("safety_tau", DEFAULTS["safety_tau"]))
        return {"answers": ans, "consistency": rep, "card": card(ans, self.th),
                "latency_ms": round(1000 * (time.perf_counter() - t0), 1), "precision": self.bb.precision,
                "audio_s": f["meta"]["audio_s"], "n_tokens": f["meta"]["n_tokens"]}

    def __call__(self, state, questions, audios=()):
        """Decision Index-style call for the fixed rider request (state/questions must be this repo's spec)."""
        if not audios:
            raise ValueError("this engine needs one audio clip")
        out = self.decide(audios[0])
        return {"model": self.name, "answers": out["answers"]}, {"latency_ms": out["latency_ms"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--bench", type=int, default=0, help="time N calls (after 5 warm-up) and report p50/p95")
    ap.add_argument("--device", default=None)
    ap.add_argument("--precision", default=None, choices=[None, "fp32", "fp16", "bf16"])
    a = ap.parse_args(argv)
    eng = RiderVoiceEngine(a.model, a.run, a.device, a.precision)
    for c in a.clips:
        out = eng.decide(c)
        print(json.dumps({"clip": c, **out}, indent=1, ensure_ascii=False))
    if a.bench:
        for i in range(5):
            eng.decide(a.clips[i % len(a.clips)])
        ts = [eng.decide(a.clips[i % len(a.clips)])["latency_ms"] for i in range(a.bench)]
        ts.sort()
        print(json.dumps({"n": len(ts), "precision": eng.bb.precision, "p50_ms": statistics.median(ts), "p95_ms": ts[int(0.95 * (len(ts) - 1))],
                          "mean_ms": round(statistics.mean(ts), 1),
                          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
