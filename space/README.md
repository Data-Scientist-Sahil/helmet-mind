---
title: Rider Voice Decisions
emoji: 🎧
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Voice note to situation, next action, language, and a two-level safety alert
---

# Rider Voice Decisions

A dispatch console for short delivery-rider voice notes — English, Hindi, or Hinglish, including street noise. One forward pass of a frozen audio language model plus a trained head returns calibrated probabilities for four questions:

| Question | What it answers |
|---|---|
| **Situation** | Why the rider is calling in (unreachable customer, wrong address, accident, …) |
| **Next action** | Wait, call the customer, talk to support, reassign, return, or SOS |
| **Safety** | Whether anyone appears to be in danger |
| **Language** | Hinglish, Hindi, or English |

The model **suggests**; the rider confirms with one tap. Nothing is executed on its own, except that an unanswered "Are you safe?" check-in escalates to SOS after a short timeout.

## Two-level safety alerts

| Alert | When it fires |
|---|---|
| **SOS** | p(action = SOS) is at or above the SOS threshold |
| **Are you safe?** check-in | The combined safety score — max(p(safety), p(accident or harassment), p(SOS)) — is at or above the check-in threshold |
| **No alert** | A suggested next action, or a choice between the top two when confidence is low |

The SOS-threshold slider on the console shows **False SOS** and **safety recall** looked up from a held-out evaluation curve (`threshold_curve.json`), not invented numbers. The default operating point shipped in `thresholds.json` is the round-3 two-level setting: SOS τ = 0.42, check-in τ = 0.04, safety τ = 0.02.

## How to run

Locally, with a mock predictor (no weights required):

```bash
MOCK=1 python app.py
```

With a trained head, after placing `head.pt`, `config.json`, `t_entropy.json` and `thresholds.json` in `run/` — or extracting them from a results zip:

```bash
python scripts/load_run.py --run-zip results/round3_training_results.zip --run-name m1v3 \
    --thresholds thresholds.json
python app.py
```

Useful environment variables: `RUN_DIR`, `RUN_ZIP`, `RUN_NAME` (default `m1v3`), `THRESHOLDS`, `MODEL_ID` / `MODEL_DIR` (default `Darkrider/nirnaya-flash-1.0`), `PRECISION` (`fp16` on GPU, `fp32` on CPU), `MOCK=1`.

On a Hugging Face Space, pick **ZeroGPU** hardware. The first request after a cold start loads the backbone; later requests are much faster. Inference is wrapped with `@spaces.GPU` and also runs on CPU when no GPU is present.

Rebuild the evaluation curve from a predictions file (never invents metrics):

```bash
python scripts/build_threshold_curve.py --pred path/to/pred_test.jsonl \
    --checkin-tau 0.04 --out threshold_curve.json
```

Preset clips are held-out synthetic test notes, not real rider recordings.

## Disclaimer

Inspired by delivery apps. **Not affiliated with Zomato or Swiggy.** Trained on synthetic speech mixed with real environmental noise. No real rider data. Built by Sahil Builds AI.
