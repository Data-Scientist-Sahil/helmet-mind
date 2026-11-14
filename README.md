# rider-voice-decisions — voice-only decisions for last-mile delivery riders

> **Status: work in progress.** A first synthetic run (M1, door-step wedge) has been done; round 2 of the data and
> training changes is built. Numbers will be published here after round 2 and a small real-voice test set.

A short voice note goes in — *"customer ka gate band hai, phone nahi utha raha"* — and **calibrated probabilities over a fixed set of decisions** come out, in one pass, with no transcription step:

| Question | Options |
|---|---|
| **situation** | customer_unreachable · wrong_address · restaurant_delay · order_issue_at_pickup · food_spilled_or_damaged · vehicle_breakdown · road_blocked_or_weather · payment_issue · customer_dispute · accident_or_injury · harassment_or_threat · other_or_unclear |
| **action** | wait · call_customer · contact_support · request_reassign · return_to_restaurant · sos |
| **safety** | yes / no — thresholded for high recall |
| **language** | hinglish · hindi · english |

A concept for Indian last-mile delivery, built on synthetic data. **Not affiliated with any delivery platform.**

## How it works

- **Backbone:** [nirnaya-flash-1.0](https://huggingface.co/Darkrider/nirnaya-flash-1.0) (Gemma 4 E4B, reads audio natively), **frozen**. The clip and a fixed text request (one state + four questions, each option written out) go in once; the final hidden states come out.
- **Head:** the joint decision head from [nirnaya-inference](https://github.com/NeverInAsh/nirnaya-inference) (`mmclm/jh.py`) reads the state, question and option spans and scores every option of every question at once. We warm-start from the released head and train only the head, on cached features — minutes on a free T4.
- **Calibration:** an entropy-aware temperature per question, refit on our validation split. It never changes the chosen option, only how confident it is.
- **Request format:** the [Decision Index](https://github.com/apolinario/decision-index) `state` + typed `questions` interface (`choice` with `criteria`, `noul` for yes/no). The whole decision set is one file: [`spec/decision_set.json`](spec/decision_set.json).

## Evaluation rules that matter

- **Intervals resample scripts, not clips.** Every script is rendered with several voices and noise levels; those
  clips are near-duplicates, so a clip-level bootstrap looks far more certain than the data is.
- **Scope.** A milestone covers only some situations (the M1 wedge: what happens at or on the way to the door).
  Out-of-scope labels are masked for every system, and macro-F1 is computed over the classes present in test.
- **Safety threshold** comes from out-of-fold predictions on training scripts, never from test.

## How it is meant to be used

The model **suggests; the rider confirms with one tap.** It never executes an action on its own.

- The top action is shown as one big button, with the runner-up and "talk to support" beside it.
- **Safety always escalates to a human.** The SOS card appears if p(safety) ≥ τ, *or* the top situation is an accident or a threat, *or* the top action is SOS. The model can add an escalation; nothing in the system lets it suppress one.
- When confidence is low (or the situation is `other_or_unclear`), the rider gets the top two actions plus support instead of a single suggestion.
- **Consistency:** when the safety flag fires, the suggested action is restricted to SOS or support, whatever the
  action head says on its own.

## Labels are derived, not hand-picked

Every training script carries its situation plus structured cues (`attempts`, `waited_min`, `near_restaurant`, …). The gold **action** and **safety** labels are computed from those cues by the rules in `spec/decision_set.json` (`src/labels.py`). For example, *customer unreachable* defaults to **call_customer**, but after two call attempts or ten minutes of waiting it becomes **contact_support**. Accidents and threats are always safety = yes, even when the rider says they are fine.

```bash
python src/labels.py     # self-test: 12 situations, 6 actions (all reachable), rule cases
python src/request.py    # prints the request and the exact prompt text the backbone reads
```

## Data

- **Scripts:** ~480 short rider utterances across Hinglish, Hindi and English (the first 130, for the door-step wedge, are in `data/scripts/m1_wedge.raw.jsonl`), including terse fragments and *hard negatives* that sound alarming but aren't ("aage accident hua hai, isliye traffic hai").
- **Synthetic audio:** multi-voice TTS ([Indic Parler-TTS](https://huggingface.co/ai4bharat/indic-parler-tts), Apache-2.0), then a noise and channel chain — traffic, horns, engine, wind, rain, crowd, a helmet filter, a phone-mic band-pass and a low-bitrate codec round-trip. Fixed-seed evaluation tiers at clean / 20 / 10 / 5 / 0 dB SNR.
- **Real test set:** a small set of clips recorded by people on ordinary phones, prompted by *situation cards* rather than scripts, so they say things in their own words. Used **only** for testing — never for training, tuning, thresholds or calibration.

Splits are by script and by speaker, never by clip, so no voice or noise variant of a test script is ever seen in training.

## Evaluation

Compared on the same clips and the same GPU (batch 1, warm):

- Whisper large-v3-turbo → LLM (option log-probabilities), the usual cascade
- Whisper → the same backbone's text path → a head trained on transcripts (isolates *audio-native* vs *transcribe-then-decide*)
- the released head, zero-shot
- this model

Metrics: accuracy and macro-F1 per question, safety recall and false-alarm rate with confidence intervals, ECE / NLL / Brier with reliability diagrams, risk–coverage, accuracy against SNR, per-language results, and p50 / p95 latency. Differences come with paired-bootstrap confidence intervals.

## Run it on Kaggle (free T4s)

Two push-button notebooks in `notebooks/` (they are separate because the TTS library pins an older `transformers`):

1. **`01_kaggle_audio.ipynb`** — GPU T4, Internet on, secret `HF_TOKEN` (accept the Indic Parler-TTS terms on the Hub
   once). Scripts → TTS → noise and phone chain → `rv-audio/` (clips + manifest). ~1 h for the 130-script wedge.
2. **`02_kaggle_train.ipynb`** — GPU **T4 x2**, Internet on, with notebook 01's output attached. Backbone parity check →
   features (one forward per clip) → zero-shot baseline → train the head → calibrate → evaluate → latency.

Attach this repo folder as a Kaggle dataset (or set `REPO_URL`). Every stage is resumable: if a session stops, re-run.

## Run it locally

```bash
pip install -r requirements.txt
git clone https://github.com/NeverInAsh/nirnaya-inference && export PYTHONPATH=$PWD/nirnaya-inference:$PWD

python src/scripts.py build --raw data/scripts/m1_wedge.raw.jsonl data/scripts/m1_round2.raw.jsonl \
                           --out data/scripts/m1_v2.jsonl --freeze-splits data/scripts/m1_wedge.jsonl
python src/tts.py      --scripts data/scripts/m1_v2.jsonl --out data/audio/tts            # separate env with parler-tts
python src/fetch_noise.py --out data/noise
python src/augment.py  --tts-manifest data/audio/tts/tts_manifest.jsonl --scripts data/scripts/m1_v2.jsonl \
                       --noise-dir data/noise --out data/audio/clips
python src/parity.py   --model <nirnaya-flash-1.0 dir>                                       # must PASS first
python src/extract.py  --model <model dir> --manifest data/audio/clips/manifest.jsonl --out features
python train/train_head.py --features features --out runs/m1 --init <model dir>/head.pt --subset m1
python train/crossfit.py   --run runs/m1 --folds 5
python train/calibrate.py  --run runs/m1 --init-fit <model dir>/t_entropy.json
python train/predict.py    --run runs/m1 --features features --splits test --subset m1 --out runs/m1/pred_test.jsonl
python train/evaluate.py   --pred ours=runs/m1/pred_test.jsonl --out results/m1
python src/engine.py       --model <model dir> --run runs/m1 some_clip.wav --bench 50
```

The trained `head.pt` keeps the released `{"cfg", "state"}` format, so it also loads in the original nirnaya-inference engine.

## Repo layout

```
spec/decision_set.json      the decision set: state, four questions, action rules, safety rules, language rule
src/labels.py               gold action/safety from cues (+ self-test)
src/request.py              the request and the exact prompt text the backbone reads
src/scripts.py              slot grid, LLM drafting, lint + dedup, labels, script-level splits
src/tts.py                  Indic Parler-TTS renders (speaker-disjoint train vs val/test voices)
src/augment.py              helmet, road noise at SNR, wind, phone band, codec, AGC, packet loss (+ self-test)
src/fetch_noise.py          real background noise (MS-SNSD) sorted into the chain's categories, train/test kept apart
src/backbone.py             frozen backbone on 2x T4 / CPU: prompt, forward, spans, pooling
src/parity.py               P0 gate: released head on our features, 20 easy decisions
src/extract.py              one forward per clip -> cached features (resumable)
src/decide.py               features -> head logits -> calibrated probabilities -> answers
src/product.py              the one-tap card and the SOS rules (+ self-test)
src/engine.py               wav -> answers + card, latency benchmark
src/metrics.py              accuracy, macro-F1, ECE, NLL, Brier, Wilson CIs, AUROC, bootstrap (+ self-test)
src/real_manifest.py        real recordings (labels.csv) -> manifest
train/train_head.py         warm-start the joint head on cached features, early stopping
train/crossfit.py           out-of-fold logits over train+val scripts (more safety cases for the threshold)
train/calibrate.py          per-question temperature + safety threshold (recall + per-script floor), never on test
train/predict.py            features + run -> predictions.jsonl
train/evaluate.py           predictions -> metrics.json + REPORT.md (script-level CIs, per-class recall, per-script table)
notebooks/                  the two Kaggle notebooks and their builder
tests/                      tiny random Gemma 4 with the real tokenizer/processor; full pipeline test on CPU
data/  docs/                scripts, real-clip label format, backbone notes
```

Tests: `python src/labels.py && python src/metrics.py && python src/product.py && python src/augment.py --selftest`,
then `python tests/test_pipeline_tiny.py --nirnaya <nirnaya-inference dir>` (every stage end to end, ~2 min on CPU).

## Limitations (known up front)

- Synthetic training speech is cleaner and more uniform than real rider speech; the real test set is small, so every headline number comes with an interval.
- Hindi vs Hinglish is a fuzzy line in speech. A loanword rule makes the labels consistent, not the boundary real.
- Options are scored in a fixed order (features are cached once per clip).
- This is a research concept, not a safety system. Do not rely on it as the only check in an emergency.

## Licence and credit

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Built on nirnaya-flash-1.0 (a modified Gemma 4 E4B) and the nirnaya-inference code by Anshik Bansal; "Nirnaya" is his name and is used here only to say where this work comes from. Request format and evaluation style follow the Decision Index kit by Apolinário Passos; no Decision Index suite data is used.
