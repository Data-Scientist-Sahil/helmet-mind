# Backbone notes

How features are extracted. The head must be served with exactly the settings it was trained with; `src/extract.py`
writes them to `features/meta.json` and `src/engine.py` reads them back from the run.

## What is fixed (checked against the code)

- Model: `Darkrider/nirnaya-flash-1.0` — `Gemma4ForConditionalGeneration`, 7.94 B params, BF16, one 15.9 GB file.
  The vision tower is not loaded (`vision_config = None`): audio + text only.
- Released head: `head.pt`, cfg `hidden=2560, d=768, layers=4, heads=12`, 41.55 M params. Always build it from the
  checkpoint's `cfg` (the class default of 2816 does not match).
- Prompt: `mmclm.joint.prompt_text`, audio first, text written twice when <= 3,000 tokens; the head reads the spans of
  the second copy. The chat template writes `<bos>`; the processor is called with `add_special_tokens=False` and the
  code asserts exactly one `<bos>`.
- Audio: 16 kHz mono; the Gemma 4 processor inserts 25 audio tokens per second (40 ms each), so an 8 s clip is ~200
  tokens and a whole request is ~850-1,000 tokens.
- Hidden state: `last_hidden_state` of the base model = the output of the final RMSNorm, which is what vLLM's
  `token_embed` pooler returns in the reference engine. (`hidden_states[-1]` in transformers 5.17 is also post-norm;
  the pre-norm variant is captured with a hook if P0 needs it.)
- Multi-GPU: the KV-sharing layers move shared keys/values to their own device (`modeling_gemma4.py`), so splitting the
  decoder across two GPUs is safe.

## Precision on Kaggle T4 (no bf16)

`fp32` (default): weights are loaded as fp16 onto the GPUs and every module except the per-layer embedding table is
upcast to fp32 in place, buffers included. The per-layer table (2.82 B params) stays fp16 — it is a lookup and its
values are exact in fp16. Planned split: ~12.2 GB on GPU 0 (embeddings, per-layer table, audio tower, first layers) and
~11.5 GB on GPU 1.

`fp16`: ~16 GB, faster, but Gemma activations can overflow; `features()` refuses non-finite states.

## Parity run (Kaggle, 2× T4, 2026-11-13)

- Precision / device map: fp32 compute, per-layer table fp16, split over both T4s as planned.
- Hidden-state variant: post_norm (`last_hidden_state`).
- `src/parity.py`: **20/20 correct**, PASS; ~490 ms per text request at this precision.
- Zero-shot on 4 clean test clips (`cu_engl_005`, two voices): situation customer_unreachable, action call_customer,
  safety no (p ≈ 0.10–0.12), language english - sensible; 169–192 audio tokens per clip.
- End-to-end latency (audio clip → probabilities + card), fp32, batch 1: **p50 2,311 ms, p95 2,358 ms** on a T4.
  fp16 accuracy and latency are measured by notebook 02 from round 2 on.
