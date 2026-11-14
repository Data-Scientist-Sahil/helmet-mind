"""The frozen backbone: one forward pass per clip -> pooled features for every question.

Mirrors `NirnayaAudioEngine._prepare/_finish` from nirnaya-inference (mmclm/di_engine.py) without vLLM, so it runs on
Kaggle T4s (no bf16) and on CPU:

  audio first, then the joint prompt text (mmclm.joint) written twice when short enough; the head reads the spans of
  the second copy (mmclm.jh.joint_spans) from the final, post-norm hidden states; mmclm.jh.pool_joint pools them.

Precision modes
  fp32  (default on T4): weights loaded as fp16 onto the GPUs, then every module except the per-layer embedding
        table is upcast to fp32 in place. The per-layer table (~2.8 B params) is a lookup and stays fp16
        (exact for the trained bf16 values in fp16's normal range). ~26 GB across 2x T4.
  fp16  everything fp16 (~16 GB). Faster; Gemma activations can overflow in fp16 - `features()` checks for inf/nan.
  bf16  for GPUs that support it (A100/L4/RTX Pro 6000) - the closest to how the model was trained.
  On CPU, fp32 only.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np
import torch

THOUGHT = "<|channel>thought\n<channel|>"  # the empty thought add_generation_prompt appends with thinking off
SAMPLE_RATE = 16000
PLE = "model.language_model.embed_tokens_per_layer"


def _mmclm():
    try:
        from mmclm import jh, joint
    except ImportError as e:  # pragma: no cover
        raise ImportError("nirnaya-inference is not importable. Clone github.com/NeverInAsh/nirnaya-inference "
                          "(pinned commit in requirements.txt) and add it to PYTHONPATH.") from e
    return jh, joint


def load_audio(a, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Path, bytes or array -> float32 mono at 16 kHz (same as the reference engine: librosa, mono)."""
    if isinstance(a, np.ndarray):
        return a.astype(np.float32)
    import librosa

    src = io.BytesIO(a) if isinstance(a, (bytes, bytearray)) else a
    return librosa.load(src, sr=sr, mono=True)[0].astype(np.float32)


# --------------------------------------------------------------------------- device planning
def _module_sizes(config) -> dict[str, int]:
    """Parameter count of every placement unit, from a meta-device model (no weights loaded)."""
    from transformers import Gemma4ForConditionalGeneration

    with torch.device("meta"):
        m = Gemma4ForConditionalGeneration(config)
    sizes: dict[str, int] = {}
    for name, p in m.named_parameters():
        parts = name.split(".")
        unit = ".".join(parts[:4]) if name.startswith("model.language_model.layers.") else ".".join(parts[:3])
        if name.startswith("lm_head"):
            unit = "lm_head"
        sizes[unit] = sizes.get(unit, 0) + p.numel()
    return sizes


def plan_device_map(config, n_gpus: int, gpu_bytes: list[int], precision: str, reserve_gb: float = 1.6) -> dict:
    """Embeddings, the per-layer table and the audio tower on cuda:0; decoder layers filled in order; final norm on the
    last GPU used. Sizes are counted at the *final* precision (fp32 except the per-layer table in fp32 mode)."""
    sizes = _module_sizes(config)
    bpp = {"fp32": 4, "fp16": 2, "bf16": 2}[precision]

    def nbytes(unit):
        return sizes[unit] * (2 if unit == PLE else bpp)

    budget = [b - int(reserve_gb * 2**30) for b in gpu_bytes[:n_gpus]]
    dmap, used, dev = {}, [0] * n_gpus, 0
    first = [u for u in sizes if not u.startswith("model.language_model.layers.") and u not in ("model.language_model.norm",)]
    for u in first:
        dmap[u], used[0] = 0, used[0] + nbytes(u)
    layers = sorted((u for u in sizes if u.startswith("model.language_model.layers.")), key=lambda u: int(u.split(".")[-1]))
    for u in layers:
        while dev < n_gpus - 1 and used[dev] + nbytes(u) > budget[dev]:
            dev += 1
        dmap[u], used[dev] = dev, used[dev] + nbytes(u)
    dmap["model.language_model.norm"] = dev
    dmap["model.language_model.rotary_emb"] = 0
    dmap["lm_head"] = 0  # tied to embed_tokens; never called
    if any(used[i] > budget[i] for i in range(n_gpus)):
        gb = [round(x / 2**30, 1) for x in used]
        raise RuntimeError(f"model does not fit: planned {gb} GB vs budgets {[round(b / 2**30, 1) for b in budget]}; "
                           f"try precision='fp16' or more GPUs")
    return dmap


# --------------------------------------------------------------------------- backbone
@dataclass
class Prepared:
    inputs: dict
    qs: list            # [(qid, kind, keys, jq)]
    shown: list         # [(jq, perm)]
    twice: bool
    n_tokens: int = 0
    extra: dict = field(default_factory=dict)


class Backbone:
    def __init__(self, model_dir: str, precision: str | None = None, device: str | None = None, twice_max: int = 3000,
                 mem_cap: int = 256, max_audio_s: float = 30.0, attn: str = "sdpa", verbose: bool = True,
                 state_variant: str = "post_norm"):
        from transformers import AutoConfig, AutoProcessor, Gemma4ForConditionalGeneration

        self.jh, self.joint = _mmclm()
        self.twice_max, self.mem_cap, self.max_audio_s = int(twice_max), int(mem_cap), float(max_audio_s)
        if state_variant not in ("post_norm", "pre_norm"):
            raise ValueError(state_variant)
        self.state_variant = state_variant  # post_norm = what vLLM's token_embed pooler returns (the reference)
        cuda = torch.cuda.is_available() and device != "cpu"
        native_bf16 = cuda and torch.cuda.get_device_capability(0)[0] >= 8  # T4 (7.5) only emulates bf16
        self.precision = precision or ("bf16" if native_bf16 else "fp32")
        if not cuda and self.precision != "fp32":
            raise ValueError("on CPU use precision='fp32'")

        self.proc = AutoProcessor.from_pretrained(model_dir)
        self.tok = self.proc.tokenizer
        cfg = AutoConfig.from_pretrained(model_dir)
        cfg.vision_config = None  # audio + text only: skip loading the vision tower
        load_dtype = {"fp32": torch.float16 if cuda else torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[self.precision]
        kw = dict(config=cfg, dtype=load_dtype, attn_implementation=attn)
        if cuda:
            n = torch.cuda.device_count()
            mem = [torch.cuda.get_device_properties(i).total_memory for i in range(n)]
            kw["device_map"] = plan_device_map(cfg, n, mem, self.precision)
        model = Gemma4ForConditionalGeneration.from_pretrained(model_dir, **kw)
        if self.precision == "fp32" and cuda:
            for name, p in model.named_parameters():  # upcast in place, module by module, leaving the PLE table fp16
                if not name.startswith(PLE) and p.dtype != torch.float32:
                    p.data = p.data.float()
            for mod in model.modules():  # rotary / positional buffers too
                for bn, b in list(mod._buffers.items()):
                    if b is not None and b.is_floating_point() and b.dtype != torch.float32:
                        mod._buffers[bn] = b.float()
            torch.cuda.empty_cache()
        self.model = model.eval()
        self.hidden = cfg.text_config.hidden_size
        self.in_device = next(model.model.language_model.embed_tokens.parameters()).device
        self.feat_dtype = torch.float32 if self.precision == "fp32" else load_dtype
        if verbose:
            devs = sorted({str(p.device) for p in model.parameters()})
            print(f"[backbone] precision={self.precision} devices={devs} hidden={self.hidden} "
                  f"mem_cap={self.mem_cap} twice_max={self.twice_max}")

    # ----------------------------------------------------------------- prompt (mirrors NirnayaAudioEngine._prepare)
    def prepare(self, state, questions: dict, audio=None) -> Prepared:
        joint = self.joint
        qs = []
        for qk, q in questions.items():
            kind = q.get("type", "choice")
            if kind == "noul":
                keys, opts = ["true", "false"], ["yes", "no"]
            else:
                keys = list(q.get("criteria") or {})
                if not 2 <= len(keys) <= joint.MAX_OPTIONS:
                    raise ValueError(f"{qk}: {len(keys)} options")
                opts = [str(q["criteria"][k]).strip() or k for k in keys]
            qs.append((qk, kind, keys, {"instruction": str(q.get("instructions", "")).strip(), "type": kind,
                                         "options": opts, "gold": 0}))
        shown = [(jq, list(range(len(jq["options"])))) for *_, jq in qs]
        head_text = "" if state in ("", None, {}, []) else str(state).strip()
        text = joint.prompt_text(head_text, shown)
        twice = bool(self.twice_max) and len(self.tok(text, add_special_tokens=False)["input_ids"]) <= self.twice_max
        wav = None if audio is None else load_audio(audio)
        if wav is not None and len(wav) > self.max_audio_s * SAMPLE_RATE:
            wav = wav[: int(self.max_audio_s * SAMPLE_RATE)]
        content = ([{"type": "audio"}] if wav is not None else []) + [{"type": "text", "text": f"{text}\n\n{text}" if twice else text}]
        pr = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        if twice and pr.endswith(THOUGHT):
            pr = pr[: -len(THOUGHT)]
        # the chat template already writes <bos>; don't let the tokenizer add a second one
        if wav is not None:
            inputs = self.proc(text=pr, audio=[wav], return_tensors="pt", add_special_tokens=False)
        else:
            inputs = self.proc(text=pr, return_tensors="pt", add_special_tokens=False)
        ids = inputs["input_ids"][0]
        if int((ids == self.tok.bos_token_id).sum()) != 1:
            raise RuntimeError(f"expected exactly one <bos>, got {int((ids == self.tok.bos_token_id).sum())}")
        return Prepared(inputs=dict(inputs), qs=qs, shown=shown, twice=twice, n_tokens=len(ids),
                        extra={"audio_s": 0.0 if wav is None else len(wav) / SAMPLE_RATE})

    def spans(self, ids: list[int], p: Prepared) -> dict:
        jh, joint = self.jh, self.joint
        off = 0
        if p.twice:  # the second copy starts after the first copy's closing instruction
            instr = self.tok(joint.INSTRUCTION, add_special_tokens=False)["input_ids"]
            j = jh._find(ids, instr, 0)
            if j < 0:
                raise RuntimeError("closing instruction of the first copy not found")
            off = j + len(instr)
        sp = jh.joint_spans(self.tok, ids[off:], p.shown)
        if sp is None:
            raise RuntimeError("joint layout spans not found")
        if off:
            sh = lambda ab: (ab[0] + off, ab[1] + off)  # noqa: E731
            sp = {"state": sh(sp["state"]), "questions": [sh(x) for x in sp["questions"]],
                  "options": [[sh(x) for x in o] for o in sp["options"]]}
        return sp

    # ----------------------------------------------------------------- forward
    @torch.no_grad()
    def hidden_states(self, p: Prepared) -> torch.Tensor:
        """[T, H] final (post-norm) hidden states of the prompt, the same states vLLM's token_embed pooler returns."""
        x = {k: v.to(self.in_device) for k, v in p.inputs.items() if torch.is_tensor(v)}
        if "input_features" in x:
            x["input_features"] = x["input_features"].to(self.feat_dtype)
        if self.state_variant == "post_norm":
            return self.model.model(**x, use_cache=False, return_dict=True).last_hidden_state[0]
        # pre_norm: the input of the final RMSNorm (transformers replaces the last hidden_states entry with the
        # normed one, so it is captured with a hook instead)
        grab = {}
        hook = self.model.model.language_model.norm.register_forward_pre_hook(lambda m, args: grab.__setitem__("h", args[0]))
        try:
            self.model.model(**x, use_cache=False, return_dict=True)
        finally:
            hook.remove()
        return grab["h"][0]

    @torch.no_grad()
    def features(self, state, questions: dict, audio=None) -> dict:
        """One clip -> {"memory": [C,H] fp16, "items": [per question {end, question, options}], "meta": {...}}.
        `memory` is shared by every question of the clip (pool_joint gives each question the same token memory)."""
        p = self.prepare(state, questions, audio)
        h = self.hidden_states(p).float()
        if not torch.isfinite(h).all():
            raise FloatingPointError(f"non-finite hidden states (precision={self.precision})")
        ids = p.inputs["input_ids"][0].tolist()
        sp = self.spans(ids, p)
        self.jh.MEM_CAP = self.mem_cap if self.mem_cap > 0 else 2048
        pooled = self.jh.pool_joint(h, sp, len(h) - 1)
        mem = pooled[0]["chunks"]
        items = [{"qid": qk, "kind": kind, "keys": keys, "end": f["end"].half().cpu(), "question": f["question"].half().cpu(),
                  "options": f["options"].half().cpu()} for (qk, kind, keys, _), f in zip(p.qs, pooled)]
        audio_id = getattr(self.tok, "audio_token_id", None)
        meta = {"n_tokens": len(ids), "n_audio_tokens": int(sum(t == audio_id for t in ids)) if audio_id is not None else None,
                "twice": p.twice, "mem_cap": self.mem_cap, "precision": self.precision, "audio_s": p.extra["audio_s"],
                "state_variant": self.state_variant,
                "h_absmax": float(h.abs().max())}
        return {"memory": mem.half().cpu(), "items": items, "meta": meta}


def feature_items_for_head(feat: dict) -> list[dict]:
    """Per-question dicts in the shape mmclm.jh.collate expects (shared memory, zero digit prior)."""
    out = []
    for it in feat["items"]:
        k = it["options"].shape[0]
        out.append({"end": it["end"], "question": it["question"], "options": it["options"], "chunks": feat["memory"],
                    "prior": torch.zeros(k), "exact_prior": False})
    return out


def entropy_temperature(lg: torch.Tensor, fit: dict | None, t_global: float = 1.0) -> float:
    """T = exp(a + b*H + c*H^2 + d*log k), H the normalised entropy of softmax(lg) (as in di_engine._entropy_t)."""
    if not fit:
        return t_global
    a, b, c, d = (float(fit.get(x, 0.0)) for x in "abcd")
    k = lg.numel()
    lp = lg.float().log_softmax(-1)
    h = float(-(lp.exp() * lp).sum()) / math.log(k) if k > 1 else 0.0
    return math.exp(a + b * h + c * h * h + d * math.log(k))
