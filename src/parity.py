"""P0 gate: does our transformers forward give the RELEASED head the features it expects?

    python src/parity.py --model <model dir> [--precision fp32] [--variant post_norm|pre_norm] [--clips a.wav b.wav ...]

Runs 20 easy text decisions (obvious answers, mixed English / Hindi / Hinglish) through backbone + released head +
released temperature. If the features match what the head was trained on, it gets ~all of them right with high
confidence; a wrong hidden-state choice or a broken prompt layout shows up as near-chance accuracy. Pass = >= 18/20.
Optional --clips: zero-shot answers for a few clean audio clips with the rider request (printed, not gated).
Writes parity.json next to the features if --out is given.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.backbone import Backbone, entropy_temperature, feature_items_for_head  # noqa: E402

YN = {"type": "noul"}


def ch(instr, opts):
    return {"type": "choice", "instructions": instr, "criteria": {f"o{i}": o for i, o in enumerate(opts)}}


def noul(instr):
    return {"type": "noul", "instructions": instr}


EASY = [  # (state, question, gold key)  - gold "true"/"false" for yes/no
    ("The traffic light is red.", ch("Should a car go or stop?", ["Go", "Stop"]), "o1"),
    ("It is raining heavily outside.", noul("Is it dry outside?"), "false"),
    ("Mera phone kho gaya hai.", ch("What happened to the speaker?", ["They lost their phone", "They found money", "They ate dinner"]), "o0"),
    ("I am very hungry and have not eaten all day.", ch("What does the person want most?", ["Food", "Sleep", "Music"]), "o0"),
    ("Customer ka gate band hai, phone nahi utha raha.", noul("Is the customer answering the phone?"), "false"),
    ("The package is crushed and the food has spilled everywhere.", noul("Is the package damaged?"), "true"),
    ("What is two plus two?", ch("Pick the answer.", ["3", "4", "5"]), "o1"),
    ("A man is shouting threats and waving a knife at the delivery rider.", noul("Is someone in danger?"), "true"),
    ("The meeting is at 5 pm on Monday.", ch("When is the meeting?", ["Monday at 5 pm", "Tuesday at 9 am", "Friday at noon"]), "o0"),
    ("बहुत तेज़ बारिश हो रही है।", ch("What is the weather like?", ["Rainy", "Sunny", "Snowing"]), "o0"),
    ("My bike has a flat tyre and I can't ride it.", ch("What is the problem?", ["A puncture", "No fuel", "A lost key"]), "o0"),
    ("The restaurant says the order will take thirty more minutes.", noul("Is the order ready now?"), "false"),
    ("She said thank you, laughed and hugged her friend.", ch("How does she feel?", ["Happy", "Angry", "Scared"]), "o0"),
    ("Water boils at 100 degrees Celsius at sea level.", noul("Is this statement true?"), "true"),
    ("New Delhi is the capital of India.", ch("What is the capital of India?", ["New Delhi", "Mumbai", "Chennai"]), "o0"),
    ("The customer refused to pay and walked away without the order.", noul("Did the customer pay?"), "false"),
    ("Please call me back in ten minutes, I'm driving.", ch("What is being asked?", ["Call back later", "Send money", "Cancel the order"]), "o0"),
    ("Main bike se gir gaya, pair se khoon aa raha hai.", noul("Is the speaker injured?"), "true"),
    ("The road ahead is closed because of flooding.", noul("Can vehicles use this road right now?"), "false"),
    ("Bonjour, comment allez-vous aujourd'hui ?", ch("Which language is this?", ["French", "Hindi", "Japanese"]), "o0"),
]


@torch.no_grad()
def score(bb, head, jh, fit, state, questions, audio=None, device="cpu"):
    f = bb.features(state, questions, audio)
    lg = head(jh.collate(feature_items_for_head(f), device)).float().cpu()
    out = {}
    for i, it in enumerate(f["items"]):
        x = lg[i, : len(it["keys"])]
        p = (x / entropy_temperature(x, fit)).softmax(-1).tolist()
        out[it["qid"]] = dict(zip(it["keys"], p))
    return out, f["meta"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--variant", default="post_norm", choices=["post_norm", "pre_norm"])
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from mmclm import jh

    from src.request import build_request

    bb = Backbone(a.model, precision=a.precision, device=a.device, state_variant=a.variant)
    hd = "cuda" if torch.cuda.is_available() and a.device != "cpu" else "cpu"
    head = jh.load_head(str(Path(a.model) / "head.pt"), device=hd)
    if head.cfg["hidden"] != bb.hidden:
        raise SystemExit(f"head hidden {head.cfg['hidden']} != backbone hidden {bb.hidden}")
    tf = Path(a.model) / "t_entropy.json"
    fit = json.loads(tf.read_text()) if tf.exists() else None
    ok, rows, t0 = 0, [], time.time()
    for state, q, gold in EASY:
        probs, meta = score(bb, head, jh, fit, state, {"q": q}, None, hd)
        p = probs["q"]
        pick = max(p, key=p.get)
        ok += pick == gold
        rows.append({"state": state, "pick": pick, "gold": gold, "p_gold": round(p[gold], 3), "ok": pick == gold,
                     "h_absmax": round(meta["h_absmax"], 1)})
        print(f"[parity] {'OK ' if pick == gold else 'BAD'} p(gold)={p[gold]:.3f}  {state[:60]}")
    ms = (time.time() - t0) * 1000 / len(EASY)
    passed = ok >= 18
    print(f"[parity] {ok}/{len(EASY)} correct, mean p(gold)={sum(r['p_gold'] for r in rows) / len(rows):.3f}, "
          f"{ms:.0f} ms/request, precision={bb.precision}, variant={a.variant} -> {'PASS' if passed else 'FAIL'}")
    audio_rows = []
    if a.clips:
        state, questions = build_request()
        for c in a.clips:
            probs, meta = score(bb, head, jh, fit, state, questions, c, hd)
            top = {q: max(p, key=p.get) for q, p in probs.items()}
            audio_rows.append({"clip": str(c), "top": top, "p_safety_yes": round(probs["safety"]["true"], 3),
                               "n_audio_tokens": meta["n_audio_tokens"]})
            print(f"[parity] zero-shot {Path(c).name}: {top}  p(safety)={probs['safety']['true']:.2f}")
    if a.out:
        Path(a.out).write_text(json.dumps({"passed": passed, "correct": ok, "n": len(EASY), "precision": bb.precision,
                                           "variant": a.variant, "ms_per_request": ms, "rows": rows, "audio": audio_rows}, indent=1))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
