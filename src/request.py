"""Build the one request every clip is scored with: a state plus four typed questions, in the
Decision Index format (`choice` with `criteria`, `noul` for yes/no).

The prompt text the backbone reads is rendered by `mmclm.joint.prompt_text` from nirnaya-inference when it is
importable. `render_prompt_text` below reproduces the same layout so the repo can be tested without it.

    python src/request.py         # prints the request and the rendered prompt text
"""

from __future__ import annotations

import json

try:
    from src.labels import spec
except ImportError:  # run as a script from the repo root
    from labels import spec

INSTRUCTION = "Answer each question with the number of the correct option only."


def build_request() -> tuple[str, dict]:
    sp = spec()
    return sp["state"], sp["questions"]


def _shown(questions: dict) -> list[tuple[dict, list[int]]]:
    """Questions in the joint-head shape: options in spec order (no shuffling — features are cached)."""
    shown = []
    for q in questions.values():
        opts = ["yes", "no"] if q["type"] == "noul" else list(q["criteria"].values())
        shown.append(({"instruction": q["instructions"], "type": q["type"], "options": opts}, list(range(len(opts)))))
    return shown


def render_prompt_text(state: str, questions: dict) -> str:
    try:
        from mmclm import joint

        return joint.prompt_text(state, _shown(questions))
    except ImportError:
        blocks = []
        for i, (q, perm) in enumerate(_shown(questions)):
            lines = "\n".join(f"{j + 1}. {q['options'][p]}" for j, p in enumerate(perm))
            blocks.append(f"[Q{i + 1}] {q['instruction']}".rstrip() + "\n" + lines)
        return f"{state.strip()}\n\nQuestions:\n" + "\n".join(blocks) + f"\n\n{INSTRUCTION}"


if __name__ == "__main__":
    state, questions = build_request()
    print(json.dumps({"state": state, "questions": questions}, indent=1, ensure_ascii=False)[:600], "...\n")
    print(render_prompt_text(state, questions))
