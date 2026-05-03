"""Prompt strategies for TextVQA with Qwen2.5-VL.

Each strategy is a callable: (image, question, ocr_tokens=None) -> messages list
in Qwen chat format.

All strategies share a common structure: a system prompt enforcing concise
answers, an image, a question, and an instruction line. Each strategy
ablates exactly one variable relative to baseline:
  - baseline:           system prompt + image + question + concise instruction
  - ocr_augmented:      + OCR tokens
  - chain_of_thought:   + CoT instruction (replaces concise instruction)
  - no_system_prompt:   - system prompt
  - ocr_cot:            + OCR tokens + CoT instruction
  - ocr_only:           + OCR tokens - image
"""

from typing import Optional
from PIL import Image


def _image_content(image: Image.Image) -> dict:
    """Build image content block for Qwen message format."""
    return {"type": "image", "image": image}


_SYSTEM_PROMPT = (
    "You are a visual question answering assistant specialized in reading "
    "text from images. Always give short, precise answers — just the exact "
    "text or value requested, nothing else. Never explain your reasoning."
)

_CONCISE_INSTRUCTION = (
    "Answer the question about this image concisely.\n"
    "Answer:"
)

_COT_INSTRUCTION = (
    "Think step by step about what text in the image is relevant, "
    "then give your final answer on the last line after 'Answer:'."
)


# ---------------------------------------------------------------------------
# Strategy 1: Baseline — bare question
# ---------------------------------------------------------------------------

def baseline(image: Image.Image, question: str, ocr_tokens: Optional[list[str]] = None) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                _image_content(image),
                {"type": "text", "text": (
                    f"Question: {question}\n"
                    f"{_CONCISE_INSTRUCTION}"
                )},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Strategy 2: OCR-augmented — provide detected text tokens
# ---------------------------------------------------------------------------

def ocr_augmented(image: Image.Image, question: str, ocr_tokens: Optional[list[str]] = None) -> list[dict]:
    ocr_text = ", ".join(ocr_tokens) if ocr_tokens else "none detected"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                _image_content(image),
                {"type": "text", "text": (
                    f"The following text was detected in the image: [{ocr_text}]\n"
                    f"Question: {question}\n"
                    f"{_CONCISE_INSTRUCTION}"
                )},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Strategy 3: Chain-of-thought — step-by-step reasoning
# ---------------------------------------------------------------------------

def chain_of_thought(image: Image.Image, question: str, ocr_tokens: Optional[list[str]] = None) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                _image_content(image),
                {"type": "text", "text": (
                    f"Question: {question}\n"
                    f"{_COT_INSTRUCTION}"
                )},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Ablation: No system prompt — baseline without the system prompt
# ---------------------------------------------------------------------------

def no_system_prompt(image: Image.Image, question: str, ocr_tokens: Optional[list[str]] = None) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                _image_content(image),
                {"type": "text", "text": (
                    f"Question: {question}\n"
                    f"{_CONCISE_INSTRUCTION}"
                )},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Strategy 5: OCR + Chain-of-thought combined
# ---------------------------------------------------------------------------

def ocr_cot(image: Image.Image, question: str, ocr_tokens: Optional[list[str]] = None) -> list[dict]:
    ocr_text = ", ".join(ocr_tokens) if ocr_tokens else "none detected"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                _image_content(image),
                {"type": "text", "text": (
                    f"The following text was detected in the image: [{ocr_text}]\n"
                    f"Question: {question}\n"
                    f"{_COT_INSTRUCTION}"
                )},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Ablation: OCR-only — text tokens without the image
# ---------------------------------------------------------------------------

def ocr_only(image: Image.Image, question: str, ocr_tokens: Optional[list[str]] = None) -> list[dict]:
    ocr_text = ", ".join(ocr_tokens) if ocr_tokens else "none detected"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    f"The following text was detected in an image: [{ocr_text}]\n"
                    f"Question: {question}\n"
                    f"Answer the question concisely.\n"
                    f"Answer:"
                )},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Strategy 7: Few-shot — 2 demonstration examples before the question
# ---------------------------------------------------------------------------

# Stored few-shot examples (populated at runtime from the training set)
_FEW_SHOT_EXAMPLES: list[dict] = []


def set_few_shot_examples(examples: list[dict]):
    """Set the few-shot demonstration examples.

    Each example should have keys: image (PIL), question (str), answer (str).
    Called once at startup from the runner script.
    """
    global _FEW_SHOT_EXAMPLES
    _FEW_SHOT_EXAMPLES = examples


def few_shot(image: Image.Image, question: str, ocr_tokens: Optional[list[str]] = None) -> list[dict]:
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for ex in _FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": [
                _image_content(ex["image"]),
                {"type": "text", "text": (
                    f"Question: {ex['question']}\n"
                    f"{_CONCISE_INSTRUCTION}"
                )},
            ],
        })
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({
        "role": "user",
        "content": [
            _image_content(image),
            {"type": "text", "text": (
                f"Question: {question}\n"
                f"{_CONCISE_INSTRUCTION}"
            )},
        ],
    })
    return messages


# ---------------------------------------------------------------------------
# Strategy 8: Answer-format constraint — explicit short-answer instruction
# ---------------------------------------------------------------------------

_FORMAT_CONSTRAINT_INSTRUCTION = (
    "Answer in 1 to 3 words only. No explanation, no sentences.\n"
    "Answer:"
)


def format_constraint(image: Image.Image, question: str, ocr_tokens: Optional[list[str]] = None) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                _image_content(image),
                {"type": "text", "text": (
                    f"Question: {question}\n"
                    f"{_FORMAT_CONSTRAINT_INSTRUCTION}"
                )},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Strategy 9: Think hard — encourage careful reasoning with concise output
# ---------------------------------------------------------------------------

_THINK_HARD_INSTRUCTION = (
    "Look very carefully at ALL text visible in the image. "
    "Think hard about which text answers the question. "
    "Give only the answer, nothing else.\n"
    "Answer:"
)


def think_hard(image: Image.Image, question: str, ocr_tokens: Optional[list[str]] = None) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                _image_content(image),
                {"type": "text", "text": (
                    f"Question: {question}\n"
                    f"{_THINK_HARD_INSTRUCTION}"
                )},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGIES = {
    "baseline": baseline,
    "ocr_augmented": ocr_augmented,
    "chain_of_thought": chain_of_thought,
    "no_system_prompt": no_system_prompt,
    "ocr_cot": ocr_cot,
    "ocr_only": ocr_only,
    "few_shot": few_shot,
    "format_constraint": format_constraint,
    "think_hard": think_hard,
}


def get_strategy(name: str):
    """Get a prompt strategy function by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]


def list_strategies() -> list[str]:
    return list(STRATEGIES.keys())


def extract_final_answer(text: str) -> str:
    """Extract the final answer from a CoT-style response.

    Looks for the last line starting with 'Answer:' and returns everything after it.
    Falls back to the full text if no such marker is found.
    """
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line_lower = line.strip().lower()
        idx = line_lower.rfind("answer:")
        if idx != -1:
            return line.strip()[idx + len("answer:"):].strip()
    return text.strip()
