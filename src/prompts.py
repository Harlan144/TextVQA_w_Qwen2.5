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
# Registry
# ---------------------------------------------------------------------------

STRATEGIES = {
    "baseline": baseline,
    "ocr_augmented": ocr_augmented,
    "chain_of_thought": chain_of_thought,
    "no_system_prompt": no_system_prompt,
    "ocr_cot": ocr_cot,
    "ocr_only": ocr_only,
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
        line_stripped = line.strip()
        if line_stripped.lower().startswith("answer:"):
            return line_stripped[len("answer:"):].strip()
    return text.strip()
