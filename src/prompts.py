"""Prompt strategies for TextVQA with Qwen2.5-VL.

Each strategy is a callable: (image, question, ocr_tokens=None) -> messages list
in Qwen chat format.
"""

from typing import Optional
from PIL import Image


def _image_content(image: Image.Image) -> dict:
    """Build image content block for Qwen message format."""
    return {"type": "image", "image": image}


# ---------------------------------------------------------------------------
# Strategy 1: Baseline — bare question
# ---------------------------------------------------------------------------

def baseline(image: Image.Image, question: str, ocr_tokens: Optional[list[str]] = None) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                _image_content(image),
                {"type": "text", "text": (
                    f"Answer the question about this image concisely.\n"
                    f"Question: {question}\n"
                    f"Answer:"
                )},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Strategy 2: OCR-augmented — provide detected text tokens
# ---------------------------------------------------------------------------

def ocr_augmented(image: Image.Image, question: str, ocr_tokens: Optional[list[str]] = None) -> list[dict]:
    ocr_text = ", ".join(ocr_tokens) if ocr_tokens else "none detected"
    return [
        {
            "role": "user",
            "content": [
                _image_content(image),
                {"type": "text", "text": (
                    f"The following text was detected in the image: [{ocr_text}]\n"
                    f"Question: {question}\n"
                    f"Using the image and the detected text, provide a concise answer.\n"
                    f"Answer:"
                )},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Strategy 3: Chain-of-thought — step-by-step reasoning
# ---------------------------------------------------------------------------

def chain_of_thought(image: Image.Image, question: str, ocr_tokens: Optional[list[str]] = None) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                _image_content(image),
                {"type": "text", "text": (
                    f"Look at this image carefully.\n"
                    f"Step 1: Identify all text visible in the image.\n"
                    f"Step 2: Reason about how the visible text relates to the question.\n"
                    f"Step 3: Provide the answer.\n\n"
                    f"Question: {question}\n\n"
                    f"Think step by step, then give your final answer on the last line after 'Answer:'."
                )},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Strategy 4: Instructed-concise — system prompt + strict format
# ---------------------------------------------------------------------------

def instructed_concise(image: Image.Image, question: str, ocr_tokens: Optional[list[str]] = None) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a visual question answering assistant specialized in reading "
                "text from images. Always give short, precise answers — just the exact "
                "text or value requested, nothing else. Never explain your reasoning."
            ),
        },
        {
            "role": "user",
            "content": [
                _image_content(image),
                {"type": "text", "text": (
                    f"Question: {question}\n"
                    f"Answer with only the exact text or value requested:"
                )},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Strategy 5: OCR + Chain-of-thought combined
# ---------------------------------------------------------------------------

def ocr_cot(image: Image.Image, question: str, ocr_tokens: Optional[list[str]] = None) -> list[dict]:
    ocr_text = ", ".join(ocr_tokens) if ocr_tokens else "none detected"
    return [
        {
            "role": "system",
            "content": (
                "You are an expert at reading and understanding text in images. "
                "You are given an image, detected OCR text tokens from the image, "
                "and a question. Reason step by step, then provide a concise final answer."
            ),
        },
        {
            "role": "user",
            "content": [
                _image_content(image),
                {"type": "text", "text": (
                    f"Detected text in the image: [{ocr_text}]\n\n"
                    f"Question: {question}\n\n"
                    f"Think step by step about what text in the image is relevant, "
                    f"then give your final answer on the last line after 'Answer:'."
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
    "instructed_concise": instructed_concise,
    "ocr_cot": ocr_cot,
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
