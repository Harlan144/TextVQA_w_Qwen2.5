"""Utility functions: answer normalization, config loading, seeding, logging."""

import os
import re
import json
import random
import logging
from pathlib import Path

import yaml
import torch
import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load a YAML config file and return as dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(*configs: dict) -> dict:
    """Shallow-merge multiple config dicts (later overrides earlier)."""
    merged = {}
    for cfg in configs:
        for key, val in cfg.items():
            if isinstance(val, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **val}
            else:
                merged[key] = val
    return merged


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Answer normalization (official TextVQA / VQA style)
# ---------------------------------------------------------------------------

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "id": "i'd", "ill": "i'll", "im": "i'm", "ive": "i've", "isnt": "isn't",
    "itd": "it'd", "itll": "it'll", "its": "it's", "mightve": "might've",
    "mightnt": "mightn't", "mustve": "must've", "mustnt": "mustn't",
    "neednt": "needn't", "oughtnt": "oughtn't", "shant": "shan't",
    "shed": "she'd", "shell": "she'll", "shes": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "thats": "that's", "thered": "there'd",
    "therell": "there'll", "theres": "there's", "theyd": "they'd",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "wasnt": "wasn't", "wed": "we'd", "well": "we'll", "were": "we're",
    "weve": "we've", "werent": "weren't", "whatll": "what'll", "whats": "what's",
    "whens": "when's", "whered": "where'd", "wheres": "where's", "wholl": "who'll",
    "whos": "who's", "whyd": "why'd", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't", "youd": "you'd",
    "youll": "you'll", "youre": "you're", "youve": "you've",
}

_NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
# All punctuation except apostrophes (needed for contractions)
_PUNCT_CHARS = set(".,;:!?@#$%^&*()[]{}|/\\\"<>~_-+=`")


def normalize_answer(answer: str) -> str:
    """Normalize an answer string following the official VQA evaluation protocol."""
    answer = answer.strip().lower()

    # Expand contractions
    words = answer.split()
    words = [_CONTRACTIONS.get(w, w) for w in words]
    answer = " ".join(words)

    # Handle periods not in decimal numbers
    answer = _PERIOD_STRIP.sub("", answer)
    # Handle commas in numbers (e.g., "1,000" -> "1000")
    answer = _COMMA_STRIP.sub(r"\1\3", answer)

    # Remove remaining punctuation (keep apostrophes for contractions)
    answer = "".join(c if c not in _PUNCT_CHARS else " " for c in answer)

    # Remove articles
    answer = _ARTICLES.sub(" ", answer)

    # Number words to digits
    words = answer.split()
    words = [_NUMBER_MAP.get(w, w) for w in words]
    answer = " ".join(words)

    # Collapse whitespace
    answer = " ".join(answer.split())
    return answer


def vqa_accuracy_score(prediction: str, ground_truths: list[str]) -> float:
    """Compute accuracy: 1.0 if the prediction matches any ground truth answer."""
    norm_pred = normalize_answer(prediction)
    return 1.0 if any(normalize_answer(gt) == norm_pred for gt in ground_truths) else 0.0


def vqa_accuracy_score_3(prediction: str, ground_truths: list[str]) -> float:
    """Compute official VQA accuracy for a single prediction.

    Uses official formula: min(1.0, num_matches / 3) where num_matches is
    the count of ground truth answers that match the prediction after normalization.
    """
    norm_pred = normalize_answer(prediction)
    num_matches = sum(
        1 for gt in ground_truths if normalize_answer(gt) == norm_pred
    )
    return min(1.0, num_matches / 3)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
