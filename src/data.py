"""TextVQA dataset loading and preprocessing."""

import logging
from typing import Optional

from datasets import load_dataset
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TextVQADataset(Dataset):
    """Lightweight wrapper around the HuggingFace TextVQA dataset."""

    def __init__(
        self,
        split: str = "validation",
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        logger.info(f"Loading TextVQA split={split} ...")
        self.hf_dataset = load_dataset(
            "lmms-lab/textvqa", split=split, cache_dir=cache_dir
        )
        if max_samples is not None and max_samples < len(self.hf_dataset):
            self.hf_dataset = self.hf_dataset.select(range(max_samples))
            logger.info(f"Subsetted to {max_samples} samples")
        logger.info(f"Loaded {len(self.hf_dataset)} samples")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict:
        row = self.hf_dataset[idx]
        return {
            "image": row["image"],                  # PIL Image
            "question": row["question"],
            "answers": row.get("answers", []),       # list of 10 answer strings
            "ocr_tokens": row.get("ocr_tokens", []),
            "question_id": row.get("question_id"),
            "image_id": row.get("image_id"),
        }


def get_dataset(
    split: str,
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> TextVQADataset:
    """Convenience factory that reads from config-style kwargs."""
    return TextVQADataset(split=split, max_samples=max_samples, cache_dir=cache_dir)
