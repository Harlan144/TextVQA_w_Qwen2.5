"""LoRA fine-tuning for Qwen2.5-VL on TextVQA."""

import logging
import math
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info

from src.data import TextVQADataset
from src.prompts import get_strategy, extract_final_answer

logger = logging.getLogger(__name__)


def setup_lora(
    model,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list[str] = None,
):
    """Apply LoRA adapters to the model.

    Returns:
        peft model with trainable LoRA parameters.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def _prepare_training_sample(processor, sample: dict, strategy_fn) -> dict:
    """Prepare a single training sample with input and label."""
    image = sample["image"]
    question = sample["question"]
    answers = sample["answers"]
    ocr_tokens = sample.get("ocr_tokens", [])

    # Use the first answer as the target
    target_answer = answers[0] if answers else ""

    # Build input messages using the prompt strategy
    messages = strategy_fn(image, question, ocr_tokens)

    # Build full conversation with assistant response for training
    messages_with_answer = messages + [
        {"role": "assistant", "content": target_answer}
    ]

    # Tokenize the full conversation
    text_full = processor.apply_chat_template(
        messages_with_answer, tokenize=False, add_generation_prompt=False
    )
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(messages)

    return {
        "text_full": text_full,
        "text_input": text_input,
        "images": image_inputs,
    }


def train_lora(
    model,
    processor,
    train_dataset: TextVQADataset,
    val_dataset: Optional[TextVQADataset] = None,
    strategy_name: str = "baseline",
    output_dir: str = "results/finetune/checkpoints",
    lr: float = 2e-5,
    epochs: int = 1,
    max_grad_norm: float = 1.0,
    log_interval: int = 50,
    save_steps: int = 500,
):
    """Run LoRA fine-tuning loop.

    Args:
        model: LoRA-wrapped model from setup_lora().
        processor: The model's processor.
        train_dataset: Training data.
        val_dataset: Optional validation data for periodic eval.
        strategy_name: Which prompt strategy to use for formatting training data.
        output_dir: Where to save checkpoints.
        lr: Learning rate.
        epochs: Number of training epochs.
        max_grad_norm: Gradient clipping value.
        log_interval: Log every N steps.
        save_steps: Save checkpoint every N steps.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    strategy_fn = get_strategy(strategy_name)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    model.train()
    global_step = 0
    total_loss = 0.0

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")

        for i in tqdm(range(len(train_dataset)), desc=f"Epoch {epoch + 1}"):
            sample = train_dataset[i]

            try:
                prepared = _prepare_training_sample(processor, sample, strategy_fn)
            except Exception as e:
                logger.warning(f"Skipping sample {i}: {e}")
                continue

            # Tokenize full conversation
            inputs = processor(
                text=[prepared["text_full"]],
                images=prepared["images"],
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            # Tokenize input-only to find where labels start
            input_only = processor(
                text=[prepared["text_input"]],
                images=prepared["images"],
                padding=True,
                return_tensors="pt",
            )
            input_len = input_only.input_ids.shape[1]

            # Create labels: mask input tokens with -100
            labels = inputs.input_ids.clone()
            labels[:, :input_len] = -100

            outputs = model(
                **{k: v for k, v in inputs.items() if k != "input_ids" and k != "attention_mask" and k != "labels"},
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            global_step += 1

            if global_step % log_interval == 0:
                avg_loss = total_loss / log_interval
                logger.info(f"Step {global_step} | Loss: {avg_loss:.4f}")
                total_loss = 0.0

            if global_step % save_steps == 0:
                ckpt_path = output_path / f"checkpoint-{global_step}"
                model.save_pretrained(str(ckpt_path))
                logger.info(f"Saved checkpoint to {ckpt_path}")

        # Save at end of epoch
        ckpt_path = output_path / f"checkpoint-epoch{epoch + 1}"
        model.save_pretrained(str(ckpt_path))
        logger.info(f"Saved epoch {epoch + 1} checkpoint to {ckpt_path}")

    # Save final
    final_path = output_path / "final"
    model.save_pretrained(str(final_path))
    logger.info(f"Training complete. Final adapter saved to {final_path}")
    return model
