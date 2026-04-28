#!/usr/bin/env python3
"""Fine-tune Qwen2.5-VL on TextVQA with LoRA, then evaluate."""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, set_seed, save_json, ensure_dir, setup_logging
from src.data import get_dataset
from src.model import load_model, generate_answer
from src.train import setup_lora, train_lora
from src.prompts import get_strategy, extract_final_answer, STRATEGIES
from src.evaluate import compute_all_metrics, print_metrics


def run_finetune(config: dict):
    setup_logging()
    set_seed(config.get("seed", 42))

    model_cfg = config["model"]
    data_cfg = config.get("data", {})
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})
    output_base = Path(config.get("output", {}).get("base_dir", "results"))

    strategy_name = config.get("strategy", "baseline")

    # Load model
    model, processor = load_model(
        model_name=model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        device_map=model_cfg.get("device_map", "auto"),
    )

    # Apply LoRA
    model = setup_lora(
        model,
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules"),
    )

    # Load data
    train_dataset = get_dataset(
        split="train",
        max_samples=data_cfg.get("max_train_samples"),
        cache_dir=data_cfg.get("cache_dir"),
    )

    val_dataset = get_dataset(
        split="validation",
        max_samples=data_cfg.get("max_val_samples"),
        cache_dir=data_cfg.get("cache_dir"),
    )

    # Train
    checkpoint_dir = str(output_base / "finetune" / "checkpoints")
    model = train_lora(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        strategy_name=strategy_name,
        output_dir=checkpoint_dir,
        lr=train_cfg.get("lr", 2e-5),
        epochs=train_cfg.get("epochs", 1),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        log_interval=train_cfg.get("log_interval", 50),
        save_steps=train_cfg.get("save_steps", 500),
    )

    # Evaluate fine-tuned model
    print(f"\n{'='*60}")
    print(f"  Evaluating fine-tuned model on validation set")
    print(f"{'='*60}\n")

    model.eval()
    strategy_fn = get_strategy(strategy_name)
    max_new_tokens = model_cfg.get("max_new_tokens", 128)

    predictions = []
    ground_truths = []
    details = []

    for i in tqdm(range(len(val_dataset)), desc="Evaluating"):
        sample = val_dataset[i]
        messages = strategy_fn(
            sample["image"], sample["question"], sample.get("ocr_tokens")
        )
        raw_answer = generate_answer(model, processor, messages, max_new_tokens=max_new_tokens)

        if strategy_name in ("chain_of_thought", "ocr_cot"):
            answer = extract_final_answer(raw_answer)
        else:
            answer = raw_answer

        predictions.append(answer)
        ground_truths.append(sample["answers"])
        details.append({
            "question_id": sample.get("question_id"),
            "image_id": sample.get("image_id"),
            "question": sample["question"],
            "prediction": answer,
            "raw_output": raw_answer,
            "ground_truths": sample["answers"],
        })

    metric_names = eval_cfg.get("metrics", ["vqa_accuracy", "bleu", "meteor", "rouge", "f1"])
    metrics = compute_all_metrics(predictions, ground_truths, metrics=metric_names)
    metrics.pop("per_sample_scores", None)
    print_metrics(metrics)

    # Save results
    out_dir = ensure_dir(output_base / "finetune" / "eval" / "validation")
    save_json(metrics, out_dir / "metrics.json")
    save_json(details, out_dir / "predictions.json")
    print(f"Results saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL on TextVQA with LoRA")
    parser.add_argument("--config", type=str, default="configs/finetune.yaml")
    parser.add_argument("--strategy", type=str, default=None, choices=list(STRATEGIES.keys()))
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--dtype", type=str, default=None, choices=["bfloat16", "4bit"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.strategy:
        config["strategy"] = args.strategy
    if args.max_train_samples is not None:
        config.setdefault("data", {})["max_train_samples"] = args.max_train_samples
    if args.max_val_samples is not None:
        config.setdefault("data", {})["max_val_samples"] = args.max_val_samples
    if args.dtype:
        config.setdefault("model", {})["dtype"] = args.dtype
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.lr:
        config.setdefault("training", {})["lr"] = args.lr

    run_finetune(config)


if __name__ == "__main__":
    main()
