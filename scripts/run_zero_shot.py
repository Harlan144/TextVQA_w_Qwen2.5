#!/usr/bin/env python3
"""Zero-shot evaluation of Qwen2.5-VL on TextVQA."""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, merge_configs, set_seed, save_json, ensure_dir, setup_logging
from src.data import get_dataset
from src.model import load_model, generate_answer
from src.prompts import get_strategy, extract_final_answer, STRATEGIES
from src.evaluate import compute_all_metrics, print_metrics


def run_zero_shot(config: dict):
    setup_logging()
    set_seed(config.get("seed", 42))

    model_cfg = config["model"]
    data_cfg = config.get("data", {})
    eval_cfg = config.get("evaluation", {})
    output_base = Path(config.get("output", {}).get("base_dir", "results"))

    # Strategy for zero-shot (default: baseline)
    strategy_name = config.get("strategy", "baseline")
    strategy_fn = get_strategy(strategy_name)

    # Load model
    model, processor = load_model(
        model_name=model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        device_map=model_cfg.get("device_map", "auto"),
    )

    # Load data
    splits = config.get("splits", ["validation"])
    max_new_tokens = model_cfg.get("max_new_tokens", 128)

    for split in splits:
        print(f"\n{'='*60}")
        print(f"  Zero-shot evaluation on {split} set (strategy: {strategy_name})")
        print(f"{'='*60}\n")

        dataset = get_dataset(
            split=split,
            max_samples=data_cfg.get("max_samples"),
            cache_dir=data_cfg.get("cache_dir"),
        )

        predictions = []
        ground_truths = []
        image_classes_list = []
        raw_outputs = []
        results_detail = []

        for i in tqdm(range(len(dataset)), desc=f"Evaluating {split}"):
            sample = dataset[i]
            messages = strategy_fn(
                sample["image"], sample["question"], sample.get("ocr_tokens")
            )

            raw_answer = generate_answer(
                model, processor, messages, max_new_tokens=max_new_tokens
            )

            # For CoT strategies, extract the final answer
            if strategy_name in ("chain_of_thought", "ocr_cot"):
                answer = extract_final_answer(raw_answer)
            else:
                answer = raw_answer

            predictions.append(answer)
            ground_truths.append(sample["answers"])
            image_classes_list.append(sample.get("image_classes", []))
            raw_outputs.append(raw_answer)

            results_detail.append({
                "question_id": sample.get("question_id"),
                "image_id": sample.get("image_id"),
                "question": sample["question"],
                "prediction": answer,
                "raw_output": raw_answer,
                "ground_truths": sample["answers"],
                "image_classes": sample.get("image_classes", []),
            })

        # Compute metrics
        metric_names = eval_cfg.get("metrics", ["vqa_accuracy", "bleu", "meteor", "rouge", "f1"])
        if eval_cfg.get("llm_judge"):
            metric_names.append("llm_judge")

        metrics = compute_all_metrics(
            predictions, ground_truths,
            metrics=metric_names,
            model=model if "llm_judge" in metric_names else None,
            processor=processor if "llm_judge" in metric_names else None,
        )

        # Remove per-sample scores from summary (too large)
        per_sample = metrics.pop("per_sample_scores", None)
        print_metrics(metrics)

        # Per-category breakdown
        from src.evaluate import compute_per_category
        cat_metrics = compute_per_category(predictions, ground_truths, image_classes_list)

        # Save results
        out_dir = ensure_dir(output_base / "zero_shot" / strategy_name / split)
        save_json(metrics, out_dir / "metrics.json")
        save_json(results_detail, out_dir / "predictions.json")
        save_json(cat_metrics, out_dir / "per_category.json")
        if per_sample is not None:
            save_json(
                {"per_sample_vqa_accuracy": per_sample},
                out_dir / "per_sample_scores.json",
            )

        print(f"Results saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Zero-shot TextVQA evaluation")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to base config YAML")
    parser.add_argument("--strategy", type=str, default="baseline",
                        choices=list(STRATEGIES.keys()),
                        help="Prompt strategy to use")
    parser.add_argument("--splits", nargs="+", default=["validation"],
                        help="Dataset splits to evaluate")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Override max samples (for quick testing)")
    parser.add_argument("--dtype", type=str, default=None,
                        choices=["bfloat16", "4bit"],
                        help="Override model dtype")
    args = parser.parse_args()

    config = load_config(args.config)
    config["strategy"] = args.strategy
    config["splits"] = args.splits
    if args.max_samples is not None:
        config.setdefault("data", {})["max_samples"] = args.max_samples
    if args.dtype is not None:
        config.setdefault("model", {})["dtype"] = args.dtype

    run_zero_shot(config)


if __name__ == "__main__":
    main()
