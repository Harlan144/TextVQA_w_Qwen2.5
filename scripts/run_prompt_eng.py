#!/usr/bin/env python3
"""Run prompt engineering experiments across all strategies on TextVQA."""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, set_seed, save_json, ensure_dir, setup_logging
from src.data import get_dataset
from src.model import load_model, generate_answer
from src.prompts import get_strategy, extract_final_answer, list_strategies, STRATEGIES
from src.evaluate import compute_all_metrics, print_metrics


def evaluate_strategy(
    strategy_name: str,
    model,
    processor,
    dataset,
    max_new_tokens: int = 128,
) -> tuple[list[str], list[list[str]], list[dict]]:
    """Run a single strategy across the dataset.

    Returns:
        (predictions, ground_truths, detail_records)
    """
    strategy_fn = get_strategy(strategy_name)
    predictions = []
    ground_truths = []
    details = []

    for i in tqdm(range(len(dataset)), desc=strategy_name):
        sample = dataset[i]
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

    return predictions, ground_truths, details


def run_prompt_engineering(config: dict):
    setup_logging()
    set_seed(config.get("seed", 42))

    model_cfg = config["model"]
    data_cfg = config.get("data", {})
    eval_cfg = config.get("evaluation", {})
    output_base = Path(config.get("output", {}).get("base_dir", "results"))

    strategies = config.get("strategies", list_strategies())
    split = config.get("split", "validation")
    max_new_tokens = model_cfg.get("max_new_tokens", 128)

    # Load model once
    model, processor = load_model(
        model_name=model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        device_map=model_cfg.get("device_map", "auto"),
    )

    # Load dataset once
    dataset = get_dataset(
        split=split,
        max_samples=data_cfg.get("max_samples"),
        cache_dir=data_cfg.get("cache_dir"),
    )

    metric_names = eval_cfg.get("metrics", ["vqa_accuracy", "bleu", "meteor", "rouge", "f1"])
    all_results = {}

    for strategy_name in strategies:
        print(f"\n{'='*60}")
        print(f"  Strategy: {strategy_name}")
        print(f"{'='*60}\n")

        predictions, ground_truths, details = evaluate_strategy(
            strategy_name, model, processor, dataset, max_new_tokens
        )

        metrics = compute_all_metrics(predictions, ground_truths, metrics=metric_names)
        per_sample = metrics.pop("per_sample_scores", None)
        print_metrics(metrics)

        # Save per-strategy results
        out_dir = ensure_dir(output_base / "prompt_eng" / strategy_name / split)
        save_json(metrics, out_dir / "metrics.json")
        save_json(details, out_dir / "predictions.json")
        if per_sample is not None:
            save_json({"per_sample_vqa_accuracy": per_sample}, out_dir / "per_sample_scores.json")

        all_results[strategy_name] = metrics
        print(f"Results saved to {out_dir}")

    # Save comparison summary
    summary_dir = ensure_dir(output_base / "prompt_eng")
    save_json(all_results, summary_dir / "comparison.json")

    # Print comparison table
    print(f"\n{'='*70}")
    print("  STRATEGY COMPARISON")
    print(f"{'='*70}")
    header = f"  {'Strategy':<25s} {'VQA Acc':>10s} {'BLEU':>10s} {'METEOR':>10s} {'ROUGE-L':>10s}"
    print(header)
    print(f"  {'-'*65}")
    for name, m in all_results.items():
        print(f"  {name:<25s} {m.get('vqa_accuracy', 0):>10.2f} {m.get('bleu', 0):>10.2f} {m.get('meteor', 0):>10.2f} {m.get('rouge_l', 0):>10.2f}")
    print(f"{'='*70}\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Prompt engineering experiments on TextVQA")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--strategies", nargs="+", default=None,
                        choices=list(STRATEGIES.keys()),
                        help="Strategies to evaluate (default: all)")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dtype", type=str, default=None, choices=["bfloat16", "4bit"])
    args = parser.parse_args()

    config = load_config(args.config)
    if args.strategies:
        config["strategies"] = args.strategies
    config["split"] = args.split
    if args.max_samples is not None:
        config.setdefault("data", {})["max_samples"] = args.max_samples
    if args.dtype is not None:
        config.setdefault("model", {})["dtype"] = args.dtype

    run_prompt_engineering(config)


if __name__ == "__main__":
    main()
