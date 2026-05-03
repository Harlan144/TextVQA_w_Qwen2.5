#!/usr/bin/env python3
"""Re-run LLM-as-a-Judge on all predictions for all strategies (full dataset)."""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_json, save_json, setup_logging, load_config
from src.model import load_model
from src.evaluate import compute_llm_judge


def main():
    parser = argparse.ArgumentParser(description="Run LLM judge on all strategies")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--dtype", type=str, default="4bit", choices=["bfloat16", "4bit"])
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Strategies to evaluate (default: all found)")
    args = parser.parse_args()

    setup_logging()

    config = load_config(args.config)
    model_cfg = config.get("model", {})

    model, processor = load_model(
        model_name=model_cfg.get("name", "Qwen/Qwen2.5-VL-3B-Instruct"),
        dtype=args.dtype,
        device_map=model_cfg.get("device_map", "auto"),
    )

    results_dir = Path("results/prompt_eng")
    strategies = args.strategies
    if strategies is None:
        strategies = sorted(
            d.name for d in results_dir.iterdir()
            if d.is_dir() and (d / "validation" / "predictions.json").exists()
        )

    for strategy in strategies:
        pred_file = results_dir / strategy / "validation" / "predictions.json"
        metrics_file = results_dir / strategy / "validation" / "metrics.json"

        if not pred_file.exists():
            print(f"Skipping {strategy}: no predictions.json")
            continue

        preds = load_json(pred_file)
        metrics = load_json(metrics_file) if metrics_file.exists() else {}

        pred_strs = [p["prediction"] for p in preds]
        gt_lists = [p["ground_truths"] for p in preds]

        print(f"\n{'='*60}")
        print(f"  LLM Judge: {strategy} ({len(preds)} samples)")
        print(f"{'='*60}")

        judge_result = compute_llm_judge(pred_strs, gt_lists, model=model, processor=processor)
        print(f"  Score: {judge_result['llm_judge']:.1f}% ({judge_result['llm_judge_n_samples']} samples)")

        metrics["llm_judge"] = judge_result["llm_judge"]
        metrics["llm_judge_n_samples"] = judge_result["llm_judge_n_samples"]
        save_json(metrics, metrics_file)
        print(f"  Updated {metrics_file}")


if __name__ == "__main__":
    main()
