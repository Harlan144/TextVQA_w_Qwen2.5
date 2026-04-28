#!/usr/bin/env python3
"""Analyze and visualize results across all experiments."""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import load_json, ensure_dir, setup_logging


def load_all_results(results_dir: Path) -> dict:
    """Scan the results directory and load all metrics.json files."""
    experiments = {}

    # Zero-shot results
    zs_dir = results_dir / "zero_shot"
    if zs_dir.exists():
        for strategy_dir in sorted(zs_dir.iterdir()):
            if not strategy_dir.is_dir():
                continue
            for split_dir in sorted(strategy_dir.iterdir()):
                metrics_file = split_dir / "metrics.json"
                if metrics_file.exists():
                    key = f"zero_shot/{strategy_dir.name}/{split_dir.name}"
                    experiments[key] = load_json(metrics_file)

    # Prompt engineering results
    pe_dir = results_dir / "prompt_eng"
    if pe_dir.exists():
        for strategy_dir in sorted(pe_dir.iterdir()):
            if not strategy_dir.is_dir():
                continue
            for split_dir in sorted(strategy_dir.iterdir()):
                metrics_file = split_dir / "metrics.json"
                if metrics_file.exists():
                    key = f"prompt_eng/{strategy_dir.name}/{split_dir.name}"
                    experiments[key] = load_json(metrics_file)

    # Fine-tuning results
    ft_dir = results_dir / "finetune" / "eval"
    if ft_dir.exists():
        for split_dir in sorted(ft_dir.iterdir()):
            metrics_file = split_dir / "metrics.json"
            if metrics_file.exists():
                key = f"finetune/{split_dir.name}"
                experiments[key] = load_json(metrics_file)

    return experiments


def plot_accuracy_comparison(experiments: dict, output_dir: Path):
    """Bar chart comparing VQA accuracy across experiments."""
    names = []
    accuracies = []
    for name, metrics in sorted(experiments.items()):
        if "vqa_accuracy" in metrics:
            # Shorten name for display
            short = name.replace("prompt_eng/", "PE: ").replace("zero_shot/", "ZS: ").replace("finetune/", "FT: ")
            short = short.replace("/validation", "").replace("/test", " (test)")
            names.append(short)
            accuracies.append(metrics["vqa_accuracy"])

    if not names:
        print("No accuracy data found — skipping accuracy plot")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 6))
    colors = sns.color_palette("viridis", len(names))
    bars = ax.bar(range(len(names)), accuracies, color=colors)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("VQA Accuracy (%)")
    ax.set_title("VQA Accuracy Comparison Across Strategies")

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=150)
    plt.close()
    print(f"Saved accuracy_comparison.png")


def plot_metrics_heatmap(experiments: dict, output_dir: Path):
    """Heatmap of all metrics across experiments."""
    metric_keys = ["vqa_accuracy", "bleu", "meteor", "rouge_l", "f1"]
    available_metrics = set()
    for metrics in experiments.values():
        available_metrics.update(k for k in metrics if k in metric_keys)
    available_metrics = sorted(available_metrics)

    if not available_metrics:
        print("No metrics data found — skipping heatmap")
        return

    names = []
    data = []
    for name, metrics in sorted(experiments.items()):
        short = name.replace("prompt_eng/", "PE: ").replace("zero_shot/", "ZS: ").replace("finetune/", "FT: ")
        short = short.replace("/validation", "").replace("/test", " (test)")
        names.append(short)
        data.append([metrics.get(m, 0) for m in available_metrics])

    fig, ax = plt.subplots(figsize=(max(8, len(available_metrics) * 2), max(4, len(names) * 0.6)))
    sns.heatmap(
        data, annot=True, fmt=".1f", cmap="YlGnBu",
        xticklabels=available_metrics, yticklabels=names,
        ax=ax,
    )
    ax.set_title("Metrics Comparison Across Strategies")
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved metrics_heatmap.png")


def analyze_errors(results_dir: Path, output_dir: Path):
    """Analyze error types from prediction files."""
    # Find all prediction files
    pred_files = list(results_dir.rglob("predictions.json"))
    if not pred_files:
        print("No prediction files found — skipping error analysis")
        return

    # Use the latest/best prediction file (prefer prompt_eng over zero_shot)
    pred_files.sort(key=lambda p: ("prompt_eng" in str(p), p.stat().st_mtime))
    pred_file = pred_files[-1]
    exp_name = str(pred_file.relative_to(results_dir).parent)
    print(f"\nError analysis on: {exp_name}")

    predictions = load_json(pred_file)

    # Categorize results
    correct = []
    incorrect = []

    for item in predictions:
        from src.utils import vqa_accuracy_score
        score = vqa_accuracy_score(item["prediction"], item["ground_truths"])
        item["vqa_score"] = score
        if score > 0:
            correct.append(item)
        else:
            incorrect.append(item)

    print(f"  Correct: {len(correct)} ({100*len(correct)/len(predictions):.1f}%)")
    print(f"  Incorrect: {len(incorrect)} ({100*len(incorrect)/len(predictions):.1f}%)")

    # Classify error types
    error_types = defaultdict(list)
    for item in incorrect:
        pred = item["prediction"].lower().strip()
        gts = [g.lower().strip() for g in item["ground_truths"]]

        # Check if prediction is a substring or superstring of any GT
        has_overlap = any(pred in gt or gt in pred for gt in gts if gt)

        if not pred or pred in ("", "n/a", "unknown", "i don't know", "cannot determine"):
            error_types["no_answer"].append(item)
        elif has_overlap:
            error_types["partial_match"].append(item)
        elif len(pred) > 50:
            error_types["verbose_response"].append(item)
        else:
            error_types["wrong_answer"].append(item)

    print("\n  Error type breakdown:")
    for etype, items in sorted(error_types.items(), key=lambda x: -len(x[1])):
        print(f"    {etype}: {len(items)} ({100*len(items)/len(incorrect):.1f}% of errors)")

    # Save error analysis
    error_summary = {
        "experiment": exp_name,
        "total": len(predictions),
        "correct": len(correct),
        "incorrect": len(incorrect),
        "error_types": {k: len(v) for k, v in error_types.items()},
    }
    save_json(error_summary, output_dir / "error_analysis.json")

    # Save qualitative examples
    examples = {
        "correct_examples": [
            {k: v for k, v in item.items() if k != "vqa_score"}
            for item in correct[:5]
        ],
        "incorrect_examples": [
            {k: v for k, v in item.items() if k != "vqa_score"}
            for item in incorrect[:10]
        ],
    }
    save_json(examples, output_dir / "qualitative_examples.json")
    print(f"\n  Saved error_analysis.json and qualitative_examples.json")

    # Plot error type distribution
    if error_types:
        fig, ax = plt.subplots(figsize=(8, 5))
        types = list(error_types.keys())
        counts = [len(error_types[t]) for t in types]
        colors = sns.color_palette("Set2", len(types))
        ax.barh(types, counts, color=colors)
        ax.set_xlabel("Count")
        ax.set_title(f"Error Type Distribution ({exp_name})")
        for i, count in enumerate(counts):
            ax.text(count + 0.5, i, str(count), va="center")
        plt.tight_layout()
        plt.savefig(output_dir / "error_distribution.png", dpi=150)
        plt.close()
        print(f"  Saved error_distribution.png")

    return error_summary


def save_json(data, path):
    """Save data as JSON."""
    from src.utils import save_json as _save
    _save(data, path)


def main():
    parser = argparse.ArgumentParser(description="Analyze TextVQA experiment results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Path to results directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to save figures (default: results/figures)")
    args = parser.parse_args()

    setup_logging()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    ensure_dir(output_dir)

    # Load all experiment results
    print("Loading experiment results...")
    experiments = load_all_results(results_dir)
    if not experiments:
        print("No results found. Run experiments first.")
        return

    print(f"Found {len(experiments)} experiment(s):")
    for name in sorted(experiments):
        print(f"  - {name}")

    # Generate plots
    print("\nGenerating plots...")
    plot_accuracy_comparison(experiments, output_dir)
    plot_metrics_heatmap(experiments, output_dir)

    # Error analysis
    print("\nRunning error analysis...")
    analyze_errors(results_dir, output_dir)

    # Save consolidated summary
    save_json(experiments, output_dir / "all_metrics_summary.json")
    print(f"\nAll figures and analysis saved to {output_dir}")


if __name__ == "__main__":
    main()
