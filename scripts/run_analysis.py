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
    """Grouped bar chart comparing accuracy and VQA accuracy across experiments."""
    import numpy as np

    names = []
    acc_vals = []
    vqa_vals = []
    for name, metrics in sorted(experiments.items()):
        if "accuracy" in metrics:
            short = name.replace("prompt_eng/", "").replace("zero_shot/", "ZS: ").replace("finetune/", "FT: ")
            short = short.replace("/validation", "").replace("/test", " (test)")
            names.append(short)
            acc_vals.append(metrics["accuracy"])
            vqa_vals.append(metrics.get("vqa_accuracy", 0))

    if not names:
        print("No accuracy data found — skipping accuracy plot")
        return

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 6))
    bars1 = ax.bar(x - width / 2, acc_vals, width, label="Accuracy (match any)", color=sns.color_palette("viridis", 2)[0])
    bars2 = ax.bar(x + width / 2, vqa_vals, width, label="VQA Accuracy (min(1, n/3))", color=sns.color_palette("viridis", 2)[1])

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_title("Accuracy Comparison Across Strategies")
    ax.legend()

    for bar, val in zip(bars1, acc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars2, vqa_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=150)
    plt.close()
    print(f"Saved accuracy_comparison.png")


def plot_metrics_heatmap(experiments: dict, output_dir: Path):
    """Heatmap of all metrics across experiments."""
    metric_keys = ["accuracy", "vqa_accuracy", "bleu", "meteor", "rouge_l", "f1"]
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


def _classify_errors(predictions: list[dict]) -> dict:
    """Classify predictions into error types. Returns {type: count}."""
    from src.utils import vqa_accuracy_score

    error_types = defaultdict(int)
    total_incorrect = 0
    for item in predictions:
        score = vqa_accuracy_score(item["prediction"], item["ground_truths"])
        if score > 0:
            continue
        total_incorrect += 1
        pred = item["prediction"].lower().strip()
        gts = [g.lower().strip() for g in item["ground_truths"]]
        has_overlap = any(pred in gt or gt in pred for gt in gts if gt)

        if not pred or pred in ("", "n/a", "unknown", "i don't know", "cannot determine"):
            error_types["no_answer"] += 1
        elif has_overlap:
            error_types["partial_match"] += 1
        elif len(pred) > 50:
            error_types["verbose"] += 1
        else:
            error_types["wrong_answer"] += 1

    return dict(error_types), total_incorrect


def analyze_errors(results_dir: Path, output_dir: Path):
    """Analyze and compare error types across all strategies."""
    import numpy as np

    pred_files = list(results_dir.rglob("predictions.json"))
    if not pred_files:
        print("No prediction files found — skipping error analysis")
        return

    # Collect error breakdown for every strategy
    all_error_types = set()
    strategy_errors = {}  # {strategy_name: {error_type: count}}
    strategy_totals = {}  # {strategy_name: (total, incorrect)}
    all_summaries = {}

    for pred_file in sorted(pred_files):
        exp_name = str(pred_file.relative_to(results_dir).parent)
        # Extract short strategy name
        parts = exp_name.split("/")
        short = parts[1] if len(parts) > 1 else parts[0]

        predictions = load_json(pred_file)
        error_counts, n_incorrect = _classify_errors(predictions)
        all_error_types.update(error_counts.keys())
        strategy_errors[short] = error_counts
        strategy_totals[short] = (len(predictions), n_incorrect)

        all_summaries[short] = {
            "total": len(predictions),
            "incorrect": n_incorrect,
            "correct": len(predictions) - n_incorrect,
            "error_types": error_counts,
        }

        print(f"  {short}: {n_incorrect}/{len(predictions)} errors — {error_counts}")

    save_json(all_summaries, output_dir / "error_analysis.json")

    # Save qualitative examples for the best strategy
    best = _find_best_strategy(results_dir)
    best_pred_file = results_dir / best / "predictions.json"
    if best_pred_file.exists():
        from src.utils import vqa_accuracy_score
        preds = load_json(best_pred_file)
        correct = [p for p in preds if vqa_accuracy_score(p["prediction"], p["ground_truths"]) > 0]
        incorrect = [p for p in preds if vqa_accuracy_score(p["prediction"], p["ground_truths"]) == 0]
        save_json({
            "strategy": best,
            "correct_examples": correct[:5],
            "incorrect_examples": incorrect[:10],
        }, output_dir / "qualitative_examples.json")

    # --- Plot: grouped bar chart of error types across strategies ---
    error_type_order = ["wrong_answer", "partial_match", "verbose", "no_answer"]
    error_type_order = [e for e in error_type_order if e in all_error_types]

    strategies = sorted(strategy_errors.keys())
    x = np.arange(len(strategies))
    n_types = len(error_type_order)
    width = 0.8 / max(n_types, 1)
    colors = sns.color_palette("Set2", n_types)

    fig, ax = plt.subplots(figsize=(max(10, len(strategies) * 1.5), 6))
    for i, etype in enumerate(error_type_order):
        counts = [strategy_errors[s].get(etype, 0) for s in strategies]
        bars = ax.bar(x + i * width, counts, width, label=etype, color=colors[i])
        for bar, c in zip(bars, counts):
            if c > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        str(c), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * (n_types - 1) / 2)
    ax.set_xticklabels(strategies, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Error Type Comparison Across Strategies")
    ax.legend(title="Error Type")
    plt.tight_layout()
    plt.savefig(output_dir / "error_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved error_comparison.png")

    # --- Plot: stacked bar as percentage of total ---
    fig, ax = plt.subplots(figsize=(max(10, len(strategies) * 1.5), 6))
    bottoms = np.zeros(len(strategies))
    for i, etype in enumerate(error_type_order):
        pcts = [
            100.0 * strategy_errors[s].get(etype, 0) / strategy_totals[s][0]
            for s in strategies
        ]
        ax.bar(x, pcts, 0.6, bottom=bottoms, label=etype, color=colors[i])
        for j, (pct, bot) in enumerate(zip(pcts, bottoms)):
            if pct > 2:  # only label if visible
                ax.text(x[j], bot + pct / 2, f"{pct:.0f}%", ha="center", va="center", fontsize=7)
        bottoms += pcts

    # Add correct on top
    correct_pcts = [
        100.0 * (strategy_totals[s][0] - strategy_totals[s][1]) / strategy_totals[s][0]
        for s in strategies
    ]
    ax.bar(x, correct_pcts, 0.6, bottom=bottoms, label="correct", color=sns.color_palette("viridis", 1)[0])
    for j, (pct, bot) in enumerate(zip(correct_pcts, bottoms)):
        ax.text(x[j], bot + pct / 2, f"{pct:.0f}%", ha="center", va="center", fontsize=7, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% of Samples")
    ax.set_title("Outcome Breakdown by Strategy")
    ax.legend(title="Outcome", loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "error_breakdown_pct.png", dpi=150)
    plt.close()
    print(f"  Saved error_breakdown_pct.png")

    return all_summaries


def _find_best_strategy(results_dir: Path) -> str:
    """Return the strategy directory name with the highest accuracy."""
    best_acc, best_name = -1, "baseline"
    for metrics_file in results_dir.rglob("metrics.json"):
        metrics = load_json(metrics_file)
        acc = metrics.get("accuracy", 0)
        if acc > best_acc:
            best_acc = acc
            best_name = str(metrics_file.relative_to(results_dir).parent)
    return best_name


def plot_per_category(results_dir: Path, output_dir: Path):
    """Plot per-category accuracy for the best experiment."""
    best = _find_best_strategy(results_dir)
    cat_file = results_dir / best / "per_category.json"
    if not cat_file.exists():
        print(f"No per-category data for best strategy ({best}) — skipping")
        return
    exp_name = best
    print(f"\nPer-category analysis on: {exp_name}")

    data = load_json(cat_file)
    per_cat = data.get("per_category", {})
    if not per_cat:
        print("  Empty per-category data — skipping")
        return

    # Sort by count descending, take top 20
    sorted_cats = sorted(per_cat.items(), key=lambda x: -x[1]["count"])[:20]
    cats = [c for c, _ in sorted_cats]
    accs = [v["accuracy"] for _, v in sorted_cats]
    counts = [v["count"] for _, v in sorted_cats]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = range(len(cats))
    bars = ax1.bar(x, accs, color=sns.color_palette("viridis", len(cats)), alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, rotation=60, ha="right", fontsize=8)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(f"Per-Category Accuracy (top 20 by count) — {exp_name}")

    # Overlay count as text
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"n={count}", ha="center", va="bottom", fontsize=6)

    plt.tight_layout()
    plt.savefig(output_dir / "per_category_accuracy.png", dpi=150)
    plt.close()
    print(f"  Saved per_category_accuracy.png")


def plot_llm_judge(experiments: dict, output_dir: Path):
    """Bar chart of LLM-as-judge scores across strategies."""
    names = []
    scores = []
    for name, metrics in sorted(experiments.items()):
        if metrics.get("llm_judge") is not None:
            short = name.replace("prompt_eng/", "").replace("zero_shot/", "ZS: ").replace("finetune/", "FT: ")
            short = short.replace("/validation", "").replace("/test", " (test)")
            names.append(short)
            scores.append(metrics["llm_judge"])

    if not names:
        print("No LLM judge data found — skipping")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 6))
    colors = sns.color_palette("viridis", len(names))
    bars = ax.bar(range(len(names)), scores, color=colors)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("LLM Judge Score (%)")
    ax.set_title("LLM-as-Judge Semantic Similarity Across Strategies")

    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{s:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "llm_judge.png", dpi=150)
    plt.close()
    print(f"Saved llm_judge.png")


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
    plot_llm_judge(experiments, output_dir)

    # Per-category analysis
    print("\nPer-category analysis...")
    plot_per_category(results_dir, output_dir)

    # Error analysis
    print("\nRunning error analysis...")
    analyze_errors(results_dir, output_dir)

    # Save consolidated summary
    save_json(experiments, output_dir / "all_metrics_summary.json")
    print(f"\nAll figures and analysis saved to {output_dir}")


if __name__ == "__main__":
    main()
