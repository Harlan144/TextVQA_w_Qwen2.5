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

# Canonical strategy order used for all plots and tables.
# Grouped: baseline → prompt tweaks → reasoning → OCR variants → ablation
STRATEGY_ORDER = [
    "baseline",
    "no_system_prompt",
    "few_shot",
    "format_constraint",
    "think_hard",
    "chain_of_thought",
    "ocr_augmented",
    "ocr_cot",
    "ocr_only",
]

DISPLAY_NAMES = {
    "baseline": "Baseline",
    "no_system_prompt": "No-System-Prompt",
    "few_shot": "Few-Shot",
    "format_constraint": "Format Constraint",
    "think_hard": "Think Hard",
    "chain_of_thought": "Chain-of-Thought",
    "ocr_augmented": "OCR-Augmented",
    "ocr_cot": "OCR + CoT",
    "ocr_only": "OCR-Only",
}


def _strategy_sort_key(name: str) -> tuple:
    """Return a sort key that follows STRATEGY_ORDER.

    Works on both raw keys ('baseline') and experiment paths
    ('prompt_eng/baseline/validation').
    """
    for i, s in enumerate(STRATEGY_ORDER):
        if s in name:
            return (i, name)
    return (len(STRATEGY_ORDER), name)


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
    """Grouped bar chart comparing accuracy and LLM-as-a-Judge across experiments."""
    import numpy as np

    names = []
    acc_vals = []
    judge_vals = []
    for name, metrics in sorted(experiments.items(), key=lambda kv: _strategy_sort_key(kv[0])):
        if "accuracy" in metrics:
            short = name.replace("prompt_eng/", "").replace("zero_shot/", "ZS: ").replace("finetune/", "FT: ")
            short = short.replace("/validation", "").replace("/test", " (test)")
            names.append(DISPLAY_NAMES.get(short, short))
            acc_vals.append(metrics["accuracy"])
            judge_vals.append(metrics.get("llm_judge", 0) or 0)

    if not names:
        print("No accuracy data found — skipping accuracy plot")
        return

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 6))
    bars1 = ax.bar(x - width / 2, acc_vals, width, label="Accuracy (exact match)", color=sns.color_palette("viridis", 2)[0])
    bars2 = ax.bar(x + width / 2, judge_vals, width, label="LLM-as-a-Judge (semantic)", color=sns.color_palette("viridis", 2)[1])

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_title("Exact-Match Accuracy vs. Semantic Correctness (LLM Judge)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.35))

    for bar, val in zip(bars1, acc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars2, judge_vals):
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
    for name, metrics in sorted(experiments.items(), key=lambda kv: _strategy_sort_key(kv[0])):
        short = name.replace("prompt_eng/", "").replace("zero_shot/", "ZS: ").replace("finetune/", "FT: ")
        short = short.replace("/validation", "").replace("/test", " (test)")
        names.append(DISPLAY_NAMES.get(short, short))
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


def plot_baseline_vs_cot(results_dir: Path, output_dir: Path):
    """Generate composite figures comparing baseline vs CoT wins."""
    from PIL import Image
    from textwrap import fill
    from src.utils import vqa_accuracy_score

    baseline_preds = load_json(results_dir / "prompt_eng" / "baseline" / "validation" / "predictions.json")
    cot_preds = load_json(results_dir / "prompt_eng" / "chain_of_thought" / "validation" / "predictions.json")

    cot_by_qid = {}
    for p in cot_preds:
        cot_by_qid[p["question_id"]] = p

    examples_dir = output_dir / "examples"

    baseline_wins = []
    cot_wins = []
    for p_b in baseline_preds:
        q_id = p_b["question_id"]
        p_c = cot_by_qid.get(q_id)
        if p_c is None:
            continue
        b_score = vqa_accuracy_score(p_b["prediction"], p_b["ground_truths"])
        c_score = vqa_accuracy_score(p_c["prediction"], p_c["ground_truths"])

        img_path = examples_dir / f"{p_b['image_id']}.png"
        if not img_path.exists():
            continue

        entry = {
            "image_id": p_b["image_id"], "question": p_b["question"],
            "ground_truths": p_b["ground_truths"],
            "baseline_pred": p_b["prediction"], "cot_pred": p_c["prediction"],
            "img_path": img_path,
        }
        if b_score > 0 and c_score == 0:
            baseline_wins.append(entry)
        elif c_score > 0 and b_score == 0:
            cot_wins.append(entry)

    def _unique_answers(gts):
        seen = set()
        out = []
        for g in gts:
            g_low = g.lower().strip()
            if g_low not in seen and g_low:
                seen.add(g_low)
                out.append(g)
        return ", ".join(out[:3])

    def _make_figure(examples, title, filename, ncols=None):
        n = len(examples)
        if n == 0:
            return
        if ncols is None:
            ncols = min(n, 4)
        nrows = (n + ncols - 1) // ncols

        fig = plt.figure(figsize=(5 * ncols, 7.5 * nrows))
        gs = fig.add_gridspec(nrows * 2, ncols, height_ratios=[3, 2] * nrows,
                              hspace=0.05, wspace=0.3)

        for i, ex in enumerate(examples):
            row = (i // ncols) * 2
            col = i % ncols
            ax_img = fig.add_subplot(gs[row, col])
            img = Image.open(ex["img_path"])
            ax_img.imshow(img)
            ax_img.set_xticks([])
            ax_img.set_yticks([])

            ax_txt = fig.add_subplot(gs[row + 1, col])
            ax_txt.axis("off")

            wrap = 35
            q = fill(f"Q: {ex['question']}", width=wrap)
            gt = fill(f"GT: {_unique_answers(ex['ground_truths'])}", width=wrap)
            b_pred = fill(f"Baseline: {ex['baseline_pred']}", width=wrap)
            c_pred = ex['cot_pred']
            if len(c_pred) > 200:
                c_pred = c_pred[:197] + "..."
            c_pred = fill(f"CoT: {c_pred}", width=wrap)

            text = f"{q}\n{gt}\n{b_pred}\n{c_pred}"
            ax_txt.text(0.02, 0.95, text, transform=ax_txt.transAxes,
                        fontsize=9, family="monospace", verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                                  edgecolor="gray", alpha=0.9))

        for j in range(n, nrows * ncols):
            row = (j // ncols) * 2
            col = j % ncols
            fig.add_subplot(gs[row, col]).set_visible(False)
            fig.add_subplot(gs[row + 1, col]).set_visible(False)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
        plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {filename}")

    _make_figure(baseline_wins[:4],
                 "Baseline Correct, Chain-of-Thought Incorrect",
                 "qualitative_baseline_wins.png", ncols=4)
    _make_figure(cot_wins[:4],
                 "Chain-of-Thought Correct, Baseline Incorrect",
                 "qualitative_cot_wins.png", ncols=min(len(cot_wins), 4))
    print(f"  Found {len(baseline_wins)} baseline-wins, {len(cot_wins)} CoT-wins (with saved images)")


def plot_qualitative_examples(results_dir: Path, output_dir: Path):
    """Generate a composite figure showing easy examples (all strategies correct)
    and hard examples (all strategies wrong)."""
    from PIL import Image
    from textwrap import fill
    from src.utils import vqa_accuracy_score
    from src.data import get_dataset
    import random

    # Load predictions from strategies with full results only
    pe_dir = results_dir / "prompt_eng"
    all_strategy_preds = {}
    for strategy_dir in sorted(pe_dir.iterdir()):
        pred_file = strategy_dir / "validation" / "predictions.json"
        if pred_file.exists():
            preds = load_json(pred_file)
            if len(preds) >= 1000:  # skip incomplete runs
                all_strategy_preds[strategy_dir.name] = preds

    if len(all_strategy_preds) < 2:
        print("  Need at least 2 strategies for qualitative comparison — skipping")
        return

    strategy_names = sorted(all_strategy_preds.keys())
    print(f"  Comparing across {len(strategy_names)} strategies: {strategy_names}")

    # Index predictions by question_id for each strategy
    by_qid = {}  # {qid: {strategy: prediction_record}}
    for sname, preds in all_strategy_preds.items():
        for p in preds:
            qid = p["question_id"]
            by_qid.setdefault(qid, {})[sname] = p

    # Classify: all correct vs all wrong
    all_correct = []
    all_wrong = []
    for qid, strat_preds in by_qid.items():
        if len(strat_preds) < len(strategy_names):
            continue  # skip if missing from any strategy
        ref = list(strat_preds.values())[0]
        scores = {s: vqa_accuracy_score(p["prediction"], p["ground_truths"])
                  for s, p in strat_preds.items()}
        entry = {
            "qid": qid,
            "image_id": ref["image_id"],
            "question": ref["question"],
            "ground_truths": ref["ground_truths"],
            "preds": {s: strat_preds[s]["prediction"] for s in strategy_names},
            "scores": scores,
        }
        if all(s > 0 for s in scores.values()):
            all_correct.append(entry)
        elif all(s == 0 for s in scores.values()):
            all_wrong.append(entry)

    print(f"  All correct: {len(all_correct)}, All wrong: {len(all_wrong)}")

    # Pick diverse examples — prefer varied questions and different image_ids
    random.seed(42)
    random.shuffle(all_correct)
    random.shuffle(all_wrong)

    def _pick_diverse(candidates, n):
        """Pick up to n examples with distinct image_ids."""
        seen_imgs = set()
        picked = []
        for ex in candidates:
            if ex["image_id"] not in seen_imgs:
                seen_imgs.add(ex["image_id"])
                picked.append(ex)
            if len(picked) >= n:
                break
        return picked

    easy_picks = _pick_diverse(all_correct, 4)
    hard_picks = _pick_diverse(all_wrong, 4)

    if not easy_picks and not hard_picks:
        print("  No examples found — skipping qualitative figure")
        return

    # Load dataset to get images
    dataset = get_dataset(split="validation", max_samples=None)
    img_by_qid = {}
    for i in range(len(dataset)):
        s = dataset[i]
        qid = s.get("question_id")
        if qid in {e["qid"] for e in easy_picks + hard_picks}:
            img_by_qid[qid] = s["image"]

    # Save example images
    examples_dir = ensure_dir(output_dir / "examples")
    for ex in easy_picks + hard_picks:
        img = img_by_qid.get(ex["qid"])
        if img is not None:
            img_path = examples_dir / f"{ex['image_id']}.png"
            if not img_path.exists():
                img.save(img_path)
            ex["img"] = img

    def _unique_answers(gts):
        seen = set()
        out = []
        for g in gts:
            g_low = g.lower().strip()
            if g_low not in seen and g_low:
                seen.add(g_low)
                out.append(g)
        return ", ".join(out[:3])

    # Show a subset of strategies in the text (keep it readable)
    display_strategies = ["baseline", "chain_of_thought", "ocr_augmented"]
    display_strategies = [s for s in display_strategies if s in strategy_names]
    display_labels = {
        "baseline": "Baseline", "chain_of_thought": "CoT",
        "ocr_augmented": "OCR", "no_system_prompt": "NoSys",
        "ocr_cot": "OCR+CoT", "ocr_only": "OCR-Only",
        "few_shot": "FewShot", "format_constraint": "FmtConst",
        "think_hard": "ThinkHard",
    }

    # Build combined figure: top row = easy, bottom row = hard
    n_easy = len(easy_picks)
    n_hard = len(hard_picks)
    ncols = max(n_easy, n_hard, 1)
    nrows = 2  # easy row + hard row

    fig = plt.figure(figsize=(5 * ncols, 8 * nrows))
    gs = fig.add_gridspec(nrows * 2, ncols, height_ratios=[3, 2, 3, 2],
                          hspace=0.08, wspace=0.3)

    def _draw_row(examples, row_offset, row_label, bg_color):
        for i, ex in enumerate(examples):
            img = ex.get("img")
            if img is None:
                continue

            # Image
            ax_img = fig.add_subplot(gs[row_offset, i])
            ax_img.imshow(img)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            if i == 0:
                ax_img.set_ylabel(row_label, fontsize=12, fontweight="bold",
                                  rotation=0, labelpad=60, va="center")

            # Text
            ax_txt = fig.add_subplot(gs[row_offset + 1, i])
            ax_txt.axis("off")

            wrap = 35
            lines = [fill(f"Q: {ex['question']}", width=wrap)]
            lines.append(fill(f"GT: {_unique_answers(ex['ground_truths'])}", width=wrap))
            for s in display_strategies:
                pred = ex["preds"].get(s, "—")
                if len(pred) > 60:
                    pred = pred[:57] + "..."
                label = display_labels.get(s, s)
                lines.append(fill(f"{label}: {pred}", width=wrap))

            text = "\n".join(lines)
            ax_txt.text(0.02, 0.95, text, transform=ax_txt.transAxes,
                        fontsize=9, family="monospace", verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color,
                                  edgecolor="gray", alpha=0.9))

        # Hide unused columns
        for j in range(len(examples), ncols):
            fig.add_subplot(gs[row_offset, j]).set_visible(False)
            fig.add_subplot(gs[row_offset + 1, j]).set_visible(False)

    _draw_row(easy_picks, 0, "Easy\n(all correct)", "#d4edda")
    _draw_row(hard_picks, 2, "Hard\n(all wrong)", "#f8d7da")

    fig.suptitle("Qualitative Examples: Easy vs. Hard",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(output_dir / "qualitative_examples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved qualitative_examples.png")


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

    strategies = sorted(strategy_errors.keys(), key=_strategy_sort_key)
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
    ax.set_xticklabels([DISPLAY_NAMES.get(s, s) for s in strategies], rotation=45, ha="right", fontsize=9)
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
    ax.set_xticklabels([DISPLAY_NAMES.get(s, s) for s in strategies], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% of Samples")
    ax.set_title("Outcome Breakdown by Strategy")
    ax.legend(title="Outcome", loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig(output_dir / "error_breakdown_pct.png", dpi=150, bbox_inches="tight")
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
    for name, metrics in sorted(experiments.items(), key=lambda kv: _strategy_sort_key(kv[0])):
        if metrics.get("llm_judge") is not None:
            short = name.replace("prompt_eng/", "").replace("zero_shot/", "ZS: ").replace("finetune/", "FT: ")
            short = short.replace("/validation", "").replace("/test", " (test)")
            names.append(DISPLAY_NAMES.get(short, short))
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


def export_metrics_tsv(experiments: dict, output_dir: Path):
    """Export full metrics comparison as a TSV file."""
    metric_keys = ["accuracy", "vqa_accuracy", "bleu", "meteor", "rouge_l", "f1", "precision", "recall", "llm_judge"]
    headers = ["Strategy", "Accuracy", "VQA Acc", "BLEU", "METEOR", "ROUGE-L", "F1", "Precision", "Recall", "LLM Judge"]

    rows = []
    for strategy in STRATEGY_ORDER:
        for key, metrics in experiments.items():
            if strategy in key:
                vals = [f"{metrics.get(m, 0):.1f}" for m in metric_keys]
                rows.append([DISPLAY_NAMES.get(strategy, strategy)] + vals)
                break

    path = output_dir / "full_metrics_comparison.tsv"
    with open(path, "w") as f:
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")
    print(f"Saved full_metrics_comparison.tsv")


def export_error_breakdown_tsv(results_dir: Path, output_dir: Path):
    """Export error breakdown as a TSV file."""
    error_json = output_dir / "error_analysis.json"
    if not error_json.exists():
        print("No error_analysis.json found — skipping error TSV")
        return

    data = load_json(error_json)
    error_types = ["wrong_answer", "partial_match", "verbose", "no_answer"]
    headers = ["Strategy", "Total", "Correct", "Correct %",
               "Wrong Answer", "Wrong Answer %", "Partial Match", "Partial Match %",
               "Verbose", "Verbose %", "No Answer", "No Answer %"]

    rows = []
    for strategy in STRATEGY_ORDER:
        if strategy not in data:
            continue
        d = data[strategy]
        total = d["total"]
        correct = d["correct"]
        row = [DISPLAY_NAMES.get(strategy, strategy), str(total), str(correct), f"{100*correct/total:.1f}"]
        for etype in error_types:
            count = d["error_types"].get(etype, 0)
            row.extend([str(count), f"{100*count/total:.1f}"])
        rows.append(row)

    path = output_dir / "error_breakdown.tsv"
    with open(path, "w") as f:
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")
    print(f"Saved error_breakdown.tsv")


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

    # Qualitative examples
    print("\nGenerating qualitative example figures...")
    plot_baseline_vs_cot(results_dir, output_dir)
    plot_qualitative_examples(results_dir, output_dir)

    # Error analysis
    print("\nRunning error analysis...")
    analyze_errors(results_dir, output_dir)

    # Save consolidated summary
    save_json(experiments, output_dir / "all_metrics_summary.json")

    # Export TSV tables
    print("\nExporting TSV tables...")
    export_metrics_tsv(experiments, output_dir)
    export_error_breakdown_tsv(results_dir, output_dir)

    print(f"\nAll figures and analysis saved to {output_dir}")


if __name__ == "__main__":
    main()
