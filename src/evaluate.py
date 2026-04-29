"""Evaluation metrics for TextVQA."""

import logging
from collections import defaultdict

import nltk

from src.utils import vqa_accuracy_score, vqa_accuracy_score_3, normalize_answer

logger = logging.getLogger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)


def compute_vqa_accuracy(predictions: list[str], ground_truths: list[list[str]]) -> dict:
    """Compute both accuracy metrics over a list of predictions.

    Returns:
        Dict with 'accuracy' (match any, 0-100) and 'vqa_accuracy' (official min(1, n/3), 0-100).
    """
    scores = [
        vqa_accuracy_score(pred, gts)
        for pred, gts in zip(predictions, ground_truths)
    ]
    scores_3 = [
        vqa_accuracy_score_3(pred, gts)
        for pred, gts in zip(predictions, ground_truths)
    ]
    return {
        "accuracy": 100.0 * sum(scores) / len(scores) if scores else 0.0,
        "vqa_accuracy": 100.0 * sum(scores_3) / len(scores_3) if scores_3 else 0.0,
        "num_samples": len(scores),
        "per_sample_scores": scores,
    }


def compute_bleu(predictions: list[str], ground_truths: list[list[str]]) -> dict:
    """Compute corpus-level BLEU score."""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    # References: list of list of tokenized reference answers
    # Hypothesis: list of tokenized predictions
    refs = []
    hyps = []
    for pred, gts in zip(predictions, ground_truths):
        refs.append([gt.lower().split() for gt in gts])
        hyps.append(pred.lower().split())

    smooth = SmoothingFunction().method1
    score = corpus_bleu(refs, hyps, smoothing_function=smooth)
    return {"bleu": score * 100.0}


def compute_meteor(predictions: list[str], ground_truths: list[list[str]]) -> dict:
    """Compute average METEOR score."""
    from nltk.translate.meteor_score import meteor_score as nltk_meteor

    scores = []
    for pred, gts in zip(predictions, ground_truths):
        # METEOR compares against multiple references
        score = nltk_meteor(
            [gt.lower().split() for gt in gts],
            pred.lower().split(),
        )
        scores.append(score)
    return {"meteor": 100.0 * sum(scores) / len(scores) if scores else 0.0}


def compute_rouge(predictions: list[str], ground_truths: list[list[str]]) -> dict:
    """Compute average ROUGE-L score (best match against any reference)."""
    from rouge_score.rouge_scorer import RougeScorer

    scorer = RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, gts in zip(predictions, ground_truths):
        # Take best ROUGE-L across all ground truth answers
        best = max(
            scorer.score(gt, pred)["rougeL"].fmeasure
            for gt in gts
        )
        scores.append(best)
    return {"rouge_l": 100.0 * sum(scores) / len(scores) if scores else 0.0}


def compute_f1(predictions: list[str], ground_truths: list[list[str]]) -> dict:
    """Compute average token-level F1 score (best match against any reference)."""
    scores = []
    for pred, gts in zip(predictions, ground_truths):
        pred_tokens = set(normalize_answer(pred).split())
        best_f1 = 0.0
        for gt in gts:
            gt_tokens = set(normalize_answer(gt).split())
            if not pred_tokens and not gt_tokens:
                best_f1 = 1.0
                break
            if not pred_tokens or not gt_tokens:
                continue
            common = pred_tokens & gt_tokens
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(gt_tokens)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                best_f1 = max(best_f1, f1)
        scores.append(best_f1)
    return {"f1": 100.0 * sum(scores) / len(scores) if scores else 0.0}


def compute_precision_recall(predictions: list[str], ground_truths: list[list[str]]) -> dict:
    """Compute average token-level precision and recall (best match against any reference)."""
    precisions = []
    recalls = []
    for pred, gts in zip(predictions, ground_truths):
        pred_tokens = set(normalize_answer(pred).split())
        best_p, best_r = 0.0, 0.0
        for gt in gts:
            gt_tokens = set(normalize_answer(gt).split())
            if not pred_tokens and not gt_tokens:
                best_p, best_r = 1.0, 1.0
                break
            if not pred_tokens or not gt_tokens:
                continue
            common = pred_tokens & gt_tokens
            p = len(common) / len(pred_tokens)
            r = len(common) / len(gt_tokens)
            if p + r > best_p + best_r:
                best_p, best_r = p, r
        precisions.append(best_p)
        recalls.append(best_r)
    return {
        "precision": 100.0 * sum(precisions) / len(precisions) if precisions else 0.0,
        "recall": 100.0 * sum(recalls) / len(recalls) if recalls else 0.0,
    }


def compute_per_category(
    predictions: list[str],
    ground_truths: list[list[str]],
    image_classes: list[list[str]],
) -> dict:
    """Break down VQA accuracy by image category.

    Args:
        predictions: Predicted answer strings.
        ground_truths: List of lists of ground truth answer strings.
        image_classes: List of lists of image class labels per sample.

    Returns:
        Dict with per-category accuracy and counts.
    """
    from collections import defaultdict
    category_scores = defaultdict(list)

    for pred, gts, classes in zip(predictions, ground_truths, image_classes):
        score = vqa_accuracy_score(pred, gts)
        if classes:
            for cls in classes:
                category_scores[cls].append(score)
        else:
            category_scores["_unknown"].append(score)

    per_cat = {}
    for cat, scores in sorted(category_scores.items(), key=lambda x: -len(x[1])):
        per_cat[cat] = {
            "accuracy": 100.0 * sum(scores) / len(scores),
            "count": len(scores),
        }

    return {"per_category": per_cat}


def compute_llm_judge(
    predictions: list[str],
    ground_truths: list[list[str]],
    model=None,
    processor=None,
) -> dict:
    """Use the VLM itself as a judge to score semantic similarity.

    Asks the model to rate prediction vs ground truth on a 0-1 scale.
    Only runs on a sample to save compute.
    """
    if model is None or processor is None:
        logger.warning("LLM judge requires model and processor — skipping")
        return {"llm_judge": None}

    from src.model import generate_answer

    sample_size = min(200, len(predictions))
    step = max(1, len(predictions) // sample_size)
    scores = []

    for i in range(0, len(predictions), step):
        if len(scores) >= sample_size:
            break
        pred = predictions[i]
        gt_list = ground_truths[i]
        gt_str = "; ".join(gt_list)

        judge_messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict answer evaluator. Compare a predicted answer "
                    "against a list of reference answers. If the prediction matches "
                    "ANY of the reference answers (same meaning, even if worded "
                    "differently), respond with 1.0. If it is completely wrong, "
                    "respond with 0.0. Respond with ONLY a single number between "
                    "0.0 and 1.0."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"Predicted answer: {pred}\n"
                        f"Reference answers: {gt_str}\n"
                        f"Score (0.0-1.0):"
                    )},
                ],
            },
        ]
        try:
            result = generate_answer(model, processor, judge_messages, max_new_tokens=10)
            # Parse the numeric score
            score = float(result.strip().split()[0])
            score = max(0.0, min(1.0, score))
            scores.append(score)
        except (ValueError, IndexError):
            continue

    avg = sum(scores) / len(scores) if scores else 0.0
    return {"llm_judge": avg * 100.0, "llm_judge_n_samples": len(scores)}


def compute_all_metrics(
    predictions: list[str],
    ground_truths: list[list[str]],
    metrics: list[str] = None,
    model=None,
    processor=None,
) -> dict:
    """Compute all requested metrics.

    Args:
        predictions: Predicted answer strings.
        ground_truths: List of lists of ground truth answer strings.
        metrics: List of metric names to compute. Defaults to all standard metrics.
        model: VLM model (needed for llm_judge).
        processor: VLM processor (needed for llm_judge).

    Returns:
        Dict of metric_name -> score.
    """
    if metrics is None:
        metrics = ["vqa_accuracy", "bleu", "meteor", "rouge", "f1", "precision_recall"]

    results = {}

    metric_fns = {
        "vqa_accuracy": lambda: compute_vqa_accuracy(predictions, ground_truths),
        "bleu": lambda: compute_bleu(predictions, ground_truths),
        "meteor": lambda: compute_meteor(predictions, ground_truths),
        "rouge": lambda: compute_rouge(predictions, ground_truths),
        "f1": lambda: compute_f1(predictions, ground_truths),
        "precision_recall": lambda: compute_precision_recall(predictions, ground_truths),
        "llm_judge": lambda: compute_llm_judge(predictions, ground_truths, model, processor),
    }

    for metric_name in metrics:
        if metric_name in metric_fns:
            logger.info(f"Computing {metric_name}...")
            result = metric_fns[metric_name]()
            results.update(result)
        else:
            logger.warning(f"Unknown metric: {metric_name}")

    return results


def print_metrics(metrics: dict):
    """Pretty-print metrics table."""
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    skip_keys = {"per_sample_scores", "num_samples", "llm_judge_n_samples"}
    for key, val in sorted(metrics.items()):
        if key in skip_keys:
            continue
        if val is None:
            print(f"  {key:<20s}  —")
        elif isinstance(val, float):
            print(f"  {key:<20s}  {val:.2f}")
        else:
            print(f"  {key:<20s}  {val}")
    print("=" * 50 + "\n")
