#!/usr/bin/env python3
"""Extract prediction data for specific image_ids from baseline and chain_of_thought results."""

import json

IMAGE_IDS = [
    "003a8ae2ef43b901",
    "2b538a43dd933fc1",
    "181f00d3ee2b2076",
    "5ce862cbefd8458f",
    "fa9ffd5aca1e4e51",
    "1f54ecbe84b9805f",
    "1ef8743670718aa2",
    "43d24d5cd7aa9792",
    "14e0ea396adc7cca",
]

FILES = {
    "baseline": "results/prompt_eng/baseline/validation/predictions.json",
    "chain_of_thought": "results/prompt_eng/chain_of_thought/validation/predictions.json",
}

def load_and_filter(path, target_ids):
    with open(path) as f:
        data = json.load(f)
    lookup = {entry["image_id"]: entry for entry in data}
    return {img_id: lookup.get(img_id) for img_id in target_ids}

def main():
    results = {}
    for label, path in FILES.items():
        results[label] = load_and_filter(path, IMAGE_IDS)

    for img_id in IMAGE_IDS:
        print("=" * 90)
        print(f"IMAGE ID: {img_id}")
        print("=" * 90)

        for label in FILES:
            entry = results[label].get(img_id)
            if entry is None:
                print(f"\n  [{label.upper()}] — not found")
                continue

            print(f"\n  [{label.upper()}]")
            print(f"  Question:      {entry['question']}")
            print(f"  Prediction:    {entry['prediction']}")
            print(f"  Ground Truths: {entry['ground_truths']}")
            print(f"  Raw Output:")
            for line in entry["raw_output"].splitlines():
                print(f"    | {line}")

        print()

if __name__ == "__main__":
    main()
