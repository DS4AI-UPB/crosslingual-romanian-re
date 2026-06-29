#!/usr/bin/env python3
"""
Paired bootstrap significance test for macro-F1 differences between models.

Reads the prediction .jsonl files written by the inference/baseline scripts
(fields: gold, predicted) and tests whether the macro-F1 difference between
two models on the same test set is statistically significant.

Usage:
    python 11_significance.py \
        --a results/A_qlora_ro.jsonl \
        --b results/A_xlmr_ro.jsonl \
        --n-boot 10000

Both files must cover the same test set in the same order (same gold labels).
"""

import argparse
import json
import random

from sklearn.metrics import f1_score


COARSE_RELATIONS = [
    "Cause-Effect", "Instrument-Agency", "Product-Producer",
    "Content-Container", "Entity-Origin", "Entity-Destination",
    "Component-Whole", "Member-Collection", "Message-Topic", "Other",
]


def coarse(label):
    if "(" in label:
        return label.split("(")[0]
    return label


def load(path):
    golds, preds = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            golds.append(coarse(o["gold"]))
            preds.append(coarse(o["predicted"]))
    return golds, preds


def macro_f1(golds, preds, idx):
    g = [golds[i] for i in idx]
    p = [preds[i] for i in idx]
    return f1_score(g, p, labels=COARSE_RELATIONS, average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="predictions file for model A")
    ap.add_argument("--b", required=True, help="predictions file for model B")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    golds_a, preds_a = load(args.a)
    golds_b, preds_b = load(args.b)

    assert len(golds_a) == len(golds_b), "files differ in length"
    # Sanity check: gold labels should match between the two files
    mismatches = sum(1 for x, y in zip(golds_a, golds_b) if x != y)
    if mismatches:
        print(f"WARNING: {mismatches} gold labels differ between files; "
              f"make sure both cover the same test set in the same order.")

    n = len(golds_a)
    full_idx = list(range(n))

    f1_a = macro_f1(golds_a, preds_a, full_idx)
    f1_b = macro_f1(golds_b, preds_b, full_idx)
    observed_diff = f1_a - f1_b

    print(f"Model A: {args.a}")
    print(f"  macro-F1 = {f1_a:.4f}")
    print(f"Model B: {args.b}")
    print(f"  macro-F1 = {f1_b:.4f}")
    print(f"Observed difference (A - B) = {observed_diff:.4f}")
    print(f"Running {args.n_boot} bootstrap resamples...")

    # Paired bootstrap: resample test instances with replacement,
    # recompute the difference each time.
    diffs = []
    count_le_zero = 0
    for _ in range(args.n_boot):
        idx = [random.randrange(n) for _ in range(n)]
        d = macro_f1(golds_a, preds_a, idx) - macro_f1(golds_b, preds_b, idx)
        diffs.append(d)
        if (observed_diff > 0 and d <= 0) or (observed_diff < 0 and d >= 0):
            count_le_zero += 1

    diffs.sort()
    lo = diffs[int(0.025 * args.n_boot)]
    hi = diffs[int(0.975 * args.n_boot)]
    p_value = count_le_zero / args.n_boot

    print(f"\n95% CI for the difference: [{lo:.4f}, {hi:.4f}]")
    print(f"Bootstrap p-value (two-sided, H0: diff = 0): {p_value:.4f}")
    if p_value < 0.05:
        print("=> Difference is statistically significant at alpha = 0.05.")
    else:
        print("=> Difference is NOT statistically significant at alpha = 0.05.")


if __name__ == "__main__":
    main()
