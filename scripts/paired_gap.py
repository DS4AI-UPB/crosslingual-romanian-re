#!/usr/bin/env python3
"""Paired cross-lingual gap: compute EN-RO difference per seed, then mean/std over seeds."""
import json, os, statistics, sys

DIR = sys.argv[1] if len(sys.argv) > 1 else "results_variance"
SEEDS = [1, 2, 3, 4, 5]

def f1(lang, k, s):
    f = os.path.join(DIR, f"A_{lang}_k{k}_seed{s}_metrics.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    return d.get("metrics", d)["macro_f1"]

print("Paired cross-lingual gap (EN - RO), same exemplar seed in both languages")
print(f"{'k':<4} {'n':<3} {'gap mean (pp)':<15} {'gap std (pp)':<14} {'per-seed gaps (pp)'}")
print("-" * 80)

for k in [1, 3, 5]:
    gaps = []
    for s in SEEDS:
        en, ro = f1("en", k, s), f1("ro", k, s)
        if en is None or ro is None:
            continue
        gaps.append((en - ro) * 100)
    if not gaps:
        print(f"{k:<4} -- no paired data --")
        continue
    m = statistics.mean(gaps)
    sd = statistics.stdev(gaps) if len(gaps) > 1 else 0.0
    gs = " ".join(f"{g:+.2f}" for g in gaps)
    print(f"{k:<4} {len(gaps):<3} {m:<15.2f} {sd:<14.2f} {gs}")

    # 95% CI (t-based approximation with normal for simplicity)
    if len(gaps) > 1:
        se = sd / (len(gaps) ** 0.5)
        lo, hi = m - 1.96 * se, m + 1.96 * se
        sig = "" if lo <= 0 <= hi else "  <-- excludes zero"
        print(f"     95% CI: [{lo:+.2f}, {hi:+.2f}]pp{sig}")
