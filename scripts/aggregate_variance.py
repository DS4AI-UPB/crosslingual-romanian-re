#!/usr/bin/env python3
"""Aggregate few-shot variance runs: mean +/- std of macro F1-Score over seeds."""
import json, glob, statistics, os, sys

DIR = sys.argv[1] if len(sys.argv) > 1 else "results_variance"
SEEDS = [1, 2, 3, 4, 5]

rows = []
print(f"{'Lang':<5} {'k':<3} {'n':<3} {'mean F1':<9} {'std':<8} {'min':<8} {'max':<8}  values")
print("-" * 85)

for lang in ["en", "ro"]:
    for k in [1, 3, 5]:
        vals = []
        missing = []
        for s in SEEDS:
            f = os.path.join(DIR, f"A_{lang}_k{k}_seed{s}_metrics.json")
            if not os.path.exists(f):
                missing.append(s)
                continue
            d = json.load(open(f))
            m = d.get("metrics", d)
            vals.append(m["macro_f1"])
        if not vals:
            print(f"{lang:<5} {k:<3} -- no data --")
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        vs = " ".join(f"{v:.3f}" for v in vals)
        print(f"{lang:<5} {k:<3} {len(vals):<3} {mean:<9.4f} {std:<8.4f} {min(vals):<8.4f} {max(vals):<8.4f}  {vs}")
        if missing:
            print(f"      (lipsesc seed-urile: {missing})")
        rows.append((lang, k, mean, std, len(vals)))

# Cross-lingual gap per k, cu propagarea erorii
print("\n" + "=" * 85)
print("Cross-lingual gap (EN - RO) in pp, cu std propagat")
print(f"{'k':<4} {'EN mean':<10} {'RO mean':<10} {'gap (pp)':<10} {'gap std (pp)':<12}")
print("-" * 85)
d = {(l, k): (m, s) for l, k, m, s, n in rows}
for k in [1, 3, 5]:
    if ("en", k) in d and ("ro", k) in d:
        en_m, en_s = d[("en", k)]
        ro_m, ro_s = d[("ro", k)]
        gap = (en_m - ro_m) * 100
        gap_std = ((en_s ** 2 + ro_s ** 2) ** 0.5) * 100
        print(f"{k:<4} {en_m:<10.4f} {ro_m:<10.4f} {gap:<10.2f} {gap_std:<12.2f}")

# LaTeX
print("\n" + "=" * 85)
print("Tabel LaTeX (macro F1-Score, mean +/- std peste 5 seed-uri, 500 exemple/rulare)")
print("-" * 85)
print(r"\begin{tabular}{lcc}")
print(r"\toprule")
print(r"\textbf{Setting} & \textbf{English} & \textbf{Romanian} \\")
print(r"\midrule")
for k in [1, 3, 5]:
    if ("en", k) in d and ("ro", k) in d:
        en_m, en_s = d[("en", k)]
        ro_m, ro_s = d[("ro", k)]
        print(f"Few-shot ($k={k}$) & ${en_m:.3f} \\pm {en_s:.3f}$ & ${ro_m:.3f} \\pm {ro_s:.3f}$ \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
