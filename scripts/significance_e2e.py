#!/usr/bin/env python3
"""
Paired bootstrap significance test for End-to-End RE predictions.

Unlike the classification test, which resamples macro F1-Score, this one
resamples the three end-to-end metrics: exact match, relation match and
entity match. Both files must contain predictions for the same test set in
the same order.

    python significance_e2e.py --a results_qwen/B_qlora_ro.jsonl \
                               --b results/B_qlora_ro.jsonl
"""
import argparse, json, random


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def norm(s):
    return (s or "").strip().lower()


def scores(rows):
    """Return per-instance 0/1 lists for exact, relation and entity match."""
    ex, rel, ent = [], [], []
    for d in rows:
        g_e1, g_e2 = norm(d.get("gold_e1")), norm(d.get("gold_e2"))
        p_e1, p_e2 = norm(d.get("pred_e1")), norm(d.get("pred_e2"))
        g_rel = norm(d.get("gold_rel", d.get("gold_relation", d.get("gold", "")))).split("(")[0]
        p_rel = norm(d.get("pred_rel", d.get("pred_relation", d.get("pred", "")))).split("(")[0]

        e1_ok = bool(p_e1) and (g_e1 in p_e1 or p_e1 in g_e1)
        e2_ok = bool(p_e2) and (g_e2 in p_e2 or p_e2 in g_e2)
        r_ok = (g_rel == p_rel)

        rel.append(1 if r_ok else 0)
        ent.append(1 if (e1_ok and e2_ok) else 0)
        ex.append(1 if (r_ok and e1_ok and e2_ok) else 0)
    return {"exact": ex, "relation": rel, "entity": ent}


def bootstrap(a, b, n_boot, seed=42):
    """Paired bootstrap over instances. Returns observed diff, p-value, CI."""
    random.seed(seed)
    n = len(a)
    obs = sum(a) / n - sum(b) / n
    diffs = []
    idx = range(n)
    for _ in range(n_boot):
        sample = [random.choice(idx) for _ in range(n)]
        da = sum(a[i] for i in sample) / n
        db = sum(b[i] for i in sample) / n
        diffs.append(da - db)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    centered = [d - obs for d in diffs]
    p = sum(1 for d in centered if abs(d) >= abs(obs)) / n_boot
    return obs, p, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Predictions file for model A")
    ap.add_argument("--b", required=True, help="Predictions file for model B")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ra, rb = load(args.a), load(args.b)
    if len(ra) != len(rb):
        raise SystemExit(f"Length mismatch: {len(ra)} vs {len(rb)}")

    sa, sb = scores(ra), scores(rb)

    print(f"Model A: {args.a}")
    print(f"Model B: {args.b}")
    print(f"Instances: {len(ra)}, resamples: {args.n_boot}\n")
    print(f"{'Metric':<12} {'A':<8} {'B':<8} {'diff (pp)':<11} {'p':<8} {'95% CI (pp)':<20} {'sig'}")
    print("-" * 78)

    for m in ["exact", "relation", "entity"]:
        a, b = sa[m], sb[m]
        obs, p, lo, hi = bootstrap(a, b, args.n_boot, args.seed)
        va, vb = sum(a) / len(a), sum(b) / len(b)
        sig = "yes" if p < 0.05 else "no"
        print(f"{m:<12} {va:<8.4f} {vb:<8.4f} {obs*100:<+11.2f} {p:<8.4f} "
              f"[{lo*100:+.2f}, {hi*100:+.2f}]{'':<6} {sig}")


if __name__ == "__main__":
    main()
