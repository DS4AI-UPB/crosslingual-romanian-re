#!/usr/bin/env python3
"""
Re-score End-to-End predictions on a cleaned subset of the Romanian test set.

The manual inspection found that a share of the Romanian translations leave the
entity inside the markers in English. Those cases penalize end-to-end scores
even when the model is right, because the gold span is taken from the marker.
This script flags them automatically by comparing the Romanian and English
entity strings recorded in the dataset's `validation` field, drops them, and
reports the metrics on the remaining instances.

    python clean_subset_eval.py \
        --test data/test_ro_clean.jsonl \
        --pred results/B_qlora_ro.jsonl
"""
import argparse, json


def norm(s):
    return (s or "").strip().lower()


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def is_untranslated(v):
    """True when either entity is identical in Romanian and English."""
    if not v:
        return False
    e1_en, e1_ro = norm(v.get("e1_en")), norm(v.get("e1_ro"))
    e2_en, e2_ro = norm(v.get("e2_en")), norm(v.get("e2_ro"))
    same1 = bool(e1_en) and e1_en == e1_ro
    same2 = bool(e2_en) and e2_en == e2_ro
    return same1 or same2


def score(rows):
    ex = rel = ent = 0
    for d in rows:
        g_e1, g_e2 = norm(d.get("gold_e1")), norm(d.get("gold_e2"))
        p_e1, p_e2 = norm(d.get("pred_e1")), norm(d.get("pred_e2"))
        g_rel = norm(d.get("gold_rel", d.get("gold_relation", d.get("gold", "")))).split("(")[0]
        p_rel = norm(d.get("pred_rel", d.get("pred_relation", d.get("pred", "")))).split("(")[0]
        e1_ok = bool(p_e1) and (g_e1 in p_e1 or p_e1 in g_e1)
        e2_ok = bool(p_e2) and (g_e2 in p_e2 or p_e2 in g_e2)
        r_ok = g_rel == p_rel
        if r_ok:
            rel += 1
        if e1_ok and e2_ok:
            ent += 1
        if r_ok and e1_ok and e2_ok:
            ex += 1
    n = len(rows) or 1
    return ex / n, rel / n, ent / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="Romanian test JSONL (with validation field)")
    ap.add_argument("--pred", required=True, help="Predictions JSONL, same order as test")
    args = ap.parse_args()

    test, pred = load(args.test), load(args.pred)
    if len(test) != len(pred):
        raise SystemExit(f"Length mismatch: test {len(test)} vs pred {len(pred)}")

    keep, drop = [], []
    for t, p in zip(test, pred):
        (drop if is_untranslated(t.get("validation")) else keep).append(p)

    print(f"Total instances : {len(pred)}")
    print(f"Flagged (entity left untranslated): {len(drop)} ({100*len(drop)/len(pred):.1f}%)")
    print(f"Clean subset    : {len(keep)}\n")

    print(f"{'Subset':<16} {'n':<7} {'exact':<9} {'relation':<10} {'entity':<9}")
    print("-" * 55)
    for name, rows in [("full", pred), ("clean", keep), ("flagged", drop)]:
        if not rows:
            continue
        e, r, n_ = score(rows)
        print(f"{name:<16} {len(rows):<7} {e:<9.4f} {r:<10.4f} {n_:<9.4f}")

    if keep and drop:
        ef, rf, nf = score(pred)
        ec, rc, nc = score(keep)
        print(f"\nDelta clean - full (pp): exact {100*(ec-ef):+.2f}, "
              f"relation {100*(rc-rf):+.2f}, entity {100*(nc-nf):+.2f}")


if __name__ == "__main__":
    main()
