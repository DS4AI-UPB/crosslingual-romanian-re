#!/usr/bin/env python3
"""
Per-relation F1-Score from classification prediction files.

Collapses the directional labels to the ten coarse SemEval relations and
reports precision, recall and F1 per relation, plus the macro average.
Pass several files to get a side-by-side comparison.

    python per_relation.py results/A_qlora_ro.jsonl results_qwen/A_qlora_ro.jsonl
"""
import argparse, json, collections, os

RELATIONS = ["Cause-Effect", "Instrument-Agency", "Product-Producer",
             "Content-Container", "Entity-Origin", "Entity-Destination",
             "Component-Whole", "Member-Collection", "Message-Topic", "Other"]


def coarse(lbl):
    return (lbl or "").split("(")[0].strip()


def per_rel_f1(path):
    tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter()
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        g = coarse(d.get("gold", d.get("gold_rel", "")))
        p = coarse(d.get("predicted", d.get("pred", d.get("pred_rel", ""))))
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    out = {}
    for r in RELATIONS:
        prec = tp[r] / (tp[r] + fp[r]) if (tp[r] + fp[r]) else 0.0
        rec = tp[r] / (tp[r] + fn[r]) if (tp[r] + fn[r]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[r] = (prec, rec, f1, tp[r] + fn[r])
    out["_macro"] = sum(v[2] for k, v in out.items() if k in RELATIONS) / len(RELATIONS)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--latex", action="store_true", help="Also print a LaTeX table body")
    args = ap.parse_args()

    res = {f: per_rel_f1(f) for f in args.files}
    names = [os.path.basename(f).replace(".jsonl", "") for f in args.files]

    header = f"{'Relation':<20} {'n':<6}" + "".join(f"{n[:16]:>18}" for n in names)
    print(header)
    print("-" * len(header))
    for r in RELATIONS:
        n = res[args.files[0]][r][3]
        row = f"{r:<20} {n:<6}"
        for f in args.files:
            row += f"{res[f][r][2]:>18.3f}"
        print(row)
    print("-" * len(header))
    row = f"{'macro F1':<20} {'':<6}"
    for f in args.files:
        row += f"{res[f]['_macro']:>18.3f}"
    print(row)

    if args.latex:
        print("\n% LaTeX table body")
        for r in RELATIONS:
            cells = " & ".join(f"{res[f][r][2]:.2f}" for f in args.files)
            print(f"{r:<20} & {cells} \\\\")


if __name__ == "__main__":
    main()
