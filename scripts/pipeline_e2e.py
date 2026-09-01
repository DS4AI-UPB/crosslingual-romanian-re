#!/usr/bin/env python3
"""
NER + classifier pipeline baseline for End-to-End RE.

Stage 1: the span detector tags e1 and e2 in the plain sentence.
Stage 2: the detected spans are wrapped in <e1>/<e2> markers and passed to the
existing relation classifier (the fine-tuned encoder).

Output is written in the same schema as the LLM end-to-end predictions
(gold_rel, gold_e1, gold_e2, pred_rel, pred_e1, pred_e2), so significance_e2e.py
and the metric scripts work unchanged.

    python pipeline_e2e.py \
        --test data/test_ro_clean.jsonl --lang ro \
        --detector models/span-detector-ro \
        --classifier models/xlmr-large-re/checkpoint-3968/checkpoint-3968 \
        --output results/B_pipeline_ro.jsonl
"""
import argparse, json, re
import torch
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          AutoModelForSequenceClassification)

LABELS = ["O", "B-E1", "I-E1", "B-E2", "I-E2"]
I2L = {i: l for i, l in enumerate(LABELS)}

# 19 directional relation labels, same order the classifier was trained on
RELATIONS = [
    "Cause-Effect(e1,e2)", "Cause-Effect(e2,e1)",
    "Instrument-Agency(e1,e2)", "Instrument-Agency(e2,e1)",
    "Product-Producer(e1,e2)", "Product-Producer(e2,e1)",
    "Content-Container(e1,e2)", "Content-Container(e2,e1)",
    "Entity-Origin(e1,e2)", "Entity-Origin(e2,e1)",
    "Entity-Destination(e1,e2)", "Entity-Destination(e2,e1)",
    "Component-Whole(e1,e2)", "Component-Whole(e2,e1)",
    "Member-Collection(e1,e2)", "Member-Collection(e2,e1)",
    "Message-Topic(e1,e2)", "Message-Topic(e2,e1)",
    "Other",
]


def strip_markers(marked):
    return re.sub(r"</?e[12]>", "", marked)


def gold_entities(marked):
    e1 = re.search(r"<e1>(.*?)</e1>", marked)
    e2 = re.search(r"<e2>(.*?)</e2>", marked)
    return (e1.group(1) if e1 else ""), (e2.group(1) if e2 else "")


@torch.no_grad()
def detect_spans(text, tok, model, max_len=192):
    enc = tok(text, truncation=True, max_length=max_len,
              return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(model.device) for k, v in enc.items()}
    pred = model(**enc).logits.argmax(-1)[0].tolist()

    spans = {"E1": [], "E2": []}
    for (a, b), p in zip(offsets, pred):
        if a == b:
            continue
        lab = I2L[p]
        if lab == "O":
            continue
        ent = lab[-2:]
        spans[ent].append((a, b))

    def merge(ranges):
        if not ranges:
            return None
        ranges.sort()
        return (ranges[0][0], ranges[-1][1])

    return merge(spans["E1"]), merge(spans["E2"]), text


def wrap_markers(text, e1_range, e2_range):
    """Insert <e1>/<e2> markers around the detected char ranges."""
    marks = []
    if e1_range:
        marks.append((e1_range[0], "<e1>")); marks.append((e1_range[1], "</e1>"))
    if e2_range:
        marks.append((e2_range[0], "<e2>")); marks.append((e2_range[1], "</e2>"))
    for pos, tag in sorted(marks, key=lambda x: -x[0]):
        text = text[:pos] + tag + text[pos:]
    return text


def convert_markers(text):
    text = text.replace("<e1>", "[E1] ").replace("</e1>", " [/E1]")
    return text.replace("<e2>", "[E2] ").replace("</e2>", " [/E2]")


@torch.no_grad()
def classify(marked_text, tok, model, max_len=192):
    inp = convert_markers(marked_text)
    enc = tok(inp, truncation=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    pid = model(**enc).logits.argmax(-1).item()
    return RELATIONS[pid] if pid < len(RELATIONS) else "Other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--lang", choices=["ro", "en"], default="ro")
    ap.add_argument("--detector", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-len", type=int, default=192)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtok = AutoTokenizer.from_pretrained(args.detector)
    dmodel = AutoModelForTokenClassification.from_pretrained(args.detector).to(dev).eval()
    ctok = AutoTokenizer.from_pretrained(args.classifier)
    cmodel = AutoModelForSequenceClassification.from_pretrained(args.classifier).to(dev).eval()

    key = "sentence_ro" if args.lang == "ro" else "sentence_en"
    out = open(args.output, "w", encoding="utf-8")
    n = 0
    for line in open(args.test, encoding="utf-8"):
        d = json.loads(line)
        marked = d[key]
        plain = strip_markers(marked)
        g_e1, g_e2 = gold_entities(marked)

        e1r, e2r, text = detect_spans(plain, dtok, dmodel, args.max_len)
        p_e1 = text[e1r[0]:e1r[1]] if e1r else ""
        p_e2 = text[e2r[0]:e2r[1]] if e2r else ""
        wrapped = wrap_markers(text, e1r, e2r)
        rel = classify(wrapped, ctok, cmodel, args.max_len) if (e1r or e2r) else "Other"

        out.write(json.dumps({
            "id": d.get("id"),
            "gold_rel": d["relation"],
            "gold_e1": g_e1, "gold_e2": g_e2,
            "pred_rel": rel,
            "pred_e1": p_e1, "pred_e2": p_e2,
        }, ensure_ascii=False) + "\n")
        n += 1
        if n % 200 == 0:
            print(f"  [{n}] processed")
    out.close()
    print(f"Wrote {n} predictions to {args.output}")


if __name__ == "__main__":
    main()
