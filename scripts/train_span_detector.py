#!/usr/bin/env python3
"""
Train a span detector for End-to-End RE.

Learns to tag the two entity spans e1 and e2 in a sentence that has no markers,
as a token-classification (BIO) problem over five labels:
O, B-E1, I-E1, B-E2, I-E2. The gold spans come from the <e1>/<e2> markers in
the training data. At inference the predicted spans are wrapped back in
<e1>/<e2> markers and fed to the existing relation classifier, giving a
NER + classifier pipeline baseline for the end-to-end task.

    python train_span_detector.py \
        --train data/train_ro_clean.jsonl \
        --lang ro --output-dir models/span-detector-ro
"""
import argparse, json, os, re
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          TrainingArguments, Trainer, DataCollatorForTokenClassification)

LABELS = ["O", "B-E1", "I-E1", "B-E2", "I-E2"]
L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for l, i in L2I.items()}

MARK = re.compile(r"<e1>(.*?)</e1>|<e2>(.*?)</e2>")


def strip_and_spans(marked):
    """Return (plain_text, [(start,end,'E1'|'E2'), ...]) with char offsets."""
    spans = []; plain_text = ""; cur = None; start = 0
    for tok in re.split(r"(</?e[12]>)", marked):
        if tok == "<e1>": cur = "E1"; start = len(plain_text)
        elif tok == "<e2>": cur = "E2"; start = len(plain_text)
        elif tok in ("</e1>", "</e2>"):
            if cur is not None: spans.append((start, len(plain_text), cur))
            cur = None
        else: plain_text += tok
    return plain_text, spans


def char_to_labels(text, spans, tokenizer, max_len):
    enc = tokenizer(text, truncation=True, max_length=max_len,
                    return_offsets_mapping=True)
    labels = []
    for (a, b) in enc["offset_mapping"]:
        if a == b:  # special token
            labels.append(-100); continue
        tag = "O"
        for (s, e, ent) in spans:
            if a >= s and b <= e:
                tag = ("B-" if a == s else "I-") + ent
                break
        labels.append(L2I[tag])
    enc.pop("offset_mapping")
    enc["labels"] = labels
    return enc


class SpanDataset(Dataset):
    def __init__(self, path, lang, tokenizer, max_len):
        self.items = []
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            marked = d["sentence_ro"] if lang == "ro" else d["sentence_en"]
            text, spans = strip_and_spans(marked)
            self.items.append(char_to_labels(text, spans, tokenizer, max_len))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--lang", choices=["ro", "en"], default="ro")
    ap.add_argument("--base-model", default="xlm-roberta-large")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    full = SpanDataset(args.train, args.lang, tok, args.max_len)

    n_val = int(len(full) * args.val_frac)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(full), generator=g).tolist()
    val_idx = set(perm[:n_val])
    train_items = [full.items[i] for i in perm[n_val:]]
    val_items = [full.items[i] for i in perm[:n_val]]

    class L(Dataset):
        def __init__(s, it): s.it = it
        def __len__(s): return len(s.it)
        def __getitem__(s, i): return s.it[i]

    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model, num_labels=len(LABELS), id2label=I2L, label2id=L2I)

    def metrics(p):
        preds = np.argmax(p.predictions, axis=2)
        labels = p.label_ids
        tp = fp = fn = 0
        for pr, la in zip(preds, labels):
            for pi, li in zip(pr, la):
                if li == -100: continue
                if li != L2I["O"] and pi == li: tp += 1
                elif pi != L2I["O"] and pi != li: fp += 1
                elif li != L2I["O"] and pi != li: fn += 1
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        return {"span_f1": f1, "span_precision": prec, "span_recall": rec}

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="span_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=args.seed,
        save_total_limit=1,
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=L(train_items), eval_dataset=L(val_items),
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=metrics,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved span detector to {args.output_dir}")


if __name__ == "__main__":
    main()
