#!/usr/bin/env python3
"""
Parametrized encoder baseline for SemEval-2010 Task 8.

Supports both multilingual (XLM-R) and monolingual Romanian models.
For monolingual models, trains on RO only and evaluates on RO only.

Usage:
    python scripts/train_encoder_baseline.py --model xlm-roberta-base --mode multilingual --tag xlmr-base
    python scripts/train_encoder_baseline.py --model dumitrescustefan/bert-base-romanian-cased-v1 --mode ro-only --tag bert-ro
    python scripts/train_encoder_baseline.py --model readerbench/RoBERT-large --mode ro-only --tag robert-large
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, accuracy_score, classification_report


# Data, results and model directories are set from CLI arguments in main().
# These module-level names are assigned there so the helper functions below
# can use them without threading the paths through every call.
DATA_DIR = "data"
RESULTS_DIR = "results"
MODELS_DIR = "models"
TRAIN_EN = TRAIN_RO = TEST_EN = TEST_RO = None


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
LABEL2ID = {lbl: i for i, lbl in enumerate(RELATIONS)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}
NUM_LABELS = len(RELATIONS)

COARSE_RELATIONS = [
    "Cause-Effect", "Instrument-Agency", "Product-Producer",
    "Content-Container", "Entity-Origin", "Entity-Destination",
    "Component-Whole", "Member-Collection", "Message-Topic", "Other",
]


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def get_sentence(item, lang):
    if lang == "ro" and "sentence_ro" in item:
        return item["sentence_ro"]
    return item["sentence_en"]


def convert_markers(text):
    text = text.replace("<e1>", "[E1] ").replace("</e1>", " [/E1]")
    text = text.replace("<e2>", "[E2] ").replace("</e2>", " [/E2]")
    return text


def coarse(label):
    if "(" in label:
        return label.split("(")[0]
    return label


class REDataset(Dataset):
    def __init__(self, items, lang, tokenizer, max_len):
        self.items = items
        self.lang = lang
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        sentence = get_sentence(item, self.lang)
        text = convert_markers(sentence)
        label = LABEL2ID.get(item["relation"], LABEL2ID["Other"])
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len, padding=False)
        enc["labels"] = label
        return enc


class MixedREDataset(Dataset):
    def __init__(self, items, tokenizer, max_len):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        lang = item.get("_lang", "en")
        sentence = get_sentence(item, lang)
        text = convert_markers(sentence)
        label = LABEL2ID.get(item["relation"], LABEL2ID["Other"])
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len, padding=False)
        enc["labels"] = label
        return enc


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    pred_labels = [ID2LABEL[p] for p in preds]
    gold_labels = [ID2LABEL[l] for l in labels]
    cp = [coarse(p) for p in pred_labels]
    cg = [coarse(g) for g in gold_labels]
    return {
        "macro_f1": f1_score(cg, cp, labels=COARSE_RELATIONS, average="macro", zero_division=0),
        "accuracy": accuracy_score(cg, cp),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name or path")
    parser.add_argument("--mode", choices=["multilingual", "ro-only"], required=True)
    parser.add_argument("--tag", required=True, help="Short tag for result files (e.g. xlmr-base)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction of train held out for checkpoint selection")
    parser.add_argument("--data-dir", default="data",
                        help="Directory with train/test JSONL files.")
    parser.add_argument("--results-dir", default="results",
                        help="Directory where predictions and metrics are written.")
    parser.add_argument("--models-dir", default="models",
                        help="Directory where the fine-tuned encoder is saved.")
    args = parser.parse_args()

    # Resolve the file paths used by the helper functions above.
    global DATA_DIR, RESULTS_DIR, MODELS_DIR, TRAIN_EN, TRAIN_RO, TEST_EN, TEST_RO
    DATA_DIR = args.data_dir
    RESULTS_DIR = args.results_dir
    MODELS_DIR = args.models_dir
    TRAIN_EN = os.path.join(DATA_DIR, "train_en.jsonl")
    TRAIN_RO = os.path.join(DATA_DIR, "train_ro_clean.jsonl")
    TEST_EN = os.path.join(DATA_DIR, "test_en.jsonl")
    TEST_RO = os.path.join(DATA_DIR, "test_ro_clean.jsonl")

    print("=" * 60)
    print(f"Encoder baseline: {args.model}")
    print(f"Mode: {args.mode}, Tag: {args.tag}")
    print("=" * 60)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    special_tokens = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.resize_token_embeddings(len(tokenizer))

    # Data
    en_train = load_jsonl(TRAIN_EN)
    ro_train = load_jsonl(TRAIN_RO)
    en_test = load_jsonl(TEST_EN)
    ro_test = load_jsonl(TEST_RO)

    val_frac = args.val_frac

    if args.mode == "multilingual":
        # Train on combined EN+RO, test on both
        mixed = []
        for it in en_train:
            new = dict(it); new["_lang"] = "en"; mixed.append(new)
        for it in ro_train:
            new = dict(it); new["_lang"] = "ro"; mixed.append(new)
        random.shuffle(mixed)

        # Hold out a validation split from train for checkpoint selection.
        n_val = int(len(mixed) * val_frac)
        val_items = mixed[:n_val]
        train_items = mixed[n_val:]

        train_dataset = MixedREDataset(train_items, tokenizer, args.max_len)
        val_dataset = MixedREDataset(val_items, tokenizer, args.max_len)
        en_test_ds = REDataset(en_test, "en", tokenizer, args.max_len)
        ro_test_ds = REDataset(ro_test, "ro", tokenizer, args.max_len)
        eval_targets = [("en", en_test_ds, en_test), ("ro", ro_test_ds, ro_test)]
        eval_during_training = val_dataset
        print(f"Train: {len(train_items)} (held out {len(val_items)} for validation) "
              f"from {len(en_train)} EN + {len(ro_train)} RO")
    else:
        # RO-only training and evaluation
        random.shuffle(ro_train)

        n_val = int(len(ro_train) * val_frac)
        val_items = ro_train[:n_val]
        train_items = ro_train[n_val:]

        train_dataset = REDataset(train_items, "ro", tokenizer, args.max_len)
        val_dataset = REDataset(val_items, "ro", tokenizer, args.max_len)
        ro_test_ds = REDataset(ro_test, "ro", tokenizer, args.max_len)
        eval_targets = [("ro", ro_test_ds, ro_test)]
        eval_during_training = val_dataset
        print(f"Train: {len(train_items)} (held out {len(val_items)} for validation) "
              f"from {len(ro_train)} RO")

    print(f"Test: {len(en_test)} EN, {len(ro_test)} RO")

    output_dir = os.path.join(MODELS_DIR, f"baseline-{args.tag}")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_during_training,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print("\nTraining...")
    trainer.train()
    print("\nBest checkpoint selected on the held-out validation split "
          f"(macro-F1 = {trainer.state.best_metric:.4f}).")

    # Final eval on the test set (never used for checkpoint selection)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for lang_name, test_ds, test_items in eval_targets:
        print(f"\nEvaluating {lang_name.upper()}: {len(test_items)} examples")
        preds_out = trainer.predict(test_ds)
        pred_ids = np.argmax(preds_out.predictions, axis=-1)

        pred_labels = [ID2LABEL[p] for p in pred_ids]
        gold_labels = [item["relation"] for item in test_items]
        cp = [coarse(p) for p in pred_labels]
        cg = [coarse(g) for g in gold_labels]

        macro_f1 = f1_score(cg, cp, labels=COARSE_RELATIONS, average="macro", zero_division=0)
        acc = accuracy_score(cg, cp)
        print(f"  Macro-F1: {macro_f1:.4f}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"\n{classification_report(cg, cp, labels=COARSE_RELATIONS, zero_division=0)}")

        metrics = {
            "experiment": "A",
            "mode": args.tag,
            "lang": lang_name,
            "model": args.model,
            "metrics": {
                "macro_f1": float(macro_f1),
                "accuracy": float(acc),
                "total": len(test_items),
            }
        }
        with open(os.path.join(RESULTS_DIR, f"A_{args.tag}_{lang_name}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        out_path = os.path.join(RESULTS_DIR, f"A_{args.tag}_{lang_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for item, pred, gold in zip(test_items, pred_labels, gold_labels):
                f.write(json.dumps({
                    "id": item.get("id"),
                    "sentence": get_sentence(item, lang_name),
                    "gold": gold,
                    "predicted": pred,
                }, ensure_ascii=False) + "\n")

    print("\nDone.")


if __name__ == "__main__":
    main()
