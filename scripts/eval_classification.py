#!/usr/bin/env python3
"""
Evaluate QLoRA fine-tuned Gemma 4 31B on SemEval-2010 Task 8.

Runs Experiment A (relation classification) on EN and RO test sets.
Outputs metrics JSON files compatible with existing results format.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from collections import Counter

import torch
from sklearn.metrics import f1_score, accuracy_score
from unsloth import FastModel


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

# Paths are set from CLI arguments in main(); defaults assume the repo layout.
BASE_MODEL = "google/gemma-4-31b-it"
LORA_DIR = "models/gemma4-ro-re-lora"
TEST_EN = "data/test_en.jsonl"
TEST_RO = "data/test_ro_clean.jsonl"
RESULTS_DIR = "results"

MAX_SEQ_LEN = 512

RELATIONS = [
    "Cause-Effect",
    "Instrument-Agency",
    "Product-Producer",
    "Content-Container",
    "Entity-Origin",
    "Entity-Destination",
    "Component-Whole",
    "Member-Collection",
    "Message-Topic",
    "Other",
]

# Regex to parse predictions like "Cause-Effect(e1,e2)"
RELATION_PATTERN = re.compile(
    r"(" + "|".join(re.escape(r) for r in RELATIONS) + r")"
    r"(?:\((e[12]),\s*(e[12])\))?"
)


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_sentence(item, lang):
    if lang == "ro" and "sentence_ro" in item:
        return item["sentence_ro"]
    return item["sentence_en"]


def parse_prediction(text):
    """Extract relation from model output."""
    text = text.strip().split("\n")[0].strip()
    match = RELATION_PATTERN.search(text)
    if match:
        rel = match.group(1)
        if match.group(2) and match.group(3):
            return f"{rel}({match.group(2)},{match.group(3)})"
        return rel
    return "Other"


def normalize_relation(rel_str):
    """Normalize relation string for comparison (ignore direction)."""
    match = RELATION_PATTERN.search(rel_str)
    if match:
        return match.group(1)
    return "Other"


def evaluate(model, tokenizer, test_path, lang, results_dir):
    """Run evaluation on a test set."""
    data = load_jsonl(test_path)
    print(f"\nEvaluating {lang.upper()}: {len(data)} examples")

    golds = []
    preds = []
    outputs = []
    errors = 0

    for i, item in enumerate(data):
        sentence = get_sentence(item, lang)
        gold = item["relation"]

        prompt_text = (
            f"Classify the semantic relation between <e1> and <e2> in this sentence.\n\n"
            f"Possible relations: {', '.join(RELATIONS)}\n\n"
            f"Sentence: {sentence}\n\n"
            f"Respond with ONLY the relation and direction, e.g. Cause-Effect(e1,e2)"
        )

        messages = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(text=input_text, return_tensors="pt", truncation=True,
                          max_length=MAX_SEQ_LEN).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode only new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred_rel = parse_prediction(pred_text)

        golds.append(normalize_relation(gold))
        preds.append(normalize_relation(pred_rel))

        outputs.append({
            "id": item.get("id", i),
            "sentence": sentence,
            "gold": gold,
            "predicted_raw": pred_text.strip(),
            "predicted": pred_rel,
        })

        if (i + 1) % 200 == 0:
            current_f1 = f1_score(golds, preds, labels=RELATIONS, average="macro", zero_division=0)
            print(f"  [{i+1}/{len(data)}] running macro-F1: {current_f1:.4f}")

    # Compute metrics
    macro_f1 = f1_score(golds, preds, labels=RELATIONS, average="macro", zero_division=0)
    acc = accuracy_score(golds, preds)

    print(f"\n  Results {lang.upper()}:")
    print(f"    Macro-F1:  {macro_f1:.4f}")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Total:     {len(data)}")

    # Per-class breakdown
    from sklearn.metrics import classification_report
    print(f"\n{classification_report(golds, preds, labels=RELATIONS, zero_division=0)}")

    # Save results (same format as existing experiments)
    metrics = {
        "experiment": "A",
        "mode": "qlora",
        "lang": lang,
        "model": LORA_DIR,
        "metrics": {
            "macro_f1": macro_f1,
            "accuracy": acc,
            "total": len(data),
        }
    }

    metrics_path = os.path.join(results_dir, f"A_qlora_{lang}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics: {metrics_path}")

    output_path = os.path.join(results_dir, f"A_qlora_{lang}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"  Saved predictions: {output_path}")

    return macro_f1, acc


def main():
    import argparse
    global BASE_MODEL, LORA_DIR, TEST_EN, TEST_RO, RESULTS_DIR
    ap = argparse.ArgumentParser(description="Evaluate the QLoRA classification adapter on the EN and RO test sets.")
    ap.add_argument("--adapter", default=LORA_DIR, help="LoRA adapter path or HF repo id.")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--test-en", default=TEST_EN)
    ap.add_argument("--test-ro", default=TEST_RO)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    args = ap.parse_args()
    BASE_MODEL, LORA_DIR = args.base_model, args.adapter
    TEST_EN, TEST_RO, RESULTS_DIR = args.test_en, args.test_ro, args.results_dir

    print("=" * 60)
    print("QLoRA Evaluation: Gemma 4 31B + LoRA for RE")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load base model + LoRA adapter
    print(f"\nLoading base model: {BASE_MODEL}")
    print(f"Loading LoRA adapter: {LORA_DIR}")

    model, tokenizer = FastModel.from_pretrained(
        LORA_DIR,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )

    FastModel.for_inference(model)

    # Evaluate EN
    en_f1, en_acc = evaluate(model, tokenizer, TEST_EN, "en", RESULTS_DIR)

    # Evaluate RO
    ro_f1, ro_acc = evaluate(model, tokenizer, TEST_RO, "ro", RESULTS_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  EN: macro-F1 = {en_f1:.4f}, accuracy = {en_acc:.4f}")
    print(f"  RO: macro-F1 = {ro_f1:.4f}, accuracy = {ro_acc:.4f}")
    print(f"  Gap: {(en_f1 - ro_f1)*100:.1f}pp")
    print("=" * 60)


if __name__ == "__main__":
    main()
