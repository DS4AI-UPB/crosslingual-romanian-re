#!/usr/bin/env python3
"""
Run Gemma 4 31B for Relation Extraction on SemEval-2010 Task 8.

Experiments:
  A) Relation Classification (entities marked with <e1>, <e2>)
  B) End-to-End RE (no entity markers, model finds entities + relation)

Each experiment runs in zero-shot and few-shot (1, 3, 5 examples).

Usage:
    python run_inference.py \
        --input data/test_ro.jsonl \
        --output results/exp_A_ro_zeroshot.jsonl \
        --experiment A \
        --mode zero-shot \
        --lang ro
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ──────────────────────────────────────────────
# 1. SemEval relation definitions
# ──────────────────────────────────────────────

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

RELATION_DESCRIPTIONS = {
    "Cause-Effect": "One entity causes or leads to the other",
    "Instrument-Agency": "One entity uses the other as an instrument or tool",
    "Product-Producer": "One entity produces or creates the other",
    "Content-Container": "One entity is contained within the other",
    "Entity-Origin": "One entity originates from the other",
    "Entity-Destination": "One entity moves toward or is destined for the other",
    "Component-Whole": "One entity is a component/part of the other",
    "Member-Collection": "One entity is a member of the other (a collection/group)",
    "Message-Topic": "One entity is a message/communication about the other",
    "Other": "None of the above relations apply",
}


# ──────────────────────────────────────────────
# 2. Prompt templates
# ──────────────────────────────────────────────

PROMPT_CLASSIFICATION_ZERO = """Given the sentence below, classify the semantic relation between the entities marked with <e1> and <e2>.

Possible relations:
{relations}

Sentence: {sentence}

Respond with ONLY the relation and direction in this exact format: Relation(e1,e2) or Relation(e2,e1)
If none of the relations apply, respond with: Other"""

PROMPT_CLASSIFICATION_FEW = """Given a sentence with two marked entities <e1> and <e2>, classify their semantic relation.

Possible relations:
{relations}

Here are some examples:
{examples}

Now classify:
Sentence: {sentence}

Respond with ONLY the relation and direction in this exact format: Relation(e1,e2) or Relation(e2,e1)
If none of the relations apply, respond with: Other"""

PROMPT_E2E_ZERO = """From the sentence below, extract the two most relevant entities and identify the semantic relation between them.

Possible relations:
{relations}

Sentence: {sentence}

Respond in this exact JSON format (nothing else):
{{"e1": "first entity", "e2": "second entity", "relation": "Relation(e1,e2)"}}"""

PROMPT_E2E_FEW = """From a sentence, extract two relevant entities and their semantic relation.

Possible relations:
{relations}

Examples:
{examples}

Now extract from:
Sentence: {sentence}

Respond in this exact JSON format (nothing else):
{{"e1": "first entity", "e2": "second entity", "relation": "Relation(e1,e2)"}}"""


def format_relations_list():
    lines = []
    for r in RELATIONS:
        lines.append(f"- {r}: {RELATION_DESCRIPTIONS[r]}")
    return "\n".join(lines)


def format_example_classification(entry, lang="en"):
    sent_key = f"sentence_{lang}" if f"sentence_{lang}" in entry else "sentence_en"
    sent = entry[sent_key]
    rel = entry["relation"]
    return f'Sentence: {sent}\nAnswer: {rel}'


def format_example_e2e(entry, lang="en"):
    sent_key = f"sentence_{lang}" if f"sentence_{lang}" in entry else "sentence_en"
    sent_tagged = entry[sent_key]

    # Extract entities from tags
    e1 = re.search(r"<e1>(.*?)</e1>", sent_tagged)
    e2 = re.search(r"<e2>(.*?)</e2>", sent_tagged)
    e1_text = e1.group(1) if e1 else "?"
    e2_text = e2.group(1) if e2 else "?"

    # Remove tags for e2e
    sent_clean = re.sub(r"</?e[12]>", "", sent_tagged)
    rel = entry["relation"]

    return f'Sentence: {sent_clean}\nAnswer: {{"e1": "{e1_text}", "e2": "{e2_text}", "relation": "{rel}"}}'


# ──────────────────────────────────────────────
# 3. Model loading
# ──────────────────────────────────────────────

def load_model(model_path: str, quantize: str = "4bit"):
    """Load Gemma 4 31B with quantization."""

    print(f"Loading model from {model_path} ({quantize})...")

    if quantize == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantize == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if quantize != "4bit" else "eager",
    )

    print(f"Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'single device'}")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    """Generate a response from the model."""

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.95,
            top_k=64,
            do_sample=True,
            repetition_penalty=1.0,
        )

    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Remove thinking tags if present (Gemma 4 thinking mode)
    response = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", response, flags=re.DOTALL).strip()

    return response


# ──────────────────────────────────────────────
# 4. Evaluation
# ──────────────────────────────────────────────

def parse_classification_response(response: str) -> str:
    """Parse model response for classification experiment."""
    response = response.strip()

    # Try to match Relation(e1,e2) or Relation(e2,e1)
    match = re.search(r"(\w[\w-]+)\((e[12]),\s*(e[12])\)", response)
    if match:
        return f"{match.group(1)}({match.group(2)},{match.group(3)})"

    # Try to match just the relation name
    for rel in RELATIONS:
        if rel.lower() in response.lower():
            return rel

    return response


def parse_e2e_response(response: str) -> dict:
    """Parse model response for end-to-end experiment."""
    # Try JSON parse
    try:
        # Find JSON in response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "e1": data.get("e1", ""),
                "e2": data.get("e2", ""),
                "relation": data.get("relation", ""),
            }
    except json.JSONDecodeError:
        pass

    return {"e1": "", "e2": "", "relation": response}


def evaluate_classification(predictions: list[dict]) -> dict:
    """Compute P, R, F1 for classification experiment."""
    from sklearn.metrics import classification_report, f1_score

    y_true = [p["gold"] for p in predictions]
    y_pred = [p["pred"] for p in predictions]

    # Get unique labels
    labels = sorted(set(y_true + y_pred))

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    return {
        "macro_f1": macro_f1,
        "accuracy": sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true),
        "total": len(y_true),
        "report": report,
    }


def evaluate_e2e(predictions: list[dict]) -> dict:
    """Compute exact and partial match for end-to-end experiment."""
    exact = 0
    partial_rel = 0
    partial_ent = 0

    for p in predictions:
        gold_e1 = p.get("gold_e1", "").lower()
        gold_e2 = p.get("gold_e2", "").lower()
        gold_rel = p.get("gold_rel", "")

        pred_e1 = p.get("pred_e1", "").lower()
        pred_e2 = p.get("pred_e2", "").lower()
        pred_rel = p.get("pred_rel", "")

        # Exact match: both entities + relation correct
        rel_match = gold_rel.split("(")[0] == pred_rel.split("(")[0] if gold_rel and pred_rel else False
        e1_match = gold_e1 in pred_e1 or pred_e1 in gold_e1
        e2_match = gold_e2 in pred_e2 or pred_e2 in gold_e2

        if rel_match and e1_match and e2_match:
            exact += 1
        if rel_match:
            partial_rel += 1
        if e1_match and e2_match:
            partial_ent += 1

    n = len(predictions)
    return {
        "exact_match": exact / n if n else 0,
        "relation_match": partial_rel / n if n else 0,
        "entity_match": partial_ent / n if n else 0,
        "total": n,
    }


# ──────────────────────────────────────────────
# 5. Main pipeline
# ──────────────────────────────────────────────

def load_data(filepath: str) -> list[dict]:
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def select_few_shot_examples(train_data: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Select n diverse examples (one per relation if possible)."""
    random.seed(seed)

    by_relation = {}
    for e in train_data:
        rel = e["relation"].split("(")[0]
        if rel not in by_relation:
            by_relation[rel] = []
        by_relation[rel].append(e)

    selected = []
    # First pass: one per relation
    for rel in RELATIONS:
        if rel in by_relation and len(selected) < n:
            selected.append(random.choice(by_relation[rel]))

    # Fill remaining
    remaining = [e for e in train_data if e not in selected]
    while len(selected) < n:
        selected.append(random.choice(remaining))

    return selected[:n]


def main():
    parser = argparse.ArgumentParser(description="Run Gemma 4 RE inference on SemEval")
    parser.add_argument("--input", required=True, help="Test data (JSONL)")
    parser.add_argument("--train", default=None, help="Train data (JSONL, for few-shot examples)")
    parser.add_argument("--output", required=True, help="Output predictions (JSONL)")
    parser.add_argument("--experiment", choices=["A", "B"], required=True,
                        help="A=classification (entities given), B=end-to-end")
    parser.add_argument("--mode", choices=["zero-shot", "few-shot-1", "few-shot-3", "few-shot-5"],
                        default="zero-shot")
    parser.add_argument("--lang", choices=["en", "ro"], default="en",
                        help="Language of input sentences")
    parser.add_argument("--model-path", default="google/gemma-4-31B-it",
                        help="Model path or HF model ID")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N entries")
    parser.add_argument("--batch-size", type=int, default=1, help="Not used yet (sequential)")
    args = parser.parse_args()

    # ── Load data ──
    print(f"Loading test data from {args.input}...")
    test_data = load_data(args.input)
    if args.limit > 0:
        test_data = test_data[:args.limit]
    print(f"  Loaded {len(test_data)} test entries")

    # ── Load few-shot examples ──
    few_shot_examples = []
    if "few-shot" in args.mode:
        n_shots = int(args.mode.split("-")[-1])
        if not args.train:
            print("ERROR: --train required for few-shot mode")
            sys.exit(1)
        train_data = load_data(args.train)
        few_shot_examples = select_few_shot_examples(train_data, n_shots)
        print(f"  Selected {len(few_shot_examples)} few-shot examples")

    # ── Load model ──
    model, tokenizer = load_model(args.model_path, args.quantize)

    # ── Build prompts and run ──
    relations_text = format_relations_list()
    predictions = []

    print(f"\nRunning experiment {args.experiment} ({args.mode}) in {args.lang}...")
    print(f"Output: {args.output}\n")

    with open(args.output, "w", encoding="utf-8") as f_out:
        for idx, entry in enumerate(test_data):
            sent_key = f"sentence_{args.lang}"
            sentence = entry.get(sent_key, entry.get("sentence_en", ""))

            # ── Build prompt ──
            if args.experiment == "A":
                # Classification: entities are marked
                if few_shot_examples:
                    examples_text = "\n\n".join(
                        format_example_classification(ex, args.lang) for ex in few_shot_examples
                    )
                    prompt = PROMPT_CLASSIFICATION_FEW.format(
                        relations=relations_text, examples=examples_text, sentence=sentence
                    )
                else:
                    prompt = PROMPT_CLASSIFICATION_ZERO.format(
                        relations=relations_text, sentence=sentence
                    )
            else:
                # End-to-end: remove entity markers
                sentence_clean = re.sub(r"</?e[12]>", "", sentence)

                if few_shot_examples:
                    examples_text = "\n\n".join(
                        format_example_e2e(ex, args.lang) for ex in few_shot_examples
                    )
                    prompt = PROMPT_E2E_FEW.format(
                        relations=relations_text, examples=examples_text, sentence=sentence_clean
                    )
                else:
                    prompt = PROMPT_E2E_ZERO.format(
                        relations=relations_text, sentence=sentence_clean
                    )

            # ── Generate ──
            response = generate_response(model, tokenizer, prompt)

            # ── Parse response ──
            if args.experiment == "A":
                pred_rel = parse_classification_response(response)
                result = {
                    "id": entry["id"],
                    "gold": entry["relation"],
                    "pred": pred_rel,
                    "response_raw": response,
                }
            else:
                parsed = parse_e2e_response(response)
                # Get gold entities
                e1_gold = re.search(r"<e1>(.*?)</e1>", entry.get(sent_key, entry["sentence_en"]))
                e2_gold = re.search(r"<e2>(.*?)</e2>", entry.get(sent_key, entry["sentence_en"]))

                result = {
                    "id": entry["id"],
                    "gold_rel": entry["relation"],
                    "gold_e1": e1_gold.group(1) if e1_gold else "",
                    "gold_e2": e2_gold.group(1) if e2_gold else "",
                    "pred_rel": parsed["relation"],
                    "pred_e1": parsed["e1"],
                    "pred_e2": parsed["e2"],
                    "response_raw": response,
                }

            predictions.append(result)
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()

            # Progress
            if (idx + 1) % 100 == 0:
                print(f"  [{idx+1}/{len(test_data)}] processed")

    # ── Evaluate ──
    print(f"\n{'='*60}")
    print(f"RESULTS: Experiment {args.experiment}, {args.mode}, {args.lang}")
    print(f"{'='*60}")

    if args.experiment == "A":
        metrics = evaluate_classification(predictions)
        print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Total:     {metrics['total']}")

        # Per-relation breakdown
        print(f"\n  Per-relation F1:")
        for rel in sorted(metrics["report"].keys()):
            if rel in ("accuracy", "macro avg", "weighted avg"):
                continue
            f1 = metrics["report"][rel]["f1-score"]
            support = metrics["report"][rel]["support"]
            print(f"    {rel:<35s} F1={f1:.3f}  (n={support})")

    else:
        metrics = evaluate_e2e(predictions)
        print(f"  Exact match:    {metrics['exact_match']:.4f}")
        print(f"  Relation match: {metrics['relation_match']:.4f}")
        print(f"  Entity match:   {metrics['entity_match']:.4f}")
        print(f"  Total:          {metrics['total']}")

    # Save metrics
    metrics_path = args.output.replace(".jsonl", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "experiment": args.experiment,
            "mode": args.mode,
            "lang": args.lang,
            "model": args.model_path,
            "metrics": {k: v for k, v in metrics.items() if k != "report"},
        }, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
