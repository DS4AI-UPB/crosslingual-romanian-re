#!/usr/bin/env python3
"""
Evaluate the QLoRA end-to-end adapter (Experiment B) on EN and RO test sets.

Uses the same JSON prompt format and the same exact/relation/entity match
metrics as run_inference.py, so results are directly comparable to the
zero-shot and few-shot Experiment B numbers.
"""

import json
import os
import re

import torch
from unsloth import FastModel


# Paths are set from CLI arguments in main(); defaults assume the repo layout.
LORA_DIR = "models/gemma4-ro-e2e-lora"
TEST_EN = "data/test_en.jsonl"
TEST_RO = "data/test_ro_clean.jsonl"
RESULTS_DIR = "results"

MAX_SEQ_LEN = 640

RELATIONS = [
    "Cause-Effect", "Instrument-Agency", "Product-Producer",
    "Content-Container", "Entity-Origin", "Entity-Destination",
    "Component-Whole", "Member-Collection", "Message-Topic", "Other",
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

PROMPT_E2E_ZERO = """From the sentence below, extract the two most relevant entities and identify the semantic relation between them.

Possible relations:
{relations}

Sentence: {sentence}

Respond in this exact JSON format (nothing else):
{{"e1": "first entity", "e2": "second entity", "relation": "Relation(e1,e2)"}}"""


def format_relations_list():
    return "\n".join(f"- {r}: {RELATION_DESCRIPTIONS[r]}" for r in RELATIONS)


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def get_tagged_sentence(item, lang):
    key = f"sentence_{lang}"
    return item.get(key, item["sentence_en"])


def parse_e2e_response(response):
    """Same parsing as run_inference.py."""
    try:
        m = re.search(r"\{.*\}", response, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return {
                "e1": data.get("e1", ""),
                "e2": data.get("e2", ""),
                "relation": data.get("relation", ""),
            }
    except json.JSONDecodeError:
        pass
    return {"e1": "", "e2": "", "relation": response}


def evaluate(model, tokenizer, test_path, lang, relations_text):
    data = load_jsonl(test_path)
    print(f"\nEvaluating {lang.upper()}: {len(data)} examples")

    exact = partial_rel = partial_ent = 0
    outputs = []

    for i, item in enumerate(data):
        tagged = get_tagged_sentence(item, lang)
        sentence_clean = re.sub(r"</?e[12]>", "", tagged)
        prompt = PROMPT_E2E_ZERO.format(relations=relations_text, sentence=sentence_clean)

        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text=input_text, return_tensors="pt", truncation=True,
                          max_length=MAX_SEQ_LEN).to(model.device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=96,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = out_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        parsed = parse_e2e_response(response)

        # Gold
        e1g = re.search(r"<e1>(.*?)</e1>", tagged)
        e2g = re.search(r"<e2>(.*?)</e2>", tagged)
        gold_e1 = (e1g.group(1) if e1g else "").lower()
        gold_e2 = (e2g.group(1) if e2g else "").lower()
        gold_rel = item["relation"]

        pred_e1 = parsed["e1"].lower()
        pred_e2 = parsed["e2"].lower()
        pred_rel = parsed["relation"]

        rel_match = (gold_rel.split("(")[0] == pred_rel.split("(")[0]) if gold_rel and pred_rel else False
        e1_match = bool(pred_e1) and (gold_e1 in pred_e1 or pred_e1 in gold_e1)
        e2_match = bool(pred_e2) and (gold_e2 in pred_e2 or pred_e2 in gold_e2)

        if rel_match and e1_match and e2_match:
            exact += 1
        if rel_match:
            partial_rel += 1
        if e1_match and e2_match:
            partial_ent += 1

        outputs.append({
            "id": item.get("id"),
            "gold_rel": gold_rel,
            "gold_e1": gold_e1,
            "gold_e2": gold_e2,
            "pred_rel": pred_rel,
            "pred_e1": pred_e1,
            "pred_e2": pred_e2,
            "response_raw": response,
        })

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(data)}] exact={exact/(i+1):.3f} rel={partial_rel/(i+1):.3f}")

    n = len(data)
    metrics = {
        "exact_match": exact / n,
        "relation_match": partial_rel / n,
        "entity_match": partial_ent / n,
        "total": n,
    }
    print(f"\n  Results {lang.upper()}:")
    print(f"    Exact match:    {metrics['exact_match']:.4f}")
    print(f"    Relation match: {metrics['relation_match']:.4f}")
    print(f"    Entity match:   {metrics['entity_match']:.4f}")

    out = {
        "experiment": "B",
        "mode": "qlora",
        "lang": lang,
        "model": LORA_DIR,
        "metrics": metrics,
    }
    with open(os.path.join(RESULTS_DIR, f"B_qlora_{lang}_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(RESULTS_DIR, f"B_qlora_{lang}.jsonl"), "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    return metrics


def main():
    import argparse
    global LORA_DIR, TEST_EN, TEST_RO, RESULTS_DIR
    ap = argparse.ArgumentParser(description="Evaluate the QLoRA end-to-end adapter on the EN and RO test sets.")
    ap.add_argument("--adapter", default=LORA_DIR, help="LoRA adapter path or HF repo id.")
    ap.add_argument("--test-en", default=TEST_EN)
    ap.add_argument("--test-ro", default=TEST_RO)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    args = ap.parse_args()
    LORA_DIR, TEST_EN, TEST_RO, RESULTS_DIR = args.adapter, args.test_en, args.test_ro, args.results_dir

    print("=" * 60)
    print("QLoRA End-to-End Evaluation (Experiment B)")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    relations_text = format_relations_list()

    print(f"\nLoading LoRA adapter: {LORA_DIR}")
    model, tokenizer = FastModel.from_pretrained(
        LORA_DIR,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )
    FastModel.for_inference(model)

    en = evaluate(model, tokenizer, TEST_EN, "en", relations_text)
    ro = evaluate(model, tokenizer, TEST_RO, "ro", relations_text)

    print("\n" + "=" * 60)
    print("SUMMARY (Experiment B, QLoRA)")
    print("=" * 60)
    print(f"  EN: exact={en['exact_match']:.4f} rel={en['relation_match']:.4f} ent={en['entity_match']:.4f}")
    print(f"  RO: exact={ro['exact_match']:.4f} rel={ro['relation_match']:.4f} ent={ro['entity_match']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
