#!/usr/bin/env python3
"""
End-to-End Relation Extraction inference with the QLoRA Gemma 4 adapter.

Loads the fine-tuned end-to-end adapter (from the Hugging Face Hub or a local
folder) with Unsloth, the same stack used for training and evaluation in the
paper. The input is a plain sentence with no entity markers: the model extracts
both entities and the relation between them, and returns them as JSON.

This is the "End-to-End RE" task.

Example:
    python infer_e2e.py \
        --sentence "The cup was filled with coffee." \
        --adapter DS4AI-UPB/gemma4-ro-e2e-lora

    python infer_e2e.py \
        --file my_sentences.txt \
        --adapter DS4AI-UPB/gemma4-ro-e2e-lora

Requirements:
    pip install unsloth
"""

import argparse
import json
import re
import sys

import torch
from unsloth import FastModel

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

PROMPT_E2E = """From the sentence below, extract the two most relevant entities and identify the semantic relation between them.

Possible relations:
{relations}

Sentence: {sentence}

Respond in this exact JSON format (nothing else):
{{"e1": "first entity", "e2": "second entity", "relation": "Relation(e1,e2)"}}"""


def relations_block():
    return "\n".join(f"- {r}: {RELATION_DESCRIPTIONS[r]}" for r in RELATIONS)


def build_prompt(sentence):
    sentence = re.sub(r"</?e[12]>", "", sentence)
    return PROMPT_E2E.format(relations=relations_block(), sentence=sentence)


def parse_response(response):
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
    return {"e1": "", "e2": "", "relation": response.strip()}


def load_model(adapter):
    print(f"Loading model + adapter: {adapter}", file=sys.stderr)
    model, tokenizer = FastModel.from_pretrained(
        adapter,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )
    FastModel.for_inference(model)
    return model, tokenizer


@torch.no_grad()
def predict(model, tokenizer, sentence):
    prompt = build_prompt(sentence)
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=96,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_response(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sentence", help="A single plain sentence (no entity markers).")
    src.add_argument("--file", help="A text file with one sentence per line.")
    parser.add_argument("--adapter", default="DS4AI-UPB/gemma4-ro-e2e-lora",
                        help="HF repo id or local path of the end-to-end adapter.")
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter)

    if args.sentence:
        sentences = [args.sentence]
    else:
        with open(args.file, "r", encoding="utf-8") as f:
            sentences = [ln.strip() for ln in f if ln.strip()]

    for sentence in sentences:
        result = predict(model, tokenizer, sentence)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()