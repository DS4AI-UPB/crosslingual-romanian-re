#!/usr/bin/env python3
"""
Relation Classification inference with the QLoRA Gemma 4 adapter.

Loads the fine-tuned adapter from the Hugging Face Hub (or a local folder)
on top of the base Gemma 4 31B model, and predicts the directional relation
between two marked entities <e1> and <e2> in a sentence.

This is the "Relation Classification" task from the paper: the entity tags are
already present in the input, and the model picks one of the ten relations.

Example:
    python infer_classification.py \
        --sentence "The <e1>cup</e1> was filled with <e2>coffee</e2>." \
        --adapter DS4AI-UPB/gemma4-ro-re-lora

    python infer_classification.py \
        --file my_sentences.txt \
        --adapter DS4AI-UPB/gemma4-ro-re-lora

Requirements:
    pip install torch transformers peft bitsandbytes accelerate
"""

import argparse
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "google/gemma-4-31b-it"

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

# Matches predictions such as "Cause-Effect(e1,e2)" and captures the direction.
RELATION_PATTERN = re.compile(
    r"(" + "|".join(re.escape(r) for r in RELATIONS) + r")"
    r"(?:\((e[12]),\s*(e[12])\))?"
)


def build_prompt(sentence):
    """The exact prompt used for training and evaluation in the paper."""
    return (
        f"Classify the semantic relation between <e1> and <e2> in this sentence.\n\n"
        f"Possible relations: {', '.join(RELATIONS)}\n\n"
        f"Sentence: {sentence}\n\n"
        f"Respond with ONLY the relation and direction, e.g. Cause-Effect(e1,e2)"
    )


def parse_prediction(text):
    """Extract a clean 'Relation(eX,eY)' string from the raw generation."""
    text = text.strip().split("\n")[0].strip()
    match = RELATION_PATTERN.search(text)
    if match:
        rel = match.group(1)
        if match.group(2) and match.group(3):
            return f"{rel}({match.group(2)},{match.group(3)})"
        return rel
    return "Other"


def load_model(adapter, base_model):
    """Load the base model in 4-bit and apply the LoRA adapter."""
    print(f"Loading base model: {base_model}", file=sys.stderr)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Applying adapter: {adapter}", file=sys.stderr)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    return model, tokenizer


@torch.no_grad()
def predict(model, tokenizer, sentence):
    """Predict the relation for a single sentence."""
    prompt = build_prompt(sentence)
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_prediction(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sentence", help="A single sentence with <e1> and <e2> markers.")
    src.add_argument("--file", help="A text file with one marked sentence per line.")
    parser.add_argument("--adapter", default="DS4AI-UPB/gemma4-ro-re-lora",
                        help="HF repo id or local path of the LoRA adapter.")
    parser.add_argument("--base-model", default=BASE_MODEL,
                        help="Base model id (default: google/gemma-4-31b-it).")
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter, args.base_model)

    if args.sentence:
        sentences = [args.sentence]
    else:
        with open(args.file, "r", encoding="utf-8") as f:
            sentences = [ln.strip() for ln in f if ln.strip()]

    for sentence in sentences:
        relation = predict(model, tokenizer, sentence)
        print(f"{relation}\t{sentence}")


if __name__ == "__main__":
    main()
