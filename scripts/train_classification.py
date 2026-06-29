#!/usr/bin/env python3
"""
QLoRA fine-tuning of Gemma 4 31B for Relation Classification on SemEval-2010 Task 8.

Trains on the combined English and Romanian training sets and saves a LoRA
adapter. The prompt and label format match infer_classification.py, so the
saved adapter can be used directly by that script.

Example:
    python scripts/train_classification.py \
        --base-model google/gemma-4-31b-it \
        --train-en data/train_en.jsonl \
        --train-ro data/train_ro_clean.jsonl \
        --output-dir models/gemma4-ro-re-lora

Reproducing the paper uses Unsloth for efficient 4-bit QLoRA training on a
single A100 40GB. Install: pip install unsloth trl datasets
"""

import argparse
import json
import random

import torch
from datasets import Dataset
from unsloth import FastModel
from trl import SFTTrainer, SFTConfig

RELATIONS = [
    "Cause-Effect", "Instrument-Agency", "Product-Producer",
    "Content-Container", "Entity-Origin", "Entity-Destination",
    "Component-Whole", "Member-Collection", "Message-Topic", "Other",
]


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


def format_example(item, lang):
    """One example as a (user, assistant) chat pair."""
    sentence = get_sentence(item, lang)
    user_msg = (
        f"Classify the semantic relation between <e1> and <e2> in this sentence.\n\n"
        f"Possible relations: {', '.join(RELATIONS)}\n\n"
        f"Sentence: {sentence}\n\n"
        f"Respond with ONLY the relation and direction, e.g. Cause-Effect(e1,e2)"
    )
    return (
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": item["relation"]},
    )


def build_dataset(train_en, train_ro, tokenizer, seed):
    en_data = load_jsonl(train_en)
    ro_data = load_jsonl(train_ro)
    print(f"Loaded {len(en_data)} English + {len(ro_data)} Romanian examples")

    conversations = []
    for item in en_data:
        u, a = format_example(item, "en")
        conversations.append({"messages": [u, a]})
    for item in ro_data:
        u, a = format_example(item, "ro")
        conversations.append({"messages": [u, a]})

    random.seed(seed)
    random.shuffle(conversations)
    print(f"Total training conversations: {len(conversations)}")

    dataset = Dataset.from_list(conversations)

    def apply_template(examples):
        return {"text": [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in examples["messages"]
        ]}

    return dataset.map(apply_template, batched=True, remove_columns=["messages"])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="google/gemma-4-31b-it",
                   help="Base model id or local path.")
    p.add_argument("--train-en", default="data/train_en.jsonl")
    p.add_argument("--train-ro", default="data/train_ro_clean.jsonl")
    p.add_argument("--output-dir", default="models/gemma4-ro-re-lora")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8, help="Effective batch = batch_size * grad_accum.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("QLoRA fine-tuning: Gemma 4 31B for Relation Classification")
    print("=" * 60)

    print(f"\nLoading base model from {args.base_model} ...")
    model, tokenizer = FastModel.from_pretrained(
        args.base_model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )

    print(f"\nAdding LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha}) ...")
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        random_state=args.seed,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    print("\nPreparing dataset ...")
    dataset = build_dataset(args.train_en, args.train_ro, tokenizer, args.seed)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        bf16=True,
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=1,
        seed=args.seed,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        packing=True,
        report_to="none",
    )

    print("\nStarting training ...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    trainer.train()

    print(f"\nSaving LoRA adapter to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
