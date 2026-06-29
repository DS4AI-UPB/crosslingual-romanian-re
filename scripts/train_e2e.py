#!/usr/bin/env python3
"""
QLoRA fine-tuning of Gemma 4 31B for End-to-End Relation Extraction on
SemEval-2010 Task 8.

The model is trained to read a plain sentence (no entity markers) and generate
both entities and the relation as JSON:
  {"e1": "...", "e2": "...", "relation": "Relation(e1,e2)"}

The prompt and target format match infer_e2e.py, so the saved adapter can be
used directly by that script.

Example:
    python scripts/train_e2e.py \
        --base-model google/gemma-4-31b-it \
        --train-en data/train_en.jsonl \
        --train-ro data/train_ro_clean.jsonl \
        --output-dir models/gemma4-ro-e2e-lora

Install: pip install unsloth trl datasets
"""

import argparse
import json
import random
import re

import torch
from datasets import Dataset
from unsloth import FastModel
from trl import SFTTrainer, SFTConfig

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


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def get_tagged_sentence(item, lang):
    return item.get(f"sentence_{lang}", item["sentence_en"])


def build_example(item, lang, relations_text):
    """Build a (prompt, target JSON) pair matching the inference format."""
    tagged = get_tagged_sentence(item, lang)
    e1 = re.search(r"<e1>(.*?)</e1>", tagged)
    e2 = re.search(r"<e2>(.*?)</e2>", tagged)
    e1_text = e1.group(1) if e1 else "?"
    e2_text = e2.group(1) if e2 else "?"
    sentence_clean = re.sub(r"</?e[12]>", "", tagged)

    prompt = PROMPT_E2E.format(relations=relations_text, sentence=sentence_clean)
    target = json.dumps(
        {"e1": e1_text, "e2": e2_text, "relation": item["relation"]},
        ensure_ascii=False,
    )
    return prompt, target


def build_dataset(train_en, train_ro, tokenizer, seed):
    relations_text = relations_block()
    en_data = load_jsonl(train_en)
    ro_data = load_jsonl(train_ro)
    print(f"Loaded {len(en_data)} English + {len(ro_data)} Romanian examples")

    conversations = []
    for item in en_data:
        prompt, target = build_example(item, "en", relations_text)
        conversations.append({"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]})
    for item in ro_data:
        prompt, target = build_example(item, "ro", relations_text)
        conversations.append({"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]})

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
    p.add_argument("--base-model", default="google/gemma-4-31b-it")
    p.add_argument("--train-en", default="data/train_en.jsonl")
    p.add_argument("--train-ro", default="data/train_ro_clean.jsonl")
    p.add_argument("--output-dir", default="models/gemma4-ro-e2e-lora")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=640,
                   help="Longer than classification: prompt lists relations + JSON answer.")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("QLoRA fine-tuning: Gemma 4 31B for End-to-End RE")
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
        packing=False,
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
