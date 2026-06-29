#!/usr/bin/env python3
"""
Relation Classification inference with a fine-tuned encoder baseline.

Loads one of the fine-tuned encoder models (from the Hugging Face Hub or a local
folder) and predicts the directional relation between two marked entities
<e1> and <e2>. The model was trained with four special entity-marker tokens and
a classification head over 19 directional labels, which are collapsed to the 10
coarse SemEval-2010 Task 8 relations at output time.

These encoders are much smaller and faster than the QLoRA Gemma adapters (they
run on a small GPU or even CPU) and reach comparable accuracy on Relation
Classification. See the paper for the comparison.

Example:
    python infer_encoder.py \
        --sentence "<e1>Zahărul</e1> a fost dizolvat în <e2>apă</e2>." \
        --model DS4AI-UPB/xlmr-large-ro-re

    python infer_encoder.py \
        --file my_sentences.txt \
        --model DS4AI-UPB/xlmr-large-ro-re

Requirements:
    pip install torch transformers
"""

import argparse
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# The 19 directional labels the classifier head was trained on.
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
ID2LABEL = {i: lbl for i, lbl in enumerate(RELATIONS)}


def convert_markers(text):
    """Map the <e1>/<e2> tags to the special tokens the model was trained with."""
    text = text.replace("<e1>", "[E1] ").replace("</e1>", " [/E1]")
    text = text.replace("<e2>", "[E2] ").replace("</e2>", " [/E2]")
    return text


def load_model(model_id):
    print(f"Loading encoder: {model_id}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


@torch.no_grad()
def predict(model, tokenizer, sentence, max_len=192):
    text = convert_markers(sentence)
    inputs = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    pred_id = int(logits.argmax(dim=-1).item())
    return ID2LABEL.get(pred_id, "Other")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sentence", help="A single sentence with <e1> and <e2> markers.")
    src.add_argument("--file", help="A text file with one marked sentence per line.")
    parser.add_argument("--model", required=True,
                        help="HF repo id or local path of the fine-tuned encoder.")
    parser.add_argument("--max-len", type=int, default=192)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.sentence:
        sentences = [args.sentence]
    else:
        with open(args.file, "r", encoding="utf-8") as f:
            sentences = [ln.strip() for ln in f if ln.strip()]

    for sentence in sentences:
        label = predict(model, tokenizer, sentence, args.max_len)
        print(f"{label}\t{sentence}")


if __name__ == "__main__":
    main()
