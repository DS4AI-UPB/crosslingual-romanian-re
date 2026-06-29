#!/usr/bin/env python3
"""
Convert SemEval-2010 Task 8 TXT format to JSONL (for English baseline).

Usage:
    python convert_semeval_to_jsonl.py --input TRAIN_FILE.TXT --output data/train_en.jsonl
    python convert_semeval_to_jsonl.py --input TEST_FILE_FULL.TXT --output data/test_en.jsonl
"""

import argparse
import json
import re


def parse_and_convert(input_path: str, output_path: str):
    entries = []

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        match = re.match(r'^(\d+)\t"(.+)"$', line)
        if not match:
            i += 1
            continue

        entry_id = int(match.group(1))
        sentence = match.group(2)

        i += 1
        relation = lines[i].strip() if i < len(lines) else ""

        # Extract entities
        e1 = re.search(r"<e1>(.*?)</e1>", sentence)
        e2 = re.search(r"<e2>(.*?)</e2>", sentence)

        entry = {
            "id": entry_id,
            "sentence_en": sentence,
            "relation": relation,
            "e1_en": e1.group(1) if e1 else "",
            "e2_en": e2.group(1) if e2 else "",
        }
        entries.append(entry)
        i += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Converted {len(entries)} entries: {input_path} → {output_path}")

    # Quick stats
    from collections import Counter
    rels = Counter(e["relation"].split("(")[0] for e in entries)
    for rel, count in rels.most_common():
        print(f"  {rel:<30s} {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    parse_and_convert(args.input, args.output)
