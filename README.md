# Cross-lingual Relation Extraction for Romanian

Code, data, and trained adapters for the paper **"Cross-lingual Relation Extraction with Large Language Models: Zero-Shot, Few-Shot, and Fine-Tuned Evaluation on Romanian"** (SYNASC 2026).

We translate the SemEval-2010 Task 8 relation extraction benchmark from English to Romanian and evaluate Gemma 4 31B (zero-shot, few-shot, and QLoRA fine-tuned) against four encoder baselines, under two task formulations: Relation Classification and End-to-End RE.

## Main findings

- On **Relation Classification**, a fine-tuned XLM-RoBERTa-large (560M) is statistically indistinguishable from QLoRA Gemma 4 (31B) on both English (p = 0.23) and Romanian (p = 0.09), despite being roughly 55x smaller.
- On **End-to-End RE**, where entities are not given, the fine-tuned LLM has a clear advantage: QLoRA raises exact match by about 39pp over zero-shot in both languages, and encoder classifiers do not apply directly.
- QLoRA fine-tuning narrows the English-Romanian gap from 3.3pp to 1.4pp on classification.

## Quick start: run inference

The two QLoRA adapters are on the Hugging Face Hub. The inference scripts download the adapter and apply it on top of the base Gemma 4 31B model (4-bit, about 25 GB of GPU memory).

```bash
pip install -r requirements.txt
hf auth login   # the base model google/gemma-4-31b-it is gated

# Relation Classification: the sentence already has <e1> and <e2> markers
python infer_classification.py \
    --sentence "<e1>Zahărul</e1> a fost dizolvat în <e2>apă</e2>." \
    --adapter DS4AI-UPB/gemma4-ro-re-lora

# End-to-End RE: a plain sentence, the model finds the entities and the relation
python infer_e2e.py \
    --sentence "Compania a lansat un nou telefon săptămâna trecută." \
    --adapter DS4AI-UPB/gemma4-ro-e2e-lora
```

Both scripts also take `--file sentences.txt` (one sentence per line) and accept a
**local folder** instead of a Hub id for `--adapter`, so a model you trained
yourself works the same way.

## Repository structure

```
.
├── infer_classification.py    # Run the classification adapter (HF or local)
├── infer_e2e.py               # Run the end-to-end adapter (HF or local)
├── requirements.txt
├── data/                      # Translated Romanian dataset + English source
│   ├── train_ro_clean.jsonl
│   ├── test_ro_clean.jsonl
│   ├── train_en.jsonl
│   └── test_en.jsonl
├── scripts/
│   ├── convert_semeval_to_jsonl.py   # SemEval .txt -> JSONL
│   ├── run_inference.py              # Zero-shot / few-shot for both tasks
│   ├── train_classification.py       # QLoRA classification training
│   ├── train_e2e.py                  # QLoRA end-to-end training
│   ├── train_encoder_baseline.py     # Encoder baselines (validation-based selection)
│   ├── eval_classification.py        # Evaluate the classification adapter
│   ├── eval_e2e.py                   # Evaluate the end-to-end adapter
│   └── significance_test.py          # Paired bootstrap significance test
├── results/                   # Predictions (.jsonl) + metrics (.json)
│   ├── A_*.jsonl                     # Relation Classification predictions
│   ├── B_*.jsonl                     # End-to-End RE predictions
│   ├── translation_sample_100.jsonl  # Manually inspected translation sample
│   └── bootstrap_results.txt         # Significance test output
└── README.md
```

## Data format

Each line is a JSON object. Romanian files keep both the English source and the Romanian translation with aligned entity markers:

```json
{
  "sentence_en": "The <e1>cup</e1> contained <e2>coffee</e2>.",
  "sentence_ro": "<e1>Cana</e1> conținea <e2>cafea</e2>.",
  "relation": "Content-Container(e2,e1)",
  "validation": {"e1_en": "cup", "e1_ro": "Cana", "e2_en": "coffee", "e2_ro": "cafea"}
}
```

The dataset is **machine-translated with automatic post-validation**, not a human gold standard. A manual inspection of 100 sentences found that 74% have both entities translated and aligned correctly; the remaining 26% have entity-level issues (14 untranslated entities, 9 misplaced markers, 3 mistranslations), 12 of them severe. Romanian End-to-End RE numbers should be read as a lower bound. See the paper, Section "Dataset Construction", for details.

## Trained adapters

The QLoRA adapters for `google/gemma-4-31b-it` are on the Hugging Face Hub:

- **Relation Classification:** [`DS4AI-UPB/gemma4-ro-re-lora`](https://huggingface.co/DS4AI-UPB/gemma4-ro-re-lora)
- **End-to-End RE:** [`DS4AI-UPB/gemma4-ro-e2e-lora`](https://huggingface.co/DS4AI-UPB/gemma4-ro-e2e-lora)

The base model (`google/gemma-4-31b-it`) is not redistributed here; download it from its official source. The encoder baselines are not distributed either, since they are reproducible from `scripts/train_encoder_baseline.py`.

## Reproducing the experiments

All experiments run on a single NVIDIA A100 40GB. QLoRA training uses Unsloth; the encoder baselines use Transformers. To install the training dependencies, uncomment the relevant lines in `requirements.txt`.

```bash
# 1. Convert the original SemEval-2010 Task 8 .txt files to JSONL
python scripts/convert_semeval_to_jsonl.py

# 2. Zero-shot / few-shot inference with the base Gemma 4
python scripts/run_inference.py --help

# 3. QLoRA training (classification, then end-to-end)
python scripts/train_classification.py --output-dir models/gemma4-ro-re-lora
python scripts/train_e2e.py --output-dir models/gemma4-ro-e2e-lora

# 4. Encoder baselines (all four, validation-based checkpoint selection)
python scripts/train_encoder_baseline.py --model xlm-roberta-large --mode multilingual --tag xlmr
python scripts/train_encoder_baseline.py --model xlm-roberta-base --mode multilingual --tag xlmr-base
python scripts/train_encoder_baseline.py --model dumitrescustefan/bert-base-romanian-cased-v1 --mode ro-only --tag bert-ro-base
python scripts/train_encoder_baseline.py --model readerbench/RoBERT-large --mode ro-only --tag robert-large

# 5. Evaluate the fine-tuned adapters
python scripts/eval_classification.py --adapter models/gemma4-ro-re-lora
python scripts/eval_e2e.py --adapter models/gemma4-ro-e2e-lora

# 6. Significance testing
python scripts/significance_test.py --a results/A_qlora_ro.jsonl --b results/A_xlmr_ro.jsonl
```

## Citation

```bibtex
@misc{vasile2026crosslingual,
  title  = {Cross-lingual Relation Extraction with Large Language Models: Zero-Shot, Few-Shot, and Fine-Tuned Evaluation on Romanian},
  author = {Vasile, Drago\c{s}-Mitru\c{t} and Apostol, Elena-Simona and Toma, \c{S}tefan-Adrian and Truic\u{a}, Ciprian-Octavian},
  year   = {2026},
  note   = {Preprint}
}
```

## License

The code is released under the MIT License. The translated dataset is derived from SemEval-2010 Task 8; please also respect the original benchmark's terms.
