# Cross-lingual Relation Extraction for Romanian

Code, data, and trained models for the paper **"Cross-lingual Relation Extraction with Large Language Models: Zero-Shot, Few-Shot, and Fine-Tuned Evaluation on Romanian"** (SYNASC 2026).

We translate the SemEval-2010 Task 8 relation extraction benchmark from English to Romanian and evaluate two open-weight LLMs, Gemma 4 31B and Qwen 2.5 32B (zero-shot, few-shot, and QLoRA fine-tuned), against four encoder baselines and a two-encoder pipeline baseline, under two task formulations: Relation Classification and End-to-End RE.

## Main findings

- On **Relation Classification**, a fine-tuned XLM-RoBERTa-large (560M) is statistically indistinguishable from both QLoRA LLMs on Romanian (vs Gemma p = 0.09, vs Qwen p = 0.13), despite being roughly 55x smaller.
- **Prompt-only performance is model-dependent:** Gemma and Qwen differ by 33pp zero-shot, but converge to within 1pp after QLoRA, both landing at the encoder ceiling.
- On **End-to-End RE**, a two-encoder pipeline (a span detector plus the relation classifier) matches the stronger LLM on exact match (vs Gemma: not significant on either language) and beats the weaker one (vs Qwen: p < 0.005), while being far smaller. The apparent LLM advantage on end-to-end does not survive a matched baseline.
- The cross-lingual gap is not distinguishable from zero in prompt-only settings once exemplar-sampling variance is accounted for; after fine-tuning it is 1.4pp for Gemma and 2.6pp for Qwen.

## Released models

All trained models are on the Hugging Face Hub under [DS4AI-UPB](https://huggingface.co/DS4AI-UPB).

**QLoRA adapters** for `google/gemma-4-31b-it` and `Qwen/Qwen2.5-32B-Instruct`:

| Task | Base | Repo |
|------|------|------|
| Relation Classification | Gemma 4 | [`DS4AI-UPB/gemma4-ro-re-lora`](https://huggingface.co/DS4AI-UPB/gemma4-ro-re-lora) |
| End-to-End RE | Gemma 4 | [`DS4AI-UPB/gemma4-ro-e2e-lora`](https://huggingface.co/DS4AI-UPB/gemma4-ro-e2e-lora) |
| Relation Classification | Qwen 2.5 | [`DS4AI-UPB/qwen25-ro-re-lora`](https://huggingface.co/DS4AI-UPB/qwen25-ro-re-lora) |
| End-to-End RE | Qwen 2.5 | [`DS4AI-UPB/qwen25-ro-e2e-lora`](https://huggingface.co/DS4AI-UPB/qwen25-ro-e2e-lora) |

**Span detectors** for the end-to-end pipeline baseline (XLM-RoBERTa-large, token classification):

| Language | Repo |
|----------|------|
| Romanian | [`DS4AI-UPB/span-detector-ro`](https://huggingface.co/DS4AI-UPB/span-detector-ro) |
| English  | [`DS4AI-UPB/span-detector-en`](https://huggingface.co/DS4AI-UPB/span-detector-en) |

**Encoder baselines** for Relation Classification (much smaller and faster, run on a small GPU or CPU):

| Model | Params | Languages | Repo |
|-------|--------|-----------|------|
| XLM-RoBERTa-large | 560M | RO + EN | [`DS4AI-UPB/xlmr-large-ro-re`](https://huggingface.co/DS4AI-UPB/xlmr-large-ro-re) |
| XLM-RoBERTa-base | 278M | RO + EN | [`DS4AI-UPB/xlmr-base-ro-re`](https://huggingface.co/DS4AI-UPB/xlmr-base-ro-re) |
| RoBERT-large | 340M | RO | [`DS4AI-UPB/robert-large-ro-re`](https://huggingface.co/DS4AI-UPB/robert-large-ro-re) |
| BERT-base-Romanian | 125M | RO | [`DS4AI-UPB/bert-base-romanian-re`](https://huggingface.co/DS4AI-UPB/bert-base-romanian-re) |

**Dataset:** [`DS4AI-UPB/romanian-re-semeval`](https://huggingface.co/datasets/DS4AI-UPB/romanian-re-semeval)

## Quick start: run inference

```bash
pip install -r requirements.txt
hf auth login   # the base model google/gemma-4-31b-it is gated
```

**Encoder (lightweight, recommended for Relation Classification):**

```bash
python scripts/infer_encoder.py \
    --sentence "<e1>Furtuna</e1> a provocat mari <e2>pagube</e2>." \
    --model DS4AI-UPB/xlmr-large-ro-re
```

**Gemma QLoRA adapter** (downloads the 31B base model, about 25 GB of GPU memory):

```bash
# Relation Classification: the sentence already has <e1> and <e2> markers
python infer_classification.py \
    --sentence "<e1>Zahărul</e1> a fost dizolvat în <e2>apă</e2>." \
    --adapter DS4AI-UPB/gemma4-ro-re-lora

# End-to-End RE: a plain sentence, the model finds the entities and the relation
python infer_e2e.py \
    --sentence "Compania a lansat un nou telefon săptămâna trecută." \
    --adapter DS4AI-UPB/gemma4-ro-e2e-lora
```

All three inference scripts also take `--file sentences.txt` (one sentence per line), and the model argument accepts a **local folder** instead of a Hub id, so a model you trained yourself works the same way.

## Repository structure

```
.
├── infer_classification.py    # Run the Gemma classification adapter (HF or local)
├── infer_e2e.py               # Run the Gemma end-to-end adapter (HF or local)
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
│   ├── infer_encoder.py              # Run an encoder baseline (HF or local)
│   ├── eval_classification.py        # Evaluate the classification adapter
│   ├── eval_e2e.py                   # Evaluate the end-to-end adapter
│   └── significance_test.py          # Paired bootstrap significance test
├── results/                   # Predictions (.jsonl) + metrics (.json)
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

The same data is on the Hub as [`DS4AI-UPB/romanian-re-semeval`](https://huggingface.co/datasets/DS4AI-UPB/romanian-re-semeval), loadable with `load_dataset("DS4AI-UPB/romanian-re-semeval")`.

## Reproducing the experiments

All experiments run on a single NVIDIA A100 40GB. QLoRA training uses Unsloth; the encoder baselines use Transformers. To install the training dependencies, uncomment the relevant lines in `requirements.txt`.

```bash
# 1. Convert the original SemEval-2010 Task 8 .txt files to JSONL
python scripts/convert_semeval_to_jsonl.py --input TRAIN_FILE.TXT --output data/train_en.jsonl

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

The base model (`google/gemma-4-31b-it`) is not redistributed here; download it from its official source.

## Reproducing the analysis

The `scripts/` directory contains the full evaluation pipeline:

- `train_span_detector.py` / `pipeline_e2e.py` - train the BIO span detector and run the end-to-end encoder pipeline baseline.
- `significance_e2e.py` / `significance_test.py` - paired bootstrap tests (10,000 resamples) for end-to-end and classification.
- `aggregate_variance.py` / `paired_gap.py` - few-shot variance across exemplar draws and the cross-lingual gap.
- `per_relation.py` - per-relation F1-Score across models.
- `clean_subset_eval.py` - end-to-end scores on the subset excluding untranslated entities.

Per-model predictions and metrics are in `results/` (Gemma + encoders) and `results_qwen/` (Qwen).

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
