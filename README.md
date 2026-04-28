# Visual Understanding with TextVQA

Final project for AI in Medicine — Spring 2026.

## Task

Evaluate and improve a vision-language model's ability to read and reason about text embedded in real-world images, using the [TextVQA](https://textvqa.org/) dataset.

**Model**: [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
**Dataset**: [TextVQA](https://huggingface.co/datasets/lmms-lab/textvqa) (34.6k train / 5k val / 5.7k test)

## Approach

1. **Zero-shot evaluation** — baseline performance of the pretrained model
2. **Prompt engineering** — 5 strategies designed to improve text reading accuracy:
   - Baseline (bare question)
   - OCR-augmented (provide detected text tokens)
   - Chain-of-thought (step-by-step reasoning)
   - Instructed-concise (system prompt enforcing short answers)
   - OCR + CoT combined
3. **LoRA fine-tuning** — parameter-efficient fine-tuning on the training set

## Setup

```bash
# Create and activate the conda environment
conda create -n textvqa python=3.12 -y
conda activate textvqa

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
pip install -r requirements.txt
```

## Usage

```bash
conda activate textvqa

# Zero-shot baseline on validation set
python scripts/run_zero_shot.py --dtype 4bit

# Run all prompt engineering strategies
python scripts/run_prompt_eng.py --dtype 4bit

# Run a single strategy
python scripts/run_prompt_eng.py --strategies ocr_cot --dtype 4bit

# Fine-tune with LoRA
python scripts/run_finetune.py --config configs/finetune.yaml

# Generate analysis plots and tables
python scripts/run_analysis.py

# Quick dev run (small sample)
python scripts/run_prompt_eng.py --max-samples 50 --dtype 4bit
```

## Evaluation Metrics

| Metric | Type | Description |
|--------|------|-------------|
| VQA Accuracy | Primary | Official TextVQA metric: `min(1, matches/3)` across 10 annotated answers |
| BLEU | Secondary | Corpus-level n-gram overlap |
| METEOR | Secondary | Alignment-based with synonyms and stemming |
| ROUGE-L | Secondary | Longest common subsequence |
| F1 | Optional | Token-level precision/recall |
| LLM-as-a-Judge | Optional | Model-based semantic similarity scoring |

## Project Structure

```
configs/                 Experiment configurations (YAML)
  base.yaml              Shared defaults
  zero_shot.yaml         Zero-shot evaluation config
  prompt_eng.yaml        Prompt engineering config
  finetune.yaml          LoRA fine-tuning config
src/                     Core library
  data.py                TextVQA dataset loading
  model.py               Model loading (bfloat16 / 4-bit quantized)
  prompts.py             Prompt strategy definitions and registry
  evaluate.py            All evaluation metrics
  train.py               LoRA fine-tuning loop
  utils.py               Answer normalization, config, seeding
scripts/                 Experiment entry points
  run_zero_shot.py       Zero-shot evaluation
  run_prompt_eng.py      Prompt engineering experiments
  run_finetune.py        Fine-tuning pipeline
  run_analysis.py        Results analysis and visualization
results/                 Auto-generated outputs
  zero_shot/             Zero-shot predictions and metrics
  prompt_eng/            Per-strategy results
  finetune/              Fine-tuning checkpoints and eval
  figures/               Comparison plots and error analysis
```

## Requirements

- Python 3.12
- NVIDIA GPU with CUDA support (tested on H100 NVL)
- ~3 GB VRAM (4-bit quantized) or ~7 GB (bfloat16)
