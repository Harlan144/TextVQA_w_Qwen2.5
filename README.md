# Visual Understanding with TextVQA

Final project for AI in Medicine — Spring 2026.

## Task

Evaluate and improve a vision-language model's ability to read and reason about text embedded in real-world images, using the [TextVQA](https://textvqa.org/) dataset.

**Model**: [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
**Dataset**: [TextVQA](https://huggingface.co/datasets/lmms-lab/textvqa) (34.6k train / 5k val / 5.7k test)

## Approach

1. **Zero-shot evaluation** — baseline performance of the pretrained model
2. **Prompt engineering** — 9 strategies exploring different prompt variables:
   - Baseline (system prompt + image + question + concise instruction)
   - OCR-augmented (+OCR tokens)
   - Chain-of-thought (+CoT instruction)
   - No-system-prompt (removes the system prompt)
   - OCR + CoT (+OCR tokens + CoT instruction)
   - OCR-only (+OCR tokens, no image — ablation)
   - Few-shot (in-context examples before the query)
   - Format-constraint (explicit "1–3 words only" instruction)
   - Think-hard (encourages careful reasoning with concise output)
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
| F1 | Secondary | Token-level precision/recall |
| Precision/Recall | Secondary | Token-level precision and recall (reported separately) |
| LLM-as-a-Judge | Secondary | Model-based semantic similarity (scores 1.0 if prediction matches any of 10 references) |

## Project Structure

```
configs/                 Experiment configurations (YAML)
  base.yaml              Shared defaults
  zero_shot.yaml         Zero-shot evaluation config
  prompt_eng.yaml        Prompt engineering config
  finetune.yaml          LoRA fine-tuning config
data/                    TextVQA dataset (HuggingFace Arrow cache)
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
  run_llm_judge.py       LLM-as-a-Judge scoring
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

## How the Pipeline Works

### Step 1: Load a sample from the dataset

`src/data.py` loads the TextVQA dataset from `data/textvqa/` (downloaded from HuggingFace and cached locally as Arrow files). Each sample contains:

- **image**: a PIL Image (the photo with text in it)
- **question**: e.g. "What brand of phone is this?"
- **answers**: 10 human-annotated answers, e.g. `["nokia", "Nokia", "NOKIA", ...]`
- **ocr_tokens**: text detected in the image by an OCR system (Rosetta), e.g. `["NOKIA", "E71", "Operator"]`
- **image_classes**: object categories in the image, e.g. `["phone"]`

The answers and ocr_tokens come from the dataset itself — we don't run OCR or annotation ourselves.

### Step 2: Build a prompt

`src/prompts.py` takes the image, question, and optionally ocr_tokens, and constructs a chat-format message for the Qwen model. All strategies share a system prompt enforcing concise answers and a common user prompt structure. Each strategy ablates exactly one variable from the baseline.

For example, the **baseline** strategy produces:

```
[{"role": "system", "content": "You are a visual question answering assistant specialized in reading text from images. Always give short, precise answers..."},
 {"role": "user", "content": [
    {image},
    "Question: What brand of phone is this?\nAnswer the question about this image concisely.\nAnswer:"
]}]
```

The **OCR-augmented** strategy adds the detected text tokens to the user prompt. The **chain-of-thought** strategy replaces the concise instruction with a step-by-step reasoning instruction. The **no-system-prompt** strategy removes the system prompt to test its contribution. The **OCR-only** strategy includes OCR tokens but removes the image to measure how much the model relies on vision vs. text.

### Step 3: Model generates an answer

`src/model.py` sends the prompt to Qwen2.5-VL-3B-Instruct:

1. `processor.apply_chat_template()` converts the message list into the token format Qwen expects
2. `process_vision_info()` extracts the PIL image and converts it to pixel tensors
3. `processor()` tokenizes the text and image together into model inputs
4. `model.generate()` runs autoregressive generation (up to 128 new tokens)
5. The generated token IDs are decoded back to text, e.g. `"Nokia"`

The model sees the actual image pixels (processed by a ViT vision encoder) and the text prompt (processed by the language model). It "reads" text in the image through its vision encoder — not from the ocr_tokens (unless the prompt strategy explicitly includes them).

For CoT strategies, the model might output something like: `"The phone has NOKIA written on it, so the brand is Nokia. Answer: Nokia"`. We parse the text after `"Answer:"` to extract just `"Nokia"`.

### Step 4: Grade the answer

`src/evaluate.py` and `src/utils.py` compare the model's prediction against the 10 human answers.

**Answer normalization** (applied to both prediction and ground truths):
- Lowercase: `"Nokia"` → `"nokia"`
- Remove articles: `"the nokia"` → `"nokia"`
- Remove punctuation: `"Nokia."` → `"nokia"`
- Number words to digits: `"two"` → `"2"`
- Expand contractions: `"dont"` → `"don't"`

**VQA Accuracy scoring** (per question):
- Count how many of the 10 ground truth answers match the normalized prediction
- Score = `min(1.0, matches / 3)`
- So if 3+ annotators wrote the same answer and the model matches, it gets full credit
- This soft voting handles annotator disagreement: a question might have answers like `["nokia", "nokia", "Nokia", "nokia phone", ...]`

**Example**: Model predicts `"Nokia."` → normalized to `"nokia"` → matches 8 of 10 ground truths → score = `min(1, 8/3)` = **1.0**

**Example**: Model predicts `"It's a Nokia phone"` → normalized to `"it's nokia phone"` → matches 0 of 10 → score = **0.0** (even though it's semantically correct)

This is why prompt strategies that produce concise answers score higher — verbose correct answers fail exact matching.

**Final VQA Accuracy** is the mean score across all questions (reported as a percentage).

Secondary metrics (BLEU, METEOR, ROUGE-L, F1) use standard NLP comparison methods that are more forgiving of partial matches, which is why CoT strategies score less badly on those.

### Step 5: Save and analyze results

Each experiment saves to `results/{experiment}/{strategy}/{split}/`:
- `predictions.json`: every question with the model's answer, raw output, and ground truths
- `metrics.json`: all computed metric scores
- `per_category.json`: accuracy broken down by image object category

`scripts/run_analysis.py` reads all results and generates comparison tables, bar charts, heatmaps, per-category breakdowns, and error analysis with qualitative examples.
