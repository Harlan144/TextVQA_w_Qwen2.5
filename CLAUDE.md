# TextVQA Final Project

## Overview
AI in Medicine (Spring 2026) final project. Evaluate and improve Qwen2.5-VL-3B-Instruct on the TextVQA dataset using prompt engineering and LoRA fine-tuning.

## Environment
- **Conda env**: `textvqa` (Python 3.12, PyTorch 2.6+cu124, transformers 5.6.2)
- **Do NOT use base conda** — it has different package versions and is shared
- Activate with `conda activate textvqa` or prefix commands with `conda run -n textvqa`
- Hardware: 2x H100 NVL (shared, often occupied) — use `--dtype 4bit` when GPU memory is tight

## Project Structure
```
configs/          YAML configs for experiments (base, zero_shot, prompt_eng, finetune)
src/              Core modules (data, model, prompts, evaluate, train, utils)
scripts/          Entry point scripts (run_zero_shot, run_prompt_eng, run_finetune, run_analysis)
results/          Auto-generated experiment outputs (predictions, metrics, figures)
```

## Running Experiments
```bash
# Zero-shot baseline
python scripts/run_zero_shot.py --dtype 4bit

# All prompt engineering strategies
python scripts/run_prompt_eng.py --dtype 4bit

# Single strategy
python scripts/run_prompt_eng.py --strategies ocr_cot --dtype 4bit

# Fine-tuning with LoRA
python scripts/run_finetune.py --config configs/finetune.yaml

# Analysis and plots
python scripts/run_analysis.py
```

Use `--max-samples N` on any script for quick dev iteration.

## Key Design Decisions
- Prompt strategies are registered in `src/prompts.py` — add new ones there
- VQA accuracy uses official normalization (lowercase, remove articles/punctuation, number words to digits)
- CoT strategies use `extract_final_answer()` to parse the "Answer:" line from model output
- Model loading supports bfloat16 and 4-bit quantization via bitsandbytes
