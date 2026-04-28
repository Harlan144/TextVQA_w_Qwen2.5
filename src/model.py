"""Model loading and generation for Qwen2.5-VL."""

import logging
from typing import Optional

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)


def load_model(
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    dtype: str = "bfloat16",
    device_map: str = "auto",
):
    """Load model and processor.

    Args:
        model_name: HuggingFace model ID.
        dtype: "bfloat16" for full precision, "4bit" for quantized.
        device_map: Device placement strategy.

    Returns:
        (model, processor) tuple.
    """
    logger.info(f"Loading model {model_name} (dtype={dtype}, device_map={device_map})")

    load_kwargs = {"device_map": device_map, "trust_remote_code": True}

    if dtype == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, **load_kwargs
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded — {param_count / 1e9:.2f}B parameters")
    return model, processor


@torch.no_grad()
def generate_answer(
    model,
    processor,
    messages: list[dict],
    max_new_tokens: int = 128,
) -> str:
    """Run inference for a single sample.

    Args:
        model: The loaded VL model.
        processor: The corresponding processor.
        messages: Chat-format messages (with image and text content).
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Decoded answer string.
    """
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # Trim the input tokens from output
    trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output[0].strip()


@torch.no_grad()
def generate_batch(
    model,
    processor,
    batch_messages: list[list[dict]],
    max_new_tokens: int = 128,
) -> list[str]:
    """Run inference for a batch of samples.

    Args:
        model: The loaded VL model.
        processor: The corresponding processor.
        batch_messages: List of chat-format message lists.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        List of decoded answer strings.
    """
    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in batch_messages
    ]

    all_images = []
    for msgs in batch_messages:
        img_inputs, _ = process_vision_info(msgs)
        if img_inputs:
            all_images.extend(img_inputs)

    inputs = processor(
        text=texts,
        images=all_images if all_images else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    answers = []
    for i, (inp, out) in enumerate(zip(inputs.input_ids, generated_ids)):
        trimmed = out[len(inp):]
        decoded = processor.decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        answers.append(decoded.strip())
    return answers
