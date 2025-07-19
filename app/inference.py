"""Inference utilities for generating responses using a causal language model."""
import threading
from typing import Generator

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer)

# Update the MODEL_PATH to choose the relative folder for the desired model (sft_model or ppo_model)
MODEL_PATH = "checkpoints/ppo_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True).to(
    device
)
model.eval()


def generate_stream(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> Generator[str, None, None]:
    """Generate a response stream from the language model given a user prompt.

    Args:
        prompt (str): The user's input prompt.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 256.
        temperature (float, optional): Sampling temperature. Defaults to 0.7.
        top_p (float, optional): Nucleus sampling probability. Defaults to 0.95.

    Yields:
        str: The next chunk of generated text from the model.
    """
    input_text = f"User: {prompt}\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        streamer=streamer,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    output_so_far = ""
    prefix_removed = False

    for new_text in streamer:
        output_so_far += new_text
        if not prefix_removed:
            idx = output_so_far.find("Assistant:")
            if idx != -1:
                output_so_far = output_so_far[idx + len("Assistant:") :].lstrip("\n ")
                prefix_removed = True
            else:
                output_so_far = ""
                continue
        yield output_so_far
        output_so_far = ""
