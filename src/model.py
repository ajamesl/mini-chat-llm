from typing import Any

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer)


def load_tokenizer(
    model_name: str, padding_side: str = "right", trust_remote_code: bool = True
) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def load_causal_lm(
    model_name: str, trust_remote_code: bool = True, device: str = "cpu"
) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    model = model.to(device)
    return model


def load_sequence_classifier(
    model_name: str,
    trust_remote_code: bool = True,
    num_labels: int = 1,
    device: str = "cpu",
) -> Any:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, num_labels=num_labels
    )
    model = model.to(device)
    return model


def apply_lora(model: Any, lora_config: LoraConfig) -> Any:
    return get_peft_model(model, lora_config)


def load_peft_model(base_model: Any, peft_path: str, device: str = "cpu") -> Any:
    model = PeftModel.from_pretrained(base_model, peft_path)
    model = model.to(device)
    return model
