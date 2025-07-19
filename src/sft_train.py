import random
from typing import Any

import evaluate
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, TaskType
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, default_data_collator

from config import (SFT_EVAL_BATCH_SIZE, SFT_EVAL_STEPS,
                    SFT_GRADIENT_ACCUMULATION_STEPS, SFT_LEARNING_RATE,
                    SFT_MAX_INPUT_LENGTH, SFT_MODEL_NAME, SFT_NUM_TRAIN_EPOCHS,
                    SFT_OUTPUT_DIR, SFT_SAVE_STEPS, SFT_SEED,
                    SFT_TRAIN_BATCH_SIZE)
from model import apply_lora, load_causal_lm, load_tokenizer
from utils import build_prompt_response_pairs, set_seed


class OASSTDataset(Dataset):
    def __init__(
        self, pairs: list[dict[str, str]], tokenizer: Any, max_length: int = 1024
    ) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        prompt = self.pairs[idx]["prompt"]
        response = self.pairs[idx]["response"]
        text = f"User: {prompt}\nAssistant: {response}"
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def compute_metrics(eval_preds: Any) -> dict[str, float]:
    label_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    bertscore_result = bertscore.compute(
        predictions=pred_str, references=label_str, lang="en"
    )
    bleu_result = bleu.compute(
        predictions=pred_str, references=[[x] for x in label_str]
    )
    meteor_result = meteor.compute(predictions=pred_str, references=label_str)
    return {
        "bertscore_precision": sum(bertscore_result["precision"])
        / len(bertscore_result["precision"]),
        "bertscore_recall": sum(bertscore_result["recall"])
        / len(bertscore_result["recall"]),
        "bertscore_f1": sum(bertscore_result["f1"]) / len(bertscore_result["f1"]),
        "bleu": bleu_result["bleu"],
        "meteor": meteor_result["meteor"],
    }


def print_sample_generations(
    model: Any, tokenizer: Any, val_pairs: list[dict[str, str]], n: int = 5
) -> None:
    model.eval()
    for _ in range(n):
        sample = random.choice(val_pairs)
        prompt = sample["prompt"]
        input_text = f"User: {prompt}\nAssistant:"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
            )
        print("\n---")
        print("Prompt:", prompt)
        print("Reference:", sample["response"])
        print("Model:", tokenizer.decode(output[0], skip_special_tokens=True))


device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(SFT_SEED)

# Load model and tokenizer

tokenizer = load_tokenizer(SFT_MODEL_NAME, trust_remote_code=True)
model = load_causal_lm(SFT_MODEL_NAME, trust_remote_code=True, device=device)
model.config.pad_token_id = tokenizer.pad_token_id

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = apply_lora(model, lora_config)
model.print_trainable_parameters()
model.train()

train_data = load_dataset("OpenAssistant/oasst1", split="train")
val_data = load_dataset("OpenAssistant/oasst1", split="validation")

train_pairs = build_prompt_response_pairs(train_data)
val_pairs = build_prompt_response_pairs(val_data)

print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

train_dataset = OASSTDataset(train_pairs, tokenizer, max_length=SFT_MAX_INPUT_LENGTH)
val_dataset = OASSTDataset(val_pairs, tokenizer, max_length=SFT_MAX_INPUT_LENGTH)

bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

training_args = TrainingArguments(
    auto_find_batch_size=True,
    output_dir=SFT_OUTPUT_DIR,
    dataloader_num_workers=4,
    dataloader_persistent_workers=True,
    dataloader_pin_memory=True,
    eval_strategy="steps",
    eval_accumulation_steps=1,
    learning_rate=SFT_LEARNING_RATE,
    per_device_train_batch_size=SFT_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=SFT_EVAL_BATCH_SIZE,
    gradient_checkpointing=False,
    fp16=True,
    adam_beta1=0.9,
    adam_beta2=0.95,
    gradient_accumulation_steps=SFT_GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=SFT_NUM_TRAIN_EPOCHS,
    optim="adamw_torch_fused",
    warmup_steps=100,
    eval_steps=SFT_EVAL_STEPS,
    save_steps=SFT_SAVE_STEPS,
    load_best_model_at_end=True,
    logging_steps=50,
    report_to="none",
    label_names=["labels"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(SFT_OUTPUT_DIR)

peft_config = PeftConfig.from_pretrained(SFT_OUTPUT_DIR)
base_model = load_causal_lm(
    peft_config.base_model_name_or_path, trust_remote_code=True, device=device
)
model = PeftModel.from_pretrained(base_model, SFT_OUTPUT_DIR)
model = model.to(device)
model.save_pretrained(SFT_OUTPUT_DIR)

merged_model = model.merge_and_unload()
merged_model.save_pretrained(f"{SFT_OUTPUT_DIR}/sft_model")

tokenizer = load_tokenizer(peft_config.base_model_name_or_path, trust_remote_code=True)
tokenizer.save_pretrained(SFT_OUTPUT_DIR)

merged_model.to(device)
merged_model.eval()

test_prompt = "How do I make a cup of tea?"
input_text = f"User: {test_prompt}\nAssistant:"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
output = merged_model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))

print_sample_generations(merged_model, tokenizer, val_pairs)
