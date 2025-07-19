from typing import Any

from accelerate import PartialState
from datasets import load_dataset
from peft import PeftModel
from trl import ModelConfig, PPOConfig, PPOTrainer, get_peft_config
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from config import (PPO_BASE_MODEL, PPO_DATASET_NAME, PPO_DATASET_SPLIT,
                    PPO_EVAL_SAMPLES, PPO_GRADIENT_ACCUMULATION_STEPS,
                    PPO_LEARNING_RATE, PPO_NUM_PPO_EPOCHS, PPO_OUTPUT_DIR,
                    PPO_PER_DEVICE_TRAIN_BATCH_SIZE, PPO_PROMPT_COLUMN,
                    PPO_REWARD_MODEL_PATH, PPO_SFT_MODEL_PATH,
                    PPO_TOTAL_EPISODES)
from model import load_causal_lm, load_sequence_classifier, load_tokenizer


def prepare_dataset(dataset: Any, tokenizer: Any) -> Any:
    def tokenize(element: dict[str, Any]) -> dict[str, Any]:
        outputs = tokenizer(
            element[PPO_PROMPT_COLUMN],
            padding=False,
        )
        return {"input_ids": outputs["input_ids"]}

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=1,
    )


def main() -> None:
    # Model and tokenizer loading
    tokenizer = load_tokenizer(
        PPO_SFT_MODEL_PATH, padding_side="left", trust_remote_code=True
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    policy = load_causal_lm(PPO_SFT_MODEL_PATH, trust_remote_code=True)
    model_args = ModelConfig(
        model_name_or_path=PPO_SFT_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    peft_config = get_peft_config(model_args)
    ref_policy = None
    if peft_config is None:
        ref_policy = load_causal_lm(PPO_SFT_MODEL_PATH, trust_remote_code=True)

    reward_model = load_sequence_classifier(
        PPO_REWARD_MODEL_PATH, trust_remote_code=True, num_labels=1
    )
    value_model = load_sequence_classifier(
        PPO_REWARD_MODEL_PATH, trust_remote_code=True, num_labels=1
    )

    # Data loading
    dataset = load_dataset(PPO_DATASET_NAME, split=PPO_DATASET_SPLIT)
    train_dataset = dataset.select(range(len(dataset) - PPO_EVAL_SAMPLES))
    eval_dataset = dataset.select(range(len(dataset) - PPO_EVAL_SAMPLES, len(dataset)))

    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    # PPO config
    ppo_args = PPOConfig(
        output_dir=PPO_OUTPUT_DIR,
        per_device_train_batch_size=PPO_PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=PPO_GRADIENT_ACCUMULATION_STEPS,
        learning_rate=PPO_LEARNING_RATE,
        total_episodes=PPO_TOTAL_EPISODES,
        num_ppo_epochs=PPO_NUM_PPO_EPOCHS,
        logging_steps=100,
        fp16=True,
        bf16=False,
        batch_size=1,
        mini_batch_size=1,
        whiten_rewards=False,
        kl_coef=0.01,
        cliprange=0.2,
        vf_coef=0.1,
        cliprange_value=0.2,
        gamma=1.0,
        lam=0.95,
        temperature=1.0,
        exp_name="ppo_config",
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
    )

    trainer = PPOTrainer(
        args=ppo_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(PPO_OUTPUT_DIR)
    trainer.generate_completions()
    print("PPO RLHF Qwen3 training complete! Model saved to:", PPO_OUTPUT_DIR)

    base_model = load_causal_lm(PPO_BASE_MODEL, trust_remote_code=True)
    final_peft = PeftModel.from_pretrained(base_model, PPO_OUTPUT_DIR)
    merged = final_peft.merge_and_unload()
    merged.save_pretrained(PPO_OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(PPO_OUTPUT_DIR)
    print("Merged full model (for direct inference) saved to ./checkpoints/ppo_model")


if __name__ == "__main__":
    main()
