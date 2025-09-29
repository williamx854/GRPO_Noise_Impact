import torch
import random
import argparse
import os
from functools import partial
from typing import List

from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import set_seed
from trl import GRPOConfig, GRPOTrainer
from huggingface_hub import login

login(token="Your_HuggingFace_Token")
from drgrpo_grader import r1_zero_reward_fn

"""
This script trains a Qwen2.5-Math-1.5B model on the MATH train dataset using Unsloth
with a reward function that masks the reward with a given probability.
"""

# Data Formatting Function 
def make_conversation_and_tokenize(example, tokenizer, max_prompt_length):
    formatted_prompt = SYSTEM_PROMPT_TEMPLATE.format(question=example["problem"])
    tokenized_prompt = tokenizer(
        formatted_prompt,
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=False
    )
    example["input_ids"] = tokenized_prompt["input_ids"]
    example["attention_mask"] = tokenized_prompt["attention_mask"]
    example["solution"] = example["solution"]
    example["prompt"] = formatted_prompt
    return example

# Reward Function Wrapper with MASKING logic
def trl_reward_fn(completions: List[str], prompts: List[str], mask_rate: float, **kwargs) -> List[float]:
    solutions = kwargs['solution']
    rewards_list = []
    
    for i, generated_text in enumerate(completions):
        ground_truth = solutions[i]
        
        # Check if we should mask the reward for this completion
        if random.random() < mask_rate:
            # Masked: return a reward of 0 regardless of correctness
            final_reward = 0.0
        else:
            # Not masked: return the true reward
            rewards_dict = r1_zero_reward_fn(generated_text, ground_truth)
            final_reward = rewards_dict.get("reward", 0.0)
            
        rewards_list.append(final_reward)
        
    return rewards_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_rate", type=float, default=0.0, help="Probability to mask the reward to 0.")
    parser.add_argument("--output_dir", type=str, default="./GRPO_Unsloth_run", help="Output directory.")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--grpo_beta", type=float, default=0.1)
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval.")
    parser.add_argument("--eval_interval", type=int, default=50, help="Evaluation interval.")
    
    args = parser.parse_args()
    
    set_seed(args.seed)

    # Model, Tokenizer, Data Loading
    print("Loading model with Unsloth's FastLanguageModel...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-Math-1.5B",
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=None,        
        device_map="auto",
    )

    # LoRA Configuration with Unsloth
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth", 
        random_state=args.seed,
    )

    # Load Datasets and Prompt
    train_path = "data/MATH_split/train_split.jsonl"
    train_dataset = Dataset.from_json(train_path)

    R1_ZERO_PROMPT_PATH = "prompts/r1_zero.prompt"
    with open(R1_ZERO_PROMPT_PATH, 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT_TEMPLATE = f.read()

    # Tokenize Datasets
    tokenization_func = partial(make_conversation_and_tokenize, tokenizer=tokenizer, max_prompt_length=512)
    train_dataset = train_dataset.map(tokenization_func, batched=False)
    
    # Configure Training
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        optim = "adamw_8bit",
        bf16=True,
        remove_unused_columns=False,
        report_to="wandb",
        beta=args.grpo_beta,
        max_prompt_length=512,
        max_completion_length=512,
        num_generations=8,
        temperature=1.0,
        top_p=1.0,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        per_device_eval_batch_size=8
    )

    # Initialize Trainer
    reward_function_with_masking = lambda completions, prompts, **kwargs: trl_reward_fn(
        completions, prompts, mask_rate=args.mask_rate, **kwargs
    )
    
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[reward_function_with_masking],
        tokenizer=tokenizer,
    )
    
    print(f"Starting GRPO training with Unsloth. Output will be saved to: {training_args.output_dir}")
    trainer.train()
    
    # Save Final Model
    print("Training finished. Saving final merged model.")
    trainer.model.save_pretrained_merged(
        os.path.join(args.output_dir, "final_merged_checkpoint"), 
        tokenizer, 
        save_method="merged_16bit"
    )
