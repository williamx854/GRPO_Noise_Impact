# Investigating the Impact of Reward Noise on GRPO

This repository contains the code to reproduce the experiments on the impact of reward noise (flipping and masking) when fine-tuning the Qwen 2.5 Math 1.5B model with GRPO.

## 1. Setup

### Environment
It is recommended to create a new Conda environment to ensure all dependencies are compatible.

```bash
# Create a new conda environment with compatible dependencies
conda create --name grpo_noise_env python=3.12 pytorch-cuda=12.4 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y

# Activate the environment
conda activate grpo_noise_env

# Install Unsloth and other required libraries
# NOTE: This command is for modern Ampere/Hopper GPUs (A100, H100) with CUDA 12.1.
# Your command may be different depending on your specific GPU and CUDA version.
pip install "unsloth[cu121-ampere-torch230]"
pip install datasets==3.6.0 transformers==4.52.4 trl==0.19.0 peft==0.16.0 accelerate==1.7.0 vllm==0.7.2
```

### Data
1.  Download the MATH dataset from https://huggingface.co/datasets/fdyrd/MATH to data/
2.  Merge all the parts with merge_math_datasets.py
3.  Split the original `train.jsonl` into a 90/10 `train_split.jsonl` and `validation_split.jsonl` with split_math_train_set.py
4.  Rename the original `validation.jsonl` to `test.jsonl`.
5.  Place these three files into the `data/` directory if not already done.

## 2. Running Experiments

The following scripts are located in the `scripts/` directory.

### A. Hyperparameter and Seed Sweep (Baseline)
This script trains the baseline models with the best hyperparameters and various random seeds.

```bash
python scripts/train_baseline_hyperparameter.py \
    --output_dir ./grpo_seed_runs/seed_{SEED_NUMBER} \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --grpo_beta 0.0 \
    --seed {SEED_NUMBER} \
    --num_train_epochs 1
```

### B. Reward Flipping Experiment
This script trains models with noisy rewards by flipping the true reward with a given probability.

```bash
python scripts/train_with_reward_flips.py \
    --output_dir ./grpo_flip_seed_runs/flip_{FLIP_RATE}_seed_{SEED_NUMBER} \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --grpo_beta 0.0 \
    --label_flip_rate {FLIP_RATE} \
    --seed {SEED_NUMBER} \
    --num_train_epochs 1
```

### C. Reward Masking Experiment
This script trains models where the reward is sometimes masked to zero.

```bash
python scripts/train_with_reward_masking.py \
    --output_dir ./grpo_mask_runs/mask_{MASK_RATE}_seed_{SEED_NUMBER} \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --grpo_beta 0.0 \
    --mask_rate {MASK_RATE} \
    --seed {SEED_NUMBER} \
    --num_train_epochs 1
```

## 3. Evaluation

After training is complete, run the corresponding evaluation script to test the models on the `test.jsonl` set and generate a summary file.

```bash
# Evaluate the baseline/seed sweep runs
python scripts/evaluate_selected_hyperparameters_more_seeds.py

# Evaluate the reward flipping runs
python scripts/evaluate_random_flips_rewards.py

# Evaluate the reward masking runs
python scripts/evaluate_masked_rewards.py
```