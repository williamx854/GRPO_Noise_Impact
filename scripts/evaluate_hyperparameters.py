import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import os
import json
import argparse
from datasets import Dataset
from glob import glob
from tqdm import tqdm

# Reward Function
from drgrpo_grader import r1_zero_reward_fn

"""
This script evaluates the hyperparameter sweep accuracy of all the merged models in the runs_dir (your directory).
"""

def evaluate_run(llm, sampling_params, prompts, solutions, lora_path):
    """Evaluates a single LoRA adapter and returns the accuracy."""
    print(f"\n--- Evaluating adapter: {lora_path} ---")
    
    # Create a unique LoRARequest for this specific adapter
    lora_request = LoRARequest(
        lora_name="grpo_adapter",
        lora_int_id=1,
        lora_local_path=lora_path
    )
    
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    
    total_correct = 0
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        ground_truth = solutions[i]
        rewards = r1_zero_reward_fn(generated_text, ground_truth)
        if rewards.get("reward", 0.0) == 1.0:
            total_correct += 1
            
    accuracy = total_correct / len(prompts) if prompts else 0
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate all GRPO hyperparameter sweep checkpoints.")
    parser.add_argument("--runs_dir", type=str, default="Your_Runs_Directory", help="Root directory of the training runs.")
    parser.add_argument("--checkpoint_name", type=str, default="final_merged_checkpoint", help="Name of the final checkpoint folder.")
    parser.add_argument("--validation_path", type=str, default="data/MATH_split/validation_split.jsonl")
    parser.add_argument("--prompt_path", type=str, default="prompts/r1_zero.prompt")
    parser.add_argument("--results_file", type=str, default="sweep_results_summary.jsonl", help="File to save all results.")
    args = parser.parse_args()

    # Load and Prepare Validation Data
    print(f"Loading validation data from {args.validation_path}")
    validation_dataset = Dataset.from_json(args.validation_path)
    
    with open(args.prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    prompts = [prompt_template.format(question=ex["problem"]) for ex in validation_dataset]
    solutions = [ex["solution"] for ex in validation_dataset]

    # Use temperature 0 for deterministic greedy decoding
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )

    # Find and Evaluate All MERGED Checkpoints
    checkpoint_paths = glob(f"{args.runs_dir}/*/{args.checkpoint_name}")
    print(f"Found {len(checkpoint_paths)} trained model checkpoints to evaluate.")

    all_results = []
    if os.path.exists(args.results_file):
        os.remove(args.results_file)

    for checkpoint_path in tqdm(sorted(checkpoint_paths), desc="Evaluating checkpoints"):
        print(f"\n--- Loading and evaluating merged model: {checkpoint_path} ---")
        
        # Load each merged model directly.
        llm = LLM(
            model=checkpoint_path,
            trust_remote_code=True,
            dtype=torch.bfloat16
        )
        
        # Evaluate this specific model
        outputs = llm.generate(prompts, sampling_params)
    
        total_correct = 0
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            ground_truth = solutions[i]
            rewards = r1_zero_reward_fn(generated_text, ground_truth)
            if rewards.get("reward", 0.0) == 1.0:
                total_correct += 1
        
        accuracy = total_correct / len(prompts) if prompts else 0
        print(f"Accuracy: {accuracy:.4f}")

        # Extract run name and save results
        run_name = os.path.basename(os.path.dirname(checkpoint_path))
        result_data = {
            "run_name": run_name,
            "accuracy": accuracy,
            "checkpoint_path": checkpoint_path
        }
        all_results.append(result_data)
        
        with open(args.results_file, 'a') as f:
            f.write(json.dumps(result_data) + '\n')

        # Clear the VLLM model from memory to load the next one
        del llm
        torch.cuda.empty_cache()

    # Print Final Summary
    print("\n\n--- Evaluation Summary ---")
    sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
    for result in sorted_results:
        print(f"Run: {result['run_name']:<40} | Accuracy: {result['accuracy']:.4f}")

if __name__ == "__main__":
    main()
