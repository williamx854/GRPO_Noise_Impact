import torch
from vllm import LLM, SamplingParams
import os
import json
import argparse
from datasets import Dataset
from glob import glob
from tqdm import tqdm

from drgrpo_grader import r1_zero_reward_fn

"""
This script evaluates the masked reward sweep accuracy of all the merged models in their output directory.
"""

def main():
    parser = argparse.ArgumentParser(description="Evaluate all GRPO reward masking sweep checkpoints.")
    parser.add_argument("--runs_dir", type=str, default="Your_Runs_Directory", help="Root directory of the mask/seed runs.")
    parser.add_argument("--checkpoint_name", type=str, default="final_merged_checkpoint", help="Name of the final checkpoint folder.")
    parser.add_argument("--test_path", type=str, default="data/MATH/test.jsonl")
    parser.add_argument("--prompt_path", type=str, default="prompts/r1_zero.prompt")
    parser.add_argument("--results_file", type=str, default="mask_sweep_summary.jsonl", help="File to save all results.")
    args = parser.parse_args()

    # Load and Prepare Test Data
    print(f"Loading test data from {args.test_path}")
    test_dataset = Dataset.from_json(args.test_path)
    
    with open(args.prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    prompts = [prompt_template.format(question=ex["problem"]) for ex in test_dataset]
    solutions = [ex["solution"] for ex in test_dataset]

    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )

    # Find and Evaluate All Checkpoints
    checkpoint_paths = glob(f"{args.runs_dir}/*/{args.checkpoint_name}")
    print(f"Found {len(checkpoint_paths)} trained model checkpoints to evaluate.")

    all_results = []
    if os.path.exists(args.results_file):
        os.remove(args.results_file)

    for checkpoint_path in tqdm(sorted(checkpoint_paths), desc="Evaluating checkpoints"):
        print(f"\n--- Loading and evaluating merged model: {checkpoint_path} ---")
        
        llm = LLM(model=checkpoint_path, trust_remote_code=True, dtype=torch.bfloat16)
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

        run_name = os.path.basename(os.path.dirname(checkpoint_path))
        result_data = { "run_name": run_name, "accuracy": accuracy }
        all_results.append(result_data)
        
        with open(args.results_file, 'a') as f:
            f.write(json.dumps(result_data) + '\n')

        del llm
        torch.cuda.empty_cache()

    # Print Final Summary
    print("\n\n--- Masking Sweep Evaluation Summary ---")
    sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
    for result in sorted_results:
        print(f"Run: {result['run_name']:<25} | Accuracy: {result['accuracy']:.4f}")

if __name__ == "__main__":
    main()
