from vllm import LLM, SamplingParams
from typing import List, Callable, Dict
import torch
import json
import os
from drgrpo_grader import r1_zero_reward_fn

"""
This script evaluates the performance of a Qwen2.5-Math-1.5B model on the MATH test dataset.
It uses the r1_zero_reward_fn to evaluate the model's accuracy.
"""

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    output_filepath: str 
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and save results to jsonl file.
    """
    print(f"Generating responses for {len(prompts)} prompts...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    print("Generation complete. Evaluating...")

    results = []
    total_correct = 0
    format_correct_answer_incorrect = 0
    format_incorrect = 0

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        ground_truth = ground_truths[i]

        # Calculate rewards
        rewards = reward_fn(generated_text, ground_truth)
        format_reward = rewards.get("format_reward", 0)
        answer_reward = rewards.get("answer_reward", 0)
        total_reward = rewards.get("reward", 0)

        if format_reward == 1 and answer_reward == 1:
            total_correct += 1
        elif format_reward == 1 and answer_reward == 0:
            format_correct_answer_incorrect += 1
        elif format_reward == 0:
            format_incorrect += 1

        results.append({
            "problem": prompt,
            "generated_response": generated_text,
            "ground_truth": ground_truth,
            "rewards": rewards
        })

    accuracy = total_correct / len(prompts) if len(prompts) > 0 else 0

    print(f"\n--- Evaluation Summary ---")
    print(f"Total examples: {len(prompts)}")
    print(f"Correct (format=1, answer=1): {total_correct}")
    print(f"Format correct, answer incorrect (format=1, answer=0): {format_correct_answer_incorrect}")
    print(f"Format incorrect (format=0, answer=0): {format_incorrect}")
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Save results to jsonl file
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    print(f"Detailed results saved to: {output_filepath}")


if __name__ == "__main__":
    # 1. Load the dataset
    validation_data_path = "data/MATH/test.jsonl"
    problems = []
    solutions = []
    with open(validation_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            problems.append(entry["problem"])
            solutions.append(entry["solution"])

    # 2. Prepare prompts
    with open("prompts/r1_zero.prompt", 'r', encoding='utf-8') as f:
        r1_zero_prompt_template = f.read()

    formatted_prompts = [r1_zero_prompt_template.format(question=p) for p in problems]

    # 3. Initialize vLLM
    vllm_model = LLM(
        model="Qwen/Qwen2.5-Math-1.5B",
        dtype=torch.bfloat16, 
        trust_remote_code=True
    )

    # 4. Define sampling parameters
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # 5. Run the evaluation
    evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=formatted_prompts,
        ground_truths=solutions,
        eval_sampling_params=eval_sampling_params,
        output_filepath="results/zero_shot_baseline_results.jsonl"
    )
