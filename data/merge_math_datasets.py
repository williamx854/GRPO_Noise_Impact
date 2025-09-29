import pandas as pd
import os
import json


def convert_parquet_to_jsonl(base_data_path, topics, split_type, output_dir):
    """
    Converts and merges multiple parquet files from different topics into a single JSONL file.

    Args:
        base_data_path (str): The base directory containing the topic subdirectories
                              (e.g., './data/hendrycks_math').
        topics (list): A list of topic names (e.g., 'algebra', 'geometry').
        split_type (str): The type of split, either 'train' or 'test'.
        output_dir (str): The directory where the final JSONL file will be saved
                          (e.g., 'data/MATH').
    """
    all_data = []

    for topic in topics:
        parquet_file_path = os.path.join(
            base_data_path,
            topic,
            f"{split_type}-00000-of-00001.parquet"
        )
        print(f"Loading parquet from: {parquet_file_path}")
        try:
            df = pd.read_parquet(parquet_file_path)
            all_data.append(df)
        except FileNotFoundError:
            print(f"Warning: File not found for topic '{topic}' at '{parquet_file_path}'. Skipping.")
            continue
        except Exception as e:
            print(f"Error loading {parquet_file_path}: {e}")
            continue

    if not all_data:
        print(f"No data found for split '{split_type}'. Exiting.")
        return

    merged_df = pd.concat(all_data, ignore_index=True)

    # Define the output filename: 'test' split will be 'validation.jsonl','train' split will be 'train.jsonl'
    output_filename = "validation.jsonl" if split_type == "test" else f"{split_type}.jsonl"
    output_jsonl_file = os.path.join(output_dir, output_filename)

    # Prepare data for JSONL
    output_records = []
    for index, row in merged_df.iterrows():
        output_records.append({
            "problem": row['problem'],
            "solution": row['solution']
        })

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_jsonl_file), exist_ok=True)

    print(f"Saving merged data to JSONL: {output_jsonl_file}")
    with open(output_jsonl_file, 'w', encoding='utf-8') as f:
        for record in output_records:
            f.write(json.dumps(record) + '\n')

    print(f"Conversion complete for {split_type} split, saved to {output_jsonl_file}!")

# Script execution
if __name__ == "__main__":
    # This should be the directory that contains 'algebra', 'counting_and_probability', etc.
    base_download_path = "./data/your_path"

    # List of all topics in the MATH dataset
    all_topics = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus"
    ]

    target_output_directory = "data/MATH" 

    # Convert the 'test' split to validation.jsonl
    print("\n--- Processing 'test' split (for validation.jsonl) ---")
    convert_parquet_to_jsonl(base_download_path, all_topics, "test", target_output_directory)

    # Convert the 'train' split to train.jsonl
    print("\n--- Processing 'train' split (for train.jsonl, used in later assignments) ---")
    convert_parquet_to_jsonl(base_download_path, all_topics, "train", target_output_directory)
