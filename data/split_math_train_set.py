from datasets import Dataset
import os

# Define Paths
full_train_path = "./data/MATH/train.jsonl"
output_dir = "data/MATH_split"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

new_train_path = os.path.join(output_dir, "train_split.jsonl")
new_validation_path = os.path.join(output_dir, "validation_split.jsonl")


# Load and Split the Data
print(f"Loading original training set from: {full_train_path}")
full_train_dataset = Dataset.from_json(full_train_path)
print(f"Original training set size: {len(full_train_dataset)}")

# Split the dataset (90% train, 10% validation)
split_dataset = full_train_dataset.train_test_split(test_size=0.1, seed=42)

new_train_dataset = split_dataset['train']
new_validation_dataset = split_dataset['test']

print(f"New training set size: {len(new_train_dataset)}")
print(f"New validation set size: {len(new_validation_dataset)}")


# Save the New Datasets to Disk
print(f"Saving new training set to: {new_train_path}")
new_train_dataset.to_json(new_train_path)

print(f"Saving new validation set to: {new_validation_path}")
new_validation_dataset.to_json(new_validation_path)

print("\nSplitting and saving complete.")