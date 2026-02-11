# scripts/prepare_data.py

"""
Download and prepare GSM8K dataset
"""

from datasets import load_dataset
import json
from src.config import Config
import random

def prepare_gsm8k():
    """Download GSM8K and create subset"""
    
    print("📥 Downloading GSM8K dataset...")
    
    # Load dataset from Hugging Face
    dataset = load_dataset("gsm8k", "main")
    
    # Get train split
    train_data = dataset["train"]
    
    # Sample 200 problems
    random.seed(42)
    indices = random.sample(range(len(train_data)), Config.NUM_EVAL_SAMPLES)
    
    subset = []
    for idx in indices:
        example = train_data[idx]
        subset.append({
            "id": f"gsm8k_{idx}",
            "question": example["question"],
            "answer": example["answer"]
        })
    
    # Save to raw data directory
    output_path = Config.RAW_DATA_DIR / "gsm8k_subset.json"
    with open(output_path, "w") as f:
        json.dump(subset, f, indent=2)
    
    print(f"✅ Saved {len(subset)} problems to {output_path}")
    
    # Print sample
    print("\n📝 Sample problem:")
    print(f"Q: {subset[0]['question']}")
    print(f"A: {subset[0]['answer']}")

if __name__ == "__main__":
    prepare_gsm8k()