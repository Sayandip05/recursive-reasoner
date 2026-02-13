# src/utils.py

"""
Utility functions
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any

def extract_answer(text: str) -> str:
    """
    Extract numerical answer from reasoning text
    Handles formats like: "#### 42", "The answer is 42", "= 42"
    """
    # Try GSM8K format first (#### answer)
    gsm_pattern = r'####\s*([0-9,.]+)'
    match = re.search(gsm_pattern, text)
    if match:
        return match.group(1).replace(',', '')
    
    # Try "answer is X" pattern
    answer_pattern = r'(?:answer is|final answer is|answer:|equals?)\s*([0-9,.]+)'
    match = re.search(answer_pattern, text.lower())
    if match:
        return match.group(1).replace(',', '')
    
    # Try last number in text
    numbers = re.findall(r'[0-9,.]+', text)
    if numbers:
        return numbers[-1].replace(',', '')
    
    return ""

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    # Remove commas, spaces, leading zeros
    answer = str(answer).replace(',', '').replace(' ', '').strip()
    answer = answer.lstrip('0') or '0'
    return answer

def answers_match(pred: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer"""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    return pred_norm == gold_norm

def save_json(data: Any, filepath: Path):
    """Save data to JSON file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved to {filepath}")

def load_json(filepath: Path) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)