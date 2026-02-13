# src/evaluate.py

"""
Evaluation logic - run model on dataset and compute metrics
"""

import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
from src.config import Config
from src.reasoning import ReasoningEngine
from src.utils import extract_answer, answers_match, save_json, load_json

class Evaluator:
    """Handles model evaluation"""
    
    def __init__(self, adapter_path: Optional[str] = None):
        """
        Initialize evaluator
        
        Args:
            adapter_path: Path to LoRA adapter (None for base model)
        """
        self.engine = ReasoningEngine(adapter_path)
        self.adapter_path = adapter_path
    
    def evaluate(self, data_path: Path, output_dir: Path, split_name: str = "eval") -> Dict:
        """
        Run evaluation on dataset
        
        Args:
            data_path: Path to JSON data file
            output_dir: Directory to save results
            split_name: Name of this evaluation run
            
        Returns:
            Dictionary with metrics
        """
        print(f"\n📊 Running evaluation: {split_name}")
        print(f"📂 Data: {data_path}")
        print(f"💾 Output: {output_dir}")
        
        # Load data
        data = load_json(data_path)
        print(f"📝 Loaded {len(data)} problems")
        
        # Run inference
        results = []
        correct = 0
        
        for item in tqdm(data, desc="Evaluating"):
            question = item["question"]
            gold_answer = extract_answer(item["answer"])
            
            # Generate reasoning
            reasoning = self.engine.generate_reasoning(question)
            
            # Extract predicted answer
            pred_answer = extract_answer(reasoning)
            
            # Check correctness
            is_correct = answers_match(pred_answer, gold_answer)
            if is_correct:
                correct += 1
            
            # Store result
            results.append({
                "id": item["id"],
                "question": question,
                "gold_answer": gold_answer,
                "reasoning": reasoning,
                "pred_answer": pred_answer,
                "correct": is_correct
            })
        
        # Compute metrics
        accuracy = correct / len(data) if data else 0
        metrics = {
            "split": split_name,
            "total": len(data),
            "correct": correct,
            "accuracy": round(accuracy * 100, 2),
            "adapter": str(self.adapter_path) if self.adapter_path else "base_model"
        }
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(results, output_dir / "predictions.json")
        save_json(metrics, output_dir / "metrics.json")
        
        # Print summary
        print(f"\n✅ Evaluation complete!")
        print(f"📈 Accuracy: {metrics['accuracy']}% ({correct}/{len(data)})")
        
        return metrics