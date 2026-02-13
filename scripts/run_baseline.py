# scripts/run_baseline.py

"""
Run baseline evaluation (iteration 0)
"""

from src.config import Config
from src.evaluate import Evaluator

def run_baseline():
    """Run baseline evaluation on iter_0"""
    
    print("=" * 60)
    print("🚀 BASELINE EVALUATION (Iteration 0)")
    print("=" * 60)
    
    # Paths
    data_path = Config.RAW_DATA_DIR / "gsm8k_subset.json"
    output_dir = Config.ITERATIONS_DIR / "iter_0"
    
    # Run evaluation (no adapter = base model)
    evaluator = Evaluator(adapter_path=None)
    metrics = evaluator.evaluate(
        data_path=data_path,
        output_dir=output_dir,
        split_name="iter_0_baseline"
    )
    
    print("\n" + "=" * 60)
    print(f"✅ Baseline metrics saved to {output_dir / 'metrics.json'}")
    print("=" * 60)

if __name__ == "__main__":
    run_baseline()