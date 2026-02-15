# scripts/download_model.py

"""
Download base model from Hugging Face
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import Config
import sys

def download_model():
    """Download base model and tokenizer"""
    
    model_name = Config.BASE_MODEL.split("/")[-1]
    save_path = Config.BASE_MODEL_DIR / model_name
    
    print(f"📥 Downloading model: {Config.BASE_MODEL}")
    print(f"📍 Saving to: {save_path}")
    
    try:
        # Download tokenizer
        print("\n1️⃣ Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            Config.BASE_MODEL,
            token=Config.HF_TOKEN,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(save_path)
        print("✅ Tokenizer downloaded")
        
        # Download model (will use cache, so won't re-download if exists)
        print("\n2️⃣ Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            Config.BASE_MODEL,
            token=Config.HF_TOKEN,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.save_pretrained(save_path)
        print("✅ Model downloaded")
        
        print("\n🎉 Download complete!")
        print(f"📂 Model saved at: {save_path}")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()