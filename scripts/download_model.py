# scripts/download_model.py

"""
Download base model from Hugging Face
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import Config
import sys

def download_model():
    """Download Mistral 7B model and tokenizer"""
    
    print(f"📥 Downloading model: {Config.BASE_MODEL}")
    print(f"📍 Saving to: {Config.BASE_MODEL_DIR}")
    
    try:
        # Download tokenizer
        print("\n1️⃣ Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            Config.BASE_MODEL,
            token=Config.HF_TOKEN
        )
        tokenizer.save_pretrained(Config.BASE_MODEL_DIR / "mistral-7b-instruct")
        print("✅ Tokenizer downloaded")
        
        # Download model (will use cache, so won't re-download if exists)
        print("\n2️⃣ Downloading model (this may take 10-20 mins)...")
        model = AutoModelForCausalLM.from_pretrained(
            Config.BASE_MODEL,
            token=Config.HF_TOKEN,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        model.save_pretrained(Config.BASE_MODEL_DIR / "mistral-7b-instruct")
        print("✅ Model downloaded")
        
        print("\n🎉 Download complete!")
        print(f"📂 Model saved at: {Config.BASE_MODEL_DIR / 'mistral-7b-instruct'}")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()