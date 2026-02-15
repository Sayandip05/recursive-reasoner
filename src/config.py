# src/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Central configuration for the project"""
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    # Model settings
    BASE_MODEL = os.getenv("BASE_MODEL", "microsoft/Phi-3-mini-4k-instruct")
    DEVICE = os.getenv("DEVICE", "cuda")
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", 2048))
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_DIR = PROJECT_ROOT / "models"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    
    # Data paths
    RAW_DATA_DIR = DATA_DIR / "raw"
    ITERATIONS_DIR = DATA_DIR / "iterations"
    PROCESSED_DIR = DATA_DIR / "processed"
    
    # Model paths
    BASE_MODEL_DIR = MODEL_DIR / "base"
    ADAPTERS_DIR = MODEL_DIR / "adapters"
    
    # Output paths
    LOGS_DIR = OUTPUT_DIR / "logs"
    METRICS_DIR = OUTPUT_DIR / "metrics"
    
    # Fine-tuning hyperparameters
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    NUM_EPOCHS = 3
    
    # Evaluation settings
    NUM_EVAL_SAMPLES = 200
    NUM_TRAIN_SAMPLES = 150
    
    @classmethod
    def ensure_dirs(cls):
        """Create all necessary directories"""
        for dir_path in [
            cls.DATA_DIR, cls.RAW_DATA_DIR, cls.ITERATIONS_DIR, cls.PROCESSED_DIR,
            cls.MODEL_DIR, cls.BASE_MODEL_DIR, cls.ADAPTERS_DIR,
            cls.OUTPUT_DIR, cls.LOGS_DIR, cls.METRICS_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        print("✅ All directories created")

# Create directories on import
Config.ensure_dirs()