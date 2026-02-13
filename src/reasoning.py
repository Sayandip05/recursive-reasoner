# src/reasoning.py

"""
LLM reasoning engine - handles inference with base model and adapters
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Optional
from src.config import Config
from src.prompts import PromptTemplates

class ReasoningEngine:
    """Handles LLM inference for reasoning"""
    
    def __init__(self, adapter_path: Optional[str] = None):
        """
        Initialize reasoning engine
        
        Args:
            adapter_path: Path to LoRA adapter (None for base model)
        """
        print("🔧 Loading model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.BASE_MODEL,
            token=Config.HF_TOKEN
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.BASE_MODEL,
            token=Config.HF_TOKEN,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Load adapter if provided
        if adapter_path:
            print(f"📎 Loading adapter from {adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path
            )
        
        self.model.eval()
        print("✅ Model loaded")
    
    def generate_reasoning(self, question: str, max_new_tokens: int = 512) -> str:
        """
        Generate step-by-step reasoning for a question
        
        Args:
            question: Math problem to solve
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated reasoning text
        """
        # Format prompt
        prompt = PromptTemplates.format_reasoning_prompt(question)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=Config.MAX_LENGTH
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove prompt)
        reasoning = generated_text[len(prompt):].strip()
        
        return reasoning
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()