# src/prompts.py

"""
Prompt templates for reasoning and correction
"""

class PromptTemplates:
    """Centralized prompt templates"""
    
    # Reasoning prompt for student model (Mistral)
    REASONING_PROMPT = """You are a math problem solver. Solve the following problem step by step.

Problem: {question}

Provide your solution in this format:
1. Break down the problem
2. Show each calculation step
3. State the final answer clearly

Your reasoning:"""

    # Correction prompt for teacher model (Groq)
    CORRECTION_PROMPT = """You are an expert math teacher. A student attempted this problem but got it wrong.

Problem: {question}

Student's incorrect attempt:
{student_reasoning}

Correct answer: {correct_answer}

Provide a clear, step-by-step correct solution that:
1. Identifies where the student went wrong
2. Shows the proper reasoning process
3. Arrives at the correct answer

Correct solution:"""

    # Evaluation extraction prompt
    EXTRACT_ANSWER_PROMPT = """Extract only the final numerical answer from this reasoning.

Reasoning: {reasoning}

Return ONLY the number, nothing else."""

    @staticmethod
    def format_reasoning_prompt(question: str) -> str:
        """Format reasoning prompt for student model"""
        return PromptTemplates.REASONING_PROMPT.format(question=question)
    
    @staticmethod
    def format_correction_prompt(question: str, student_reasoning: str, correct_answer: str) -> str:
        """Format correction prompt for teacher model"""
        return PromptTemplates.CORRECTION_PROMPT.format(
            question=question,
            student_reasoning=student_reasoning,
            correct_answer=correct_answer
        )
    
    @staticmethod
    def format_extract_prompt(reasoning: str) -> str:
        """Format answer extraction prompt"""
        return PromptTemplates.EXTRACT_ANSWER_PROMPT.format(reasoning=reasoning)