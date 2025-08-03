"""
Example demonstrating usage with multiple LLM providers.
"""
import os
from dspy_judge.data_loader.dataset_loader import CustomerSupportDatasetLoader
from dspy_judge.llm_caller import (
    OpenAIStructuredOutputCaller, 
    AnthropicStructuredOutputCaller,
    GeminiStructuredOutputCaller
)
from dspy_judge.processor.parallel_processor import ParallelProcessor
from dspy_judge.prompts.pydantic_models import JudgeResponse



def compare_providers():
    """Compare judge responses across different LLM providers."""
    
    # Load a small sample
    loader = CustomerSupportDatasetLoader()
    dataset = loader.load_dataset(split="train")
    sample = loader.get_sample(dataset, n_samples=2)
    
    judge_prompt = """You are a customer service quality judge. 
    Analyze the conversation and determine if the agent satisfied the customer's query."""
    
    providers = []
    
    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        providers.append({
            "name": "OpenAI",
            "caller": OpenAIStructuredOutputCaller(os.getenv("OPENAI_API_KEY")),
            "model": "gpt-4o-mini"
        })
    
    # Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append({
            "name": "Anthropic", 
            "caller": AnthropicStructuredOutputCaller(os.getenv("ANTHROPIC_API_KEY")),
            "model": "claude-3-haiku-20240307"
        })
    
    # Gemini
    if os.getenv("GEMINI_API_KEY"):
        providers.append({
            "name": "Gemini",
            "caller": GeminiStructuredOutputCaller(os.getenv("GEMINI_API_KEY")),
            "model": "gemini-1.5-flash"
        })
    
    if not providers:
        print("No API keys found. Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY")
        return
    
    # Process with each provider
    for provider in providers:
        print(f"\n--- {provider['name']} Results ---")
        
        processor = ParallelProcessor(provider["caller"], max_workers=1)
        results = processor.process_dataset(
            sample,
            system_prompt=judge_prompt,
            model_name=provider["model"],
            response_template=JudgeResponse,
            temperature=0.0
        )
        
        for i, example in enumerate(results):
            if example["llm_response"]:
                response = example["llm_response"]
                print(f"Example {i+1}: {response['satisfied']} - {response['explanation']}")
            
        # Token stats
        stats = processor.get_token_statistics(results)
        print(f"Tokens used: {stats['total_input_tokens']} in, {stats['total_output_tokens']} out")


if __name__ == "__main__":
    compare_providers()