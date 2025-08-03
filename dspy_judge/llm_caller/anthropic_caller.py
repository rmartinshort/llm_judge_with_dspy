"""
Anthropic LLM caller implementation.
"""
import instructor
from anthropic import Anthropic
from typing import Any, Dict, Type
from pydantic import BaseModel

from .base import LLMStructuredOutputCaller, LLMTextOutputCaller


class AnthropicStructuredOutputCaller(LLMStructuredOutputCaller):
    """Anthropic caller for structured output."""
    
    def set_up_client(self) -> Any:
        return instructor.from_anthropic(Anthropic(api_key=self.api_key))
    
    @staticmethod
    def token_counter(model_output: Dict[str, Any]) -> Dict[str, int]:
        usage_stats = model_output["metadata"].usage
        return {
            "input_tokens": usage_stats.input_tokens,
            "output_tokens": usage_stats.output_tokens
        }


class AnthropicTextOutputCaller(LLMTextOutputCaller):
    """Anthropic caller for text output."""
    
    def set_up_client(self) -> Any:
        return Anthropic(api_key=self.api_key)
    
    def invoke(
        self,
        input_string: str,
        system_prompt: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        try:
            response = self.client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": input_string}]
            )
            return {
                "metadata": response,
                "text": response.content[0].text
            }
        except Exception as e:
            print(f"Error calling {model_name}: {e}")
            return {}
    
    @staticmethod
    def token_counter(model_output: Dict[str, Any]) -> Dict[str, int]:
        usage_stats = model_output["metadata"].usage
        return {
            "input_tokens": usage_stats.input_tokens,
            "output_tokens": usage_stats.output_tokens
        }