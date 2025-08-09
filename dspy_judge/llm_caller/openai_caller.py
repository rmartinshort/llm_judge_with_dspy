"""
OpenAI LLM caller implementation.
"""

import instructor
from openai import OpenAI
from typing import Any, Dict

from .base import LLMStructuredOutputCaller, LLMTextOutputCaller


class OpenAIStructuredOutputCaller(LLMStructuredOutputCaller):
    """OpenAI caller for structured output."""

    def set_up_client(self) -> Any:
        client = instructor.from_openai(
            OpenAI(api_key=self.api_key), mode=instructor.Mode.TOOLS_STRICT
        )
        return client

    @staticmethod
    def token_counter(model_output: Dict[str, Any]) -> Dict[str, int]:
        usage_stats = model_output["metadata"].usage
        return {
            "input_tokens": usage_stats.prompt_tokens,
            "output_tokens": usage_stats.completion_tokens,
        }


class OpenAITextOutputCaller(LLMTextOutputCaller):
    """OpenAI caller for text output."""

    def set_up_client(self) -> Any:
        return OpenAI(api_key=self.api_key)

    def invoke(
        self,
        input_string: str,
        system_prompt: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        inputs = self.craft_input(input_string, system_prompt)
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=inputs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return {"metadata": response, "text": response.choices[0].message.content}
        except Exception as e:
            print(f"Error calling {model_name}: {e}")
            return {}

    @staticmethod
    def token_counter(model_output: Dict[str, Any]) -> Dict[str, int]:
        usage_stats = model_output["metadata"].usage
        return {
            "input_tokens": usage_stats.prompt_tokens,
            "output_tokens": usage_stats.completion_tokens,
        }
