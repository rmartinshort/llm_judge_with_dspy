"""
Gemini LLM caller implementation.
"""

import instructor
import google.generativeai as genai
from typing import Any, Dict, Type
from pydantic import BaseModel

from .base import LLMStructuredOutputCaller, LLMTextOutputCaller


class GeminiStructuredOutputCaller(LLMStructuredOutputCaller):
    """Gemini caller for structured output."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model
        self.client = self.set_up_client()

    def set_up_client(self) -> Any:
        genai.configure(api_key=self.api_key)
        client = instructor.from_gemini(
            client=genai.GenerativeModel(model_name=f"models/{self.model_name}"),
            mode=instructor.Mode.GEMINI_JSON,
        )
        return client

    def invoke(
        self,
        input_string: str,
        system_prompt: str,
        response_template: Type[BaseModel],
        model_name: str = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        inputs = self.craft_input(input_string, system_prompt)
        try:
            res, completions = self.client.messages.create_with_completion(
                response_model=response_template,
                messages=inputs,
                generation_config={"temperature": temperature},
            )
            return {"metadata": completions, "model_dump": res.model_dump()}
        except Exception as e:
            print(f"Error calling {model_name or self.model_name}: {e}")
            return {}

    @staticmethod
    def token_counter(model_output: Dict[str, Any]) -> Dict[str, int]:
        usage_stats = model_output["metadata"].usage_metadata
        return {
            "input_tokens": usage_stats.prompt_token_count,
            "output_tokens": usage_stats.candidates_token_count,
        }


class GeminiTextOutputCaller(LLMTextOutputCaller):
    """Gemini caller for text output."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model
        self.client = self.set_up_client()

    def set_up_client(self) -> Any:
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(model_name=f"models/{self.model_name}")

    def invoke(
        self,
        input_string: str,
        system_prompt: str,
        model_name: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        try:
            prompt = f"System: {system_prompt}\n\nUser: {input_string}"
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature, max_output_tokens=max_tokens
                ),
            )
            return {"metadata": response, "text": response.text}
        except Exception as e:
            print(f"Error calling {model_name or self.model_name}: {e}")
            return {}

    @staticmethod
    def token_counter(model_output: Dict[str, Any]) -> Dict[str, int]:
        usage_stats = model_output["metadata"].usage_metadata
        return {
            "input_tokens": usage_stats.prompt_token_count,
            "output_tokens": usage_stats.candidates_token_count,
        }
