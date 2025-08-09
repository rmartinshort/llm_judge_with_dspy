"""
Base classes and data structures for LLM callers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type
from pydantic import BaseModel


class LLMCallerBase(ABC):
    """Abstract base class for LLM callers."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = self.set_up_client()

    @abstractmethod
    def set_up_client(self) -> Any:
        """Set up the client for the specific LLM."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def token_counter(model_output: Dict[str, Any]) -> Dict[str, int]:
        """Count input and output tokens from model response."""
        raise NotImplementedError

    @staticmethod
    def craft_input(string_input: str, system_prompt: str) -> List[Dict[str, str]]:
        """Craft input messages for the LLM."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": string_input},
        ]


class LLMStructuredOutputCaller(LLMCallerBase):
    """Base class for structured output LLM callers."""

    def invoke(
        self,
        input_string: str,
        system_prompt: str,
        response_template: Type[BaseModel],
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Invoke the LLM to generate structured output.

        Args:
            input_string: Input text to process
            system_prompt: System prompt string
            response_template: Pydantic model for structured response
            model_name: Name of the model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with metadata and model response
        """
        inputs = self.craft_input(input_string, system_prompt)
        try:
            res, completions = self.client.chat.completions.create_with_completion(
                model=model_name,
                response_model=response_template,
                temperature=temperature,
                messages=inputs,
                max_tokens=max_tokens,
            )
            return {"metadata": completions, "model_dump": res.model_dump()}
        except Exception as e:
            print(f"Error calling {model_name}: {e}")
            return {}


class LLMTextOutputCaller(LLMCallerBase):
    """Base class for text output LLM callers."""

    @abstractmethod
    def invoke(
        self,
        input_string: str,
        system_prompt: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Invoke the LLM to generate text output.

        Args:
            input_string: Input text to process
            system_prompt: System prompt string
            model_name: Name of the model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with metadata and text response
        """
        raise NotImplementedError
