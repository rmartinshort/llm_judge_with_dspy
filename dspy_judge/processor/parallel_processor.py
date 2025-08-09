"""
Parallel processor for running LLM calls on datasets with rate limiting and backoff.
"""

import backoff
from typing import Any, Dict, Optional
from datasets import Dataset
from tqdm import tqdm
import json

from ..llm_caller.base import LLMCallerBase
from ..llm_caller import (
    OpenAIStructuredOutputCaller,
    OpenAITextOutputCaller,
    AnthropicStructuredOutputCaller,
    AnthropicTextOutputCaller,
    GeminiStructuredOutputCaller,
    GeminiTextOutputCaller,
)
import logging

# Silence DSPy warnings
logging.getLogger("dspy.adapters.json_adapter").setLevel(logging.ERROR)


# Global configuration for multiprocessing - avoid pickling issues
_GLOBAL_CONFIG = {}


def _serialize_metadata(metadata: Any) -> Dict[str, Any]:
    """
    Convert complex metadata objects to Arrow-serializable format.

    Args:
        metadata: Raw metadata object from LLM response

    Returns:
        Serializable dictionary representation
    """
    if metadata is None:
        return None

    try:
        # First, try to use model_dump if it's a Pydantic model
        if hasattr(metadata, "model_dump"):
            return metadata.model_dump()

        # For OpenAI ChatCompletion objects, manually extract key fields
        if hasattr(metadata, "id") and hasattr(metadata, "usage"):
            return {
                "id": metadata.id,
                "model": getattr(metadata, "model", None),
                "created": getattr(metadata, "created", None),
                "usage": {
                    "prompt_tokens": metadata.usage.prompt_tokens
                    if metadata.usage
                    else 0,
                    "completion_tokens": metadata.usage.completion_tokens
                    if metadata.usage
                    else 0,
                    "total_tokens": metadata.usage.total_tokens
                    if metadata.usage
                    else 0,
                },
                "finish_reason": metadata.choices[0].finish_reason
                if metadata.choices
                else None,
                "object": getattr(metadata, "object", None),
            }

        # Try converting to dict
        if hasattr(metadata, "__dict__"):
            result = {}
            for key, value in metadata.__dict__.items():
                try:
                    # Test if value is JSON serializable
                    json.dumps(value)
                    result[key] = value
                except (TypeError, ValueError):
                    # If not serializable, convert to string
                    result[key] = str(value)
            return result

        # Fallback to string representation
        return {"raw": str(metadata)}

    except Exception as e:
        # Ultimate fallback - return error info
        return {
            "error": f"Failed to serialize metadata: {str(e)}",
            "type": str(type(metadata)),
        }


def _initialize_worker(caller_type: str, api_key: str, model: Optional[str] = None):
    """Initialize worker process with LLM caller."""
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = {"caller_type": caller_type, "api_key": api_key, "model": model}


def _initialize_worker_dspy_from_signature(
    model_name: str,
    api_key: str,
    signature_docstring: str,
    signature_fields: Dict[str, Any],
    demos: list,
    temperature: float = 0,
):
    """Initialize worker process by recreating DSPy module from signature info and demos."""
    global _GLOBAL_CONFIG

    # Import dspy here to avoid import issues in multiprocessing
    import dspy

    # Configure DSPy with LM and adapter
    model = dspy.LM(model_name, api_key=api_key, cache=False, temperature=temperature)
    adapter = dspy.JSONAdapter()

    # Configure dspy
    dspy.configure(lm=model, track_usage=True, adapter=adapter)

    # Create the signature class dynamically using type()
    class_dict = {"__doc__": signature_docstring}
    annotations = {}

    # Add fields with proper annotations
    for field_name, field_info in signature_fields.items():
        # Convert annotation string back to type
        annotation_str = field_info["annotation_str"]
        if isinstance(annotation_str, str):
            annotation = str
        elif isinstance(annotation_str, bool):
            annotation = bool
        elif isinstance(annotation_str, int):
            annotation = int
        elif isinstance(annotation_str, float):
            annotation = float
        else:
            annotation = str  # fallback to str

        annotations[field_name] = annotation

        # Create the appropriate field
        if field_info["field_type"] == "input":
            field = dspy.InputField(desc=field_info["desc"])
        else:  # output field
            field = dspy.OutputField(desc=field_info["desc"])

        class_dict[field_name] = field

    # Set the annotations
    class_dict["__annotations__"] = annotations

    # Create the dynamic signature class
    DynamicSignature = type("DynamicSignature", (dspy.Signature,), class_dict)

    # Create the DSPy module
    dspy_module = dspy.ChainOfThought(DynamicSignature)

    # Set the demos on the recreated module
    dspy_module.predictors()[0].demos = demos

    _GLOBAL_CONFIG = {"dspy_module": dspy_module}


@backoff.on_exception(backoff.expo, Exception, max_tries=3, base=2, max_value=60)
def _process_single_example(args: tuple) -> Dict[str, Any]:
    """
    Process a single example with LLM call. This function is pickle-able for multiprocessing.
    """
    (
        example,
        system_prompt,
        model_name,
        input_field,
        temperature,
        max_tokens,
        response_template,
    ) = args

    global _GLOBAL_CONFIG
    caller_type = _GLOBAL_CONFIG["caller_type"]
    api_key = _GLOBAL_CONFIG["api_key"]
    model = _GLOBAL_CONFIG.get("model")

    # Recreate the LLM caller from config
    if caller_type == "openai_structured":
        llm_caller = OpenAIStructuredOutputCaller(api_key)
    elif caller_type == "openai_text":
        llm_caller = OpenAITextOutputCaller(api_key)
    elif caller_type == "anthropic_structured":
        llm_caller = AnthropicStructuredOutputCaller(api_key)
    elif caller_type == "anthropic_text":
        llm_caller = AnthropicTextOutputCaller(api_key)
    elif caller_type == "gemini_structured":
        model = model or "gemini-1.5-flash"
        llm_caller = GeminiStructuredOutputCaller(api_key, model)
    elif caller_type == "gemini_text":
        model = model or "gemini-1.5-flash"
        llm_caller = GeminiTextOutputCaller(api_key, model)
    else:
        raise ValueError(f"Unknown caller type: {caller_type}")

    input_text = example[input_field]

    if response_template is not None:
        # Structured output
        result = llm_caller.invoke(
            input_string=input_text,
            system_prompt=system_prompt,
            response_template=response_template,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        # Text output
        result = llm_caller.invoke(
            input_string=input_text,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if result:
        # Add token counts
        token_counts = llm_caller.token_counter(result)

        example.update(
            {
                "llm_response": result.get("model_dump") or result.get("text"),
                "llm_metadata": _serialize_metadata(result["metadata"]),
                "input_tokens": token_counts["input_tokens"],
                "output_tokens": token_counts["output_tokens"],
            }
        )
    else:
        example.update(
            {
                "llm_response": None,
                "llm_metadata": None,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        )

    return example


@backoff.on_exception(backoff.expo, Exception, max_tries=3, base=2, max_value=60)
def _process_single_example_dspy(args: tuple) -> Dict[str, Any]:
    """
    Process a single example with DSPy module. This function is pickle-able for multiprocessing.
    """
    (example, input_field) = args

    global _GLOBAL_CONFIG
    dspy_module = _GLOBAL_CONFIG["dspy_module"]

    input_text = example[input_field]

    try:
        # Call the dspy module's predict method
        result = dspy_module.predict(transcript=input_text)

        # Get the dictionary representation and usage metadata
        response_dict = result.toDict()
        usage_metadata = result.get_lm_usage()

        example.update(
            {"dspy_response": response_dict, "dspy_metadata": usage_metadata}
        )
    except Exception as e:
        example.update({"dspy_response": None, "dspy_metadata": {"error": str(e)}})

    return example


class ParallelProcessor:
    """Processes datasets with parallel LLM calls and rate limiting."""

    def __init__(
        self, llm_caller: Optional[LLMCallerBase] = None, max_workers: int = 4
    ):
        """
        Initialize the parallel processor.

        Args:
            llm_caller: LLM caller instance (for traditional processing)
            max_workers: Maximum number of parallel workers
        """
        self.llm_caller = llm_caller
        self.max_workers = max_workers

        if llm_caller is not None:
            self.llm_caller_config = self._get_caller_config(llm_caller)
        else:
            self.llm_caller_config = None

    def _extract_signature_info_from_module(self, dspy_module) -> Dict[str, Any]:
        """Extract signature information and demos from a DSPy module for multiprocessing."""
        # Get the signature and demos from the module's first predictor
        predictor = dspy_module.predictors()[0]
        signature = predictor.signature
        demos = predictor.demos

        # Get the docstring
        docstring = signature.__doc__ or ""

        # Extract field information
        fields = {}
        for name, field in signature.model_fields.items():
            # Convert annotation to string representation for serialization
            annotation = field.annotation
            if isinstance(annotation, str):
                annotation_str = "str"
            elif isinstance(annotation, bool):
                annotation_str = "bool"
            elif isinstance(annotation, int):
                annotation_str = "int"
            elif isinstance(annotation, float):
                annotation_str = "float"
            else:
                annotation_str = str(annotation)

            field_info = {
                "desc": field.json_schema_extra.get("desc", ""),
                "field_type": field.json_schema_extra.get(
                    "__dspy_field_type", "output"
                ),
                "annotation_str": annotation_str,
                "prefix": field.json_schema_extra.get("prefix", ""),
                "required": field.is_required(),
            }
            fields[name] = field_info

        return {"docstring": docstring, "fields": fields, "demos": demos}

    def _get_caller_config(self, llm_caller: LLMCallerBase) -> Dict[str, Any]:
        """Get configuration to recreate the LLM caller in subprocess."""
        config = {"api_key": llm_caller.api_key}

        if isinstance(llm_caller, OpenAIStructuredOutputCaller):
            config["type"] = "openai_structured"
        elif isinstance(llm_caller, OpenAITextOutputCaller):
            config["type"] = "openai_text"
        elif isinstance(llm_caller, AnthropicStructuredOutputCaller):
            config["type"] = "anthropic_structured"
        elif isinstance(llm_caller, AnthropicTextOutputCaller):
            config["type"] = "anthropic_text"
        elif isinstance(llm_caller, GeminiStructuredOutputCaller):
            config["type"] = "gemini_structured"
            config["model"] = getattr(llm_caller, "model_name", "gemini-1.5-flash")
        elif isinstance(llm_caller, GeminiTextOutputCaller):
            config["type"] = "gemini_text"
            config["model"] = getattr(llm_caller, "model_name", "gemini-1.5-flash")
        else:
            raise ValueError(f"Unknown LLM caller type: {type(llm_caller)}")

        return config

    def _single_llm_call(
        self,
        example: Dict[str, Any],
        system_prompt: str,
        model_name: str,
        input_field: str = "conversation",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_template: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Make a single LLM call - for single-threaded processing.

        Args:
            example: Single dataset example
            system_prompt: System prompt for the LLM
            model_name: Name of the model to use
            input_field: Field name in the example to use as input
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_template: Pydantic model for structured output (optional)

        Returns:
            Updated example with LLM response
        """
        # For single calls, use the instance's LLM caller directly
        input_text = example[input_field]

        if response_template is not None:
            # Structured output
            result = self.llm_caller.invoke(
                input_string=input_text,
                system_prompt=system_prompt,
                response_template=response_template,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            # Text output
            result = self.llm_caller.invoke(
                input_string=input_text,
                system_prompt=system_prompt,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        if result:
            # Add token counts
            token_counts = self.llm_caller.token_counter(result)

            example.update(
                {
                    "llm_response": result.get("model_dump") or result.get("text"),
                    "llm_metadata": _serialize_metadata(result["metadata"]),
                    "input_tokens": token_counts["input_tokens"],
                    "output_tokens": token_counts["output_tokens"],
                }
            )
        else:
            example.update(
                {
                    "llm_response": None,
                    "llm_metadata": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            )

        return example

    def process_dataset(
        self,
        dataset: Dataset,
        system_prompt: str,
        model_name: str,
        input_field: str = "conversation",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_template: Optional[Any] = None,
        num_proc: Optional[int] = None,
    ) -> Dataset:
        """
        Process entire dataset with parallel LLM calls.

        Args:
            dataset: Dataset to process
            system_prompt: System prompt for the LLM
            model_name: Name of the model to use
            input_field: Field name in the dataset to use as input
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_template: Pydantic model for structured output (optional)
            num_proc: Number of processes to use (defaults to max_workers)

        Returns:
            Dataset with LLM responses added
        """
        from multiprocessing import Pool
        import functools

        num_proc = num_proc or self.max_workers

        # Get caller configuration
        config = self.llm_caller_config

        # Create initialization function for workers
        init_func = functools.partial(
            _initialize_worker,
            caller_type=config["type"],
            api_key=config["api_key"],
            model=config.get("model"),
        )

        print(f"Processing {len(dataset)} examples with {num_proc} workers...")

        # Prepare arguments for each example
        args_list = [
            (
                example,
                system_prompt,
                model_name,
                input_field,
                temperature,
                max_tokens,
                response_template,
            )
            for example in dataset
        ]

        # Process with multiprocessing
        with Pool(num_proc, initializer=init_func) as pool:
            processed_examples = list(
                tqdm(
                    pool.imap(_process_single_example, args_list),
                    total=len(args_list),
                    desc="Processing with LLM",
                )
            )

        # Create new dataset from processed examples
        processed_dataset = Dataset.from_list(processed_examples)

        return processed_dataset

    def process_dataset_with_dspy(
        self,
        dataset: Dataset,
        dspy_module: Any,
        dspy_config: Dict[str, Any],
        input_field: str = "transcript",
        num_proc: Optional[int] = None,
    ) -> Dataset:
        """
        Process entire dataset with parallel DSPy module calls.

        Args:
            dataset: Dataset to process
            dspy_module: Pre-initialized DSPy module (used only for signature extraction)
            dspy_config: Configuration dict with model_name, api_key, and temperature
            input_field: Field name in the dataset to use as input
            num_proc: Number of processes to use (defaults to max_workers)

        Returns:
            Dataset with DSPy responses added
        """
        from multiprocessing import Pool
        import functools

        num_proc = num_proc or self.max_workers

        # Extract signature information and demos from the provided module
        signature_info = self._extract_signature_info_from_module(dspy_module)

        # Create initialization function for workers
        init_func = functools.partial(
            _initialize_worker_dspy_from_signature,
            model_name=dspy_config["model_name"],
            api_key=dspy_config["api_key"],
            signature_docstring=signature_info["docstring"],
            signature_fields=signature_info["fields"],
            demos=signature_info["demos"],
            temperature=dspy_config.get("temperature", 0),
        )

        print(
            f"Processing {len(dataset)} examples with {num_proc} workers using DSPy..."
        )

        # Prepare arguments for each example
        args_list = [(example, input_field) for example in dataset]

        # Process with multiprocessing
        with Pool(num_proc, initializer=init_func) as pool:
            processed_examples = list(
                tqdm(
                    pool.imap(_process_single_example_dspy, args_list),
                    total=len(args_list),
                    desc="Processing with DSPy",
                )
            )

        # Create new dataset from processed examples
        processed_dataset = Dataset.from_list(processed_examples)

        return processed_dataset

    def get_token_statistics(self, processed_dataset: Dataset) -> Dict[str, Any]:
        """
        Calculate token usage statistics from processed dataset.

        Args:
            processed_dataset: Dataset that has been processed with LLM calls

        Returns:
            Dictionary with token usage statistics
        """
        input_tokens = [
            ex["input_tokens"] for ex in processed_dataset if ex["input_tokens"] > 0
        ]
        output_tokens = [
            ex["output_tokens"] for ex in processed_dataset if ex["output_tokens"] > 0
        ]

        stats = {
            "total_examples": len(processed_dataset),
            "successful_calls": len(input_tokens),
            "failed_calls": len(processed_dataset) - len(input_tokens),
            "total_input_tokens": sum(input_tokens),
            "total_output_tokens": sum(output_tokens),
            "avg_input_tokens": sum(input_tokens) / len(input_tokens)
            if input_tokens
            else 0,
            "avg_output_tokens": sum(output_tokens) / len(output_tokens)
            if output_tokens
            else 0,
        }

        return stats
