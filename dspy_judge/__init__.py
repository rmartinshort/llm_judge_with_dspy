"""
dspy_judge: A package for LLM-based judging with DSPy integration.

This package provides tools for:
- Loading and processing customer support datasets
- Making parallel LLM calls with rate limiting
- Using multiple LLM providers (OpenAI, Anthropic, Google Gemini)
- Training and evaluating DSPy modules for judging tasks
- Comprehensive logging and monitoring
"""

# Import logging configuration to ensure it's set up when package is imported
from . import logging_config

# Version information
__version__ = "0.1.0"

# Main exports
from .data_loader.dataset_loader import CustomerSupportDatasetLoader
from .processor.parallel_processor import ParallelProcessor
from .processor.conversation_truncator import ConversationTruncator
from .logging_config import setup_logging, get_logger, configure_from_env

__all__ = [
    "CustomerSupportDatasetLoader",
    "ParallelProcessor",
    "ConversationTruncator",
    "setup_logging",
    "get_logger",
    "configure_from_env",
]
