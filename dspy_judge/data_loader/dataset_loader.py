"""
Dataset loader for loading and managing the local airline support conversations dataset.
"""

import os
from pathlib import Path
from typing import Optional
from datasets import load_dataset, Dataset
from dspy_judge.processor.utils import detect_language
from dspy_judge.logging_config import get_logger

logger = get_logger(__name__)


class CustomerSupportDatasetLoader:
    """Loads and manages the local airline support conversations dataset."""

    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize the dataset loader.

        Args:
            dataset_path: Path to the local dataset directory. If None, uses default path.
        """
        self.dataset_path = dataset_path or os.path.join(
            os.getcwd(), "datasets", "airline_support_conversations"
        )
        if not Path(self.dataset_path).exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

    def load_dataset(self, split: str = None) -> Dataset:
        """
        Load the local airline support conversations dataset.

        Args:
            split: Dataset split to load (ignored for local dataset, kept for compatibility)

        Returns:
            Dataset: The loaded dataset
        """
        logger.info(f"Loading local dataset from {self.dataset_path}...")
        dataset = load_dataset(
            "arrow", data_files=os.path.join(self.dataset_path, "*.arrow")
        )["train"]
        logger.info(f"Dataset loaded successfully. Size: {len(dataset)}")
        return dataset

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """
        Preprocess the airline support conversations dataset.

        Args:
            dataset: Raw dataset to preprocess

        Returns:
            Dataset: Preprocessed dataset with conversation_word_count and language detection
        """
        # Filter out entries with blank companies (if company field exists)
        if "company" in dataset.column_names:
            ds_filtered = dataset.filter(lambda example: example["company"] != "")
        else:
            ds_filtered = dataset
            logger.info("No 'company' field found, skipping company filtering")

        # Remove unnecessary columns if they exist
        columns_to_remove = []
        if "llm_metadata" in ds_filtered.column_names:
            columns_to_remove.append("llm_metadata")
        if "llm_response" in ds_filtered.column_names and all(
            x is None or x == "" for x in ds_filtered["llm_response"][:10]
        ):
            columns_to_remove.append("llm_response")

        if columns_to_remove:
            ds_cleaned = ds_filtered.remove_columns(columns_to_remove)
            logger.info(f"Removed columns: {columns_to_remove}")
        else:
            ds_cleaned = ds_filtered

        # Add conversation word count if conversation field exists
        if "conversation" in ds_cleaned.column_names:
            ds_word_count = ds_cleaned.map(
                lambda example: {
                    "conversation_word_count": len(example["conversation"].split())
                }
            )
        elif "Text" in ds_cleaned.column_names:
            # Use 'Text' field if 'conversation' doesn't exist
            ds_word_count = ds_cleaned.map(
                lambda example: {
                    "conversation_word_count": len(example["Text"].split())
                }
            )
        else:
            ds_word_count = ds_cleaned
            logger.warning(
                "No conversation or Text field found for word count calculation"
            )

        # Add conversation language detection
        conversation_field = (
            "conversation" if "conversation" in ds_word_count.column_names else "Text"
        )
        if conversation_field in ds_word_count.column_names:
            ds_with_lang = ds_word_count.map(
                lambda batch: {
                    "conversation_language": [
                        detect_language(text) for text in batch[conversation_field]
                    ]
                },
                batched=True,
            )
        else:
            ds_with_lang = ds_word_count
            logger.warning("No text field found for language detection")

        return ds_with_lang

    def save_dataset_locally(self, dataset: Dataset, save_path: str) -> None:
        """
        Save the dataset locally.

        Args:
            dataset: The dataset to save
            save_path: Path where to save the dataset
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(save_path)
        logger.info(f"Dataset saved to {save_path}")

    def load_local_dataset(self, load_path: str) -> Dataset:
        """
        Load a locally saved dataset.

        Args:
            load_path: Path to the saved dataset

        Returns:
            Dataset: The loaded dataset
        """
        from datasets import load_from_disk

        dataset = load_from_disk(load_path)
        logger.info(f"Local dataset loaded from {load_path}. Size: {len(dataset)}")
        return dataset

    def get_sample(
        self, dataset: Dataset, n_samples: int = 5, seed: int = 42
    ) -> Dataset:
        """
        Get a random sample of the dataset for testing purposes.

        Args:
            dataset: The dataset to sample from
            n_samples: Number of samples to return
            seed: Random seed for reproducible sampling

        Returns:
            Dataset: A randomly sampled subset of the original dataset
        """
        shuffled_dataset = dataset.shuffle(seed=seed)
        return shuffled_dataset.select(range(min(n_samples, len(dataset))))
