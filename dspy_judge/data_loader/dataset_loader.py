"""
Dataset loader for downloading and managing the customer support dataset.
"""
import os
from pathlib import Path
from typing import Optional
from datasets import load_dataset, Dataset
from dspy_judge.processor.utils import detect_language


class CustomerSupportDatasetLoader:
    """Loads and manages the customer support on Twitter conversation dataset."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the dataset loader.
        
        Args:
            cache_dir: Directory to cache the dataset. If None, uses default cache.
        """
        self.dataset_name = "TNE-AI/customer-support-on-twitter-conversation"
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "data", "cache")
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, split: str = "train") -> Dataset:
        """
        Load the dataset from HuggingFace.
        
        Args:
            split: Dataset split to load (e.g., 'train', 'test')
            
        Returns:
            Dataset: The loaded dataset
        """
        print(f"Loading dataset {self.dataset_name} (split: {split})...")
        dataset = load_dataset(
            self.dataset_name,
            split=split,
            cache_dir=self.cache_dir
        )
        print(f"Dataset loaded successfully. Size: {len(dataset)}")
        return dataset

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:

        # remove all blank companies
        ds_filtered = dataset.filter(lambda example: example['company'] != "")
        # remove summary column, which is blank
        ds_cleaned = ds_filtered.remove_columns("summary")
        # get conversation word count
        ds_word_count = ds_cleaned.map(
            lambda example: {"conversation_word_count": len(example["conversation"].split())}
        )
        # get conversation langauge
        ds_with_lang = ds_word_count.map(
            lambda batch: {"conversation_language": [detect_language(text) for text in batch["conversation"]]},
            batched=True
        )
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
        print(f"Dataset saved to {save_path}")
    
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
        print(f"Local dataset loaded from {load_path}. Size: {len(dataset)}")
        return dataset
    
    def get_sample(self, dataset: Dataset, n_samples: int = 5, seed: int = 42) -> Dataset:
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