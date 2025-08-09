"""
Example demonstrating conversation truncation for data augmentation.
"""

from dspy_judge.data_loader.dataset_loader import CustomerSupportDatasetLoader
from dspy_judge.processor.conversation_truncator import ConversationTruncator


def demonstrate_truncation():
    """Demonstrate conversation truncation functionality."""

    # Load dataset
    loader = CustomerSupportDatasetLoader()
    dataset = loader.load_dataset(split="train")
    sample = loader.get_sample(dataset, n_samples=3)

    print("Original conversations:")
    for i, example in enumerate(sample):
        print(f"\n--- Example {i + 1} (Original) ---")
        print(example["conversation"])

    # Create truncator
    truncator = ConversationTruncator(seed=42)

    # Truncate conversations
    truncated_dataset = truncator.process_dataset(
        sample, min_turns=2, max_turns=4, ensure_customer_last=True
    )

    print("\n" + "=" * 50)
    print("Truncated conversations (customer speaks last):")

    for i, example in enumerate(truncated_dataset):
        print(f"\n--- Example {i + 1} (Truncated) ---")
        print(example["truncated_conversation"])

    # Get statistics
    stats = truncator.get_truncation_stats(sample, truncated_dataset)
    print("\n--- Truncation Statistics ---")
    print(f"Average original turns: {stats['avg_original_turns']:.1f}")
    print(f"Average truncated turns: {stats['avg_truncated_turns']:.1f}")
    print(f"Average reduction: {stats['avg_reduction']:.1f} turns")

    # Demonstrate reproducibility
    print("\n--- Reproducibility Test ---")
    truncated_again = truncator.process_dataset(
        sample, min_turns=2, max_turns=4, ensure_customer_last=True
    )

    # Check if results are identical
    all_match = True
    for orig, new in zip(truncated_dataset, truncated_again):
        if orig["truncated_conversation"] != new["truncated_conversation"]:
            all_match = False
            break

    print(f"Truncation is reproducible: {all_match}")


if __name__ == "__main__":
    demonstrate_truncation()
