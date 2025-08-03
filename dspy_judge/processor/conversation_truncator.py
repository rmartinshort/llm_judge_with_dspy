"""
Conversation truncation module for data augmentation.
"""
import random
from typing import List, Dict, Any, Optional
from datasets import Dataset


class ConversationTruncator:
    """Truncates conversations for data augmentation."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the conversation truncator.
        
        Args:
            seed: Random seed for reproducible truncation
        """
        self.seed = seed
    
    def parse_conversation(self, conversation: str) -> List[Dict[str, str]]:
        """
        Parse conversation string into structured format.
        
        Args:
            conversation: Raw conversation string
            
        Returns:
            List of dictionaries with role and content
        """
        lines = conversation.strip().split('\n')
        parsed = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to identify speaker patterns
            if line.startswith('Customer:') or line.startswith('User:'):
                parsed.append({
                    'role': 'customer',
                    'content': line.split(':', 1)[1].strip()
                })
            elif line.startswith('Agent:') or line.startswith('Support:'):
                parsed.append({
                    'role': 'agent', 
                    'content': line.split(':', 1)[1].strip()
                })
            else:
                # Fallback: alternate between customer and agent
                if not parsed:
                    role = 'customer'
                else:
                    role = 'agent' if parsed[-1]['role'] == 'customer' else 'customer'
                parsed.append({
                    'role': role,
                    'content': line
                })
        
        return parsed
    
    def truncate_conversation(
        self, 
        conversation: str, 
        min_turns: int = 2,
        max_turns: int = None,
        ensure_customer_last: bool = True,
        example_idx: int = 0
    ) -> str:
        """
        Randomly truncate a conversation.
        
        Args:
            conversation: Original conversation string
            min_turns: Minimum number of turns to keep
            max_turns: Maximum number of turns to keep (None = no limit)
            ensure_customer_last: Ensure customer speaks last
            example_idx: Index of the example (for consistent seeding)
            
        Returns:
            Truncated conversation string
        """
        # Set seed based on global seed and example index for reproducibility
        random.seed(self.seed + example_idx)
        
        parsed = self.parse_conversation(conversation)

        if len(parsed) <= min_turns:
            return conversation
        
        # Determine truncation point
        max_possible = len(parsed)
        if max_turns:
            max_possible = min(max_possible, max_turns)
        
        truncate_at = random.randint(min_turns, max_possible)
        
        # Ensure customer speaks last if requested
        if ensure_customer_last:
            # Find the last customer turn within our truncation range
            last_customer_idx = -1
            for i in range(truncate_at - 1, -1, -1):
                if parsed[i]['role'] == 'customer':
                    last_customer_idx = i
                    break
            
            if last_customer_idx != -1:
                truncate_at = last_customer_idx + 1
        
        # Reconstruct conversation
        truncated = parsed[:truncate_at]
        
        # Format back to string
        formatted_lines = []
        for turn in truncated:
            speaker = "Customer" if turn['role'] == 'customer' else "Agent"
            formatted_lines.append(f"{speaker}: {turn['content']}")
        
        return '\n'.join(formatted_lines)
    
    def process_dataset(
        self,
        dataset: Dataset,
        conversation_field: str = "conversation",
        min_turns: int = 2,
        max_turns: int = None,
        ensure_customer_last: bool = True,
        new_field_name: str = "truncated_conversation"
    ) -> Dataset:
        """
        Process entire dataset to create truncated conversations.
        
        Args:
            dataset: Input dataset
            conversation_field: Field containing conversation text
            min_turns: Minimum number of turns to keep
            max_turns: Maximum number of turns to keep
            ensure_customer_last: Ensure customer speaks last
            new_field_name: Name for the new truncated conversation field
            
        Returns:
            Dataset with truncated conversations added
        """
        def truncate_example(example, idx):
            truncated = self.truncate_conversation(
                example[conversation_field],
                min_turns=min_turns,
                max_turns=max_turns,
                ensure_customer_last=ensure_customer_last,
                example_idx=idx
            )
            example[new_field_name] = truncated
            return example
        
        return dataset.map(truncate_example, with_indices=True, desc="Truncating conversations")
    
    def get_truncation_stats(
        self,
        original_dataset: Dataset,
        truncated_dataset: Dataset,
        original_field: str = "conversation",
        truncated_field: str = "truncated_conversation"
    ) -> Dict[str, Any]:
        """
        Get statistics about the truncation process.
        
        Args:
            original_dataset: Original dataset
            truncated_dataset: Dataset with truncated conversations
            original_field: Field name for original conversations
            truncated_field: Field name for truncated conversations
            
        Returns:
            Dictionary with truncation statistics
        """
        original_lengths = []
        truncated_lengths = []
        
        for orig, trunc in zip(original_dataset, truncated_dataset):
            orig_turns = len(self.parse_conversation(orig[original_field]))
            trunc_turns = len(self.parse_conversation(trunc[truncated_field]))
            
            original_lengths.append(orig_turns)
            truncated_lengths.append(trunc_turns)
        
        return {
            "avg_original_turns": sum(original_lengths) / len(original_lengths),
            "avg_truncated_turns": sum(truncated_lengths) / len(truncated_lengths),
            "avg_reduction": (sum(original_lengths) - sum(truncated_lengths)) / len(original_lengths),
            "total_examples": len(truncated_dataset)
        }