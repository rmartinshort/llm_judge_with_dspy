# 🤖 DSPy Judge: LLM-Powered Customer Service Evaluation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced system for evaluating customer service interactions using Large Language Models (LLMs) with automatic prompt optimization through DSPy. This project enables you to generate realistic customer service responses, train AI judges to evaluate them, and optimize both components for maximum performance.

## ✨ Key Features

- 🎯 **Automated Response Generation**: Generate realistic customer service agent responses
- ⚖️ **Intelligent Judging System**: Evaluate response quality with AI-powered judges
- 🚀 **DSPy Integration**: Automatic prompt optimization for improved performance
- 🔄 **Multi-Provider Support**: OpenAI, Anthropic, and Google Gemini APIs
- ⚡ **Parallel Processing**: Efficient batch processing with rate limiting and backoff
- 📊 **Comprehensive Metrics**: Performance tracking with Cohen's Kappa and accuracy scores
- 🔧 **Data Augmentation**: Conversation truncation for dataset expansion
- 📝 **Professional Logging**: Configurable logging with performance monitoring
- 🎛️ **Flexible Configuration**: Environment-based settings and customizable parameters

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dspy_judge
cd dspy_judge

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file with your API keys:

```bash
# Required API keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here

# Optional logging configuration
DSPY_JUDGE_LOG_LEVEL=INFO
DSPY_JUDGE_LOG_TO_FILE=true
```

### Basic Usage

```python
from dspy_judge import CustomerSupportDatasetLoader, ParallelProcessor
from dspy_judge.llm_caller import OpenAIStructuredOutputCaller
from dspy_judge.prompts.pydantic_models import JudgeResponse

# Load dataset
loader = CustomerSupportDatasetLoader()
dataset = loader.load_dataset(split="train")
sample = loader.get_sample(dataset, n_samples=100)

# Set up LLM caller
llm_caller = OpenAIStructuredOutputCaller(api_key="your_key")
processor = ParallelProcessor(llm_caller, max_workers=4)

# Judge customer service interactions
judge_prompt = """Analyze this customer service interaction and determine 
if the agent provided a satisfactory response."""

results = processor.process_dataset(
    sample,
    system_prompt=judge_prompt,
    model_name="gpt-4",
    response_template=JudgeResponse,
    temperature=0.0
)

# Get performance statistics
stats = processor.get_token_statistics(results)
print(f"Processed {stats['successful_calls']} interactions")
print(f"Used {stats['total_input_tokens']} input tokens")
```

## 📋 Main Workflows

### 1. Dataset Preparation

Transform raw customer support data into a structured format:

```python
from dspy_judge.processor.conversation_truncator import ConversationTruncator

# Load and preprocess dataset
loader = CustomerSupportDatasetLoader()
dataset = loader.load_dataset()
processed_dataset = loader.preprocess_dataset(dataset)

# Apply conversation truncation for data augmentation
truncator = ConversationTruncator(seed=42)
truncated_dataset = truncator.process_dataset(
    processed_dataset,
    min_turns=2,
    max_turns=6,
    ensure_customer_last=True
)

# Save processed dataset
loader.save_dataset_locally(truncated_dataset, "datasets/processed")
```

### 2. Response Generation

Generate customer service responses using either traditional approaches or DSPy optimization:

#### Traditional Approach
```python
from dspy_judge.llm_caller import OpenAITextOutputCaller
from dspy_judge.prompts.base_prompts import baseline_customer_response_support_system_prompt

# Set up model and processor
model = OpenAITextOutputCaller(api_key="your_key")
processor = ParallelProcessor(model, max_workers=4)

# Generate responses
results = processor.process_dataset(
    dataset,
    system_prompt=baseline_customer_response_support_system_prompt,
    model_name="gpt-3.5-turbo",
    input_field="company_and_transcript",
    temperature=1.0
)
```

#### DSPy Optimization Approach
```python
import dspy
from dspy_judge.prompts.dspy_signatures import SupportTranscriptNextResponse

# Configure DSPy
dspy.configure(
    lm=dspy.LM("openai/gpt-3.5-turbo", api_key="your_key"),
    adapter=dspy.JSONAdapter()
)

# Create and optimize module
generator = dspy.ChainOfThought(SupportTranscriptNextResponse)

# Process with DSPy
dspy_config = {
    "model_name": "openai/gpt-3.5-turbo",
    "api_key": "your_key",
    "temperature": 1.0
}

processor = ParallelProcessor()
results = processor.process_dataset_with_dspy(
    dataset,
    input_field="company_and_transcript",
    dspy_module=generator,
    dspy_config=dspy_config
)
```

### 3. Judge Training and Optimization

Train AI judges to evaluate response quality:

```python
from dspy_judge.prompts.dspy_signatures import SupportTranscriptJudge
from dspy_judge.metrics import match_judge_metric
from dspy_judge.processor.utils import convert_dataset_to_dspy_examples

# Load gold standard labels
gold_dataset = loader.load_local_dataset("datasets/gold_standard_judge_result")

# Convert to DSPy examples
examples = convert_dataset_to_dspy_examples(
    gold_dataset,
    field_mapping={"transcript": "output_transcript", "satisfied": "satisfied"},
    input_field="transcript"
)

# Set up judge model
judge_model = dspy.LM("gemini/gemini-1.5-flash", api_key="your_key")
dspy.configure(lm=judge_model, adapter=dspy.JSONAdapter())
judge = dspy.ChainOfThought(SupportTranscriptJudge)

# Optimize with DSPy
optimizer = dspy.MIPROv2(
    metric=match_judge_metric,
    auto="medium",
    init_temperature=1.0
)

# Split dataset
training_set = examples[:110]
validation_set = examples[110:]

# Compile optimized judge
optimized_judge = optimizer.compile(
    judge,
    trainset=training_set,
    valset=validation_set,
    requires_permission_to_run=False
)

# Evaluate performance
evaluator = dspy.Evaluate(metric=match_judge_metric, devset=examples)
score = evaluator(optimized_judge)
print(f"Judge accuracy: {score}%")
```

### 4. Multi-Provider Comparison

Compare performance across different LLM providers:

```python
import os
from dspy_judge.llm_caller import (
    OpenAIStructuredOutputCaller,
    AnthropicStructuredOutputCaller, 
    GeminiStructuredOutputCaller
)

providers = [
    {
        "name": "OpenAI",
        "caller": OpenAIStructuredOutputCaller(os.getenv("OPENAI_API_KEY")),
        "model": "gpt-4o-mini"
    },
    {
        "name": "Anthropic", 
        "caller": AnthropicStructuredOutputCaller(os.getenv("ANTHROPIC_API_KEY")),
        "model": "claude-3-haiku-20240307"
    },
    {
        "name": "Gemini",
        "caller": GeminiStructuredOutputCaller(os.getenv("GEMINI_API_KEY")),
        "model": "gemini-1.5-flash"
    }
]

# Test each provider
for provider in providers:
    processor = ParallelProcessor(provider["caller"], max_workers=2)
    results = processor.process_dataset(
        sample,
        system_prompt=judge_prompt,
        model_name=provider["model"],
        response_template=JudgeResponse,
        temperature=0.0
    )
    
    stats = processor.get_token_statistics(results)
    print(f"{provider['name']}: {stats['successful_calls']} successful calls")
    print(f"Tokens: {stats['total_input_tokens']} in, {stats['total_output_tokens']} out")
```

## 🎯 Performance Metrics

The system tracks various performance metrics:

- **Accuracy**: Percentage of correct judgments
- **Cohen's Kappa**: Agreement between AI judge and gold standard
- **Token Usage**: Input/output token consumption
- **Processing Speed**: Examples processed per second
- **Success Rate**: Percentage of successful API calls

Example optimization results:
- **Baseline Judge**: 66.2% accuracy
- **Optimized Judge**: 72.5% accuracy  
- **Response Generator**: 91.2% judge satisfaction score

## 📁 Project Structure

```
dspy_judge/
├── data_loader/          # Dataset loading and preprocessing
├── llm_caller/          # Multi-provider LLM interfaces
├── processor/           # Parallel processing and utilities
├── prompts/            # DSPy signatures and prompt templates
├── examples/           # Example scripts and demonstrations
├── metrics.py          # Evaluation metrics
├── plotting.py         # Visualization utilities
└── logging_config.py   # Centralized logging configuration
```

## 🔧 Configuration

### Logging Configuration

Configure logging via environment variables:

```bash
DSPY_JUDGE_LOG_LEVEL=DEBUG        # DEBUG, INFO, WARNING, ERROR
DSPY_JUDGE_LOG_TO_FILE=true       # Enable file logging
DSPY_JUDGE_LOG_FILE=logs/app.log  # Custom log file path
```

Or programmatically:

```python
from dspy_judge.logging_config import setup_logging

setup_logging(
    log_level="INFO",
    log_to_file=True,
    log_file_path="custom_logs/dspy_judge.log"
)
```

### Model Configuration

Each LLM caller supports customizable parameters:

```python
# OpenAI configuration
openai_caller = OpenAIStructuredOutputCaller(
    api_key="your_key",
    # Additional OpenAI-specific parameters
)

# Processing configuration
processor = ParallelProcessor(
    llm_caller=openai_caller,
    max_workers=8  # Adjust based on rate limits
)
```

## 🧪 Examples and Notebooks

The project includes several Jupyter notebooks demonstrating key workflows:

- `prepare_datasets.ipynb`: Dataset preprocessing and conversation truncation
- `generate_gold_standard_labels.ipynb`: Creating high-quality training labels
- `judge_optimization.ipynb`: Training and optimizing AI judges
- `optimize_main_prompt.ipynb`: End-to-end generation and judging pipeline
- `run_judge_optimization.ipynb`: Complete optimization workflow

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
