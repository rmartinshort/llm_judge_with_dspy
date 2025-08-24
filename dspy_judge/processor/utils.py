from langdetect import detect, LangDetectException
import dspy
from dspy_judge.logging_config import get_logger

logger = get_logger(__name__)


def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


# functions to use with datasets.map()
def extract_llm_response_fields(example):
    # Assumes llm_response is already a dict, not a stringified JSON
    resp = example["llm_response"]
    return {
        "explanation": resp.get("explanation", None),
        "satisfied": resp.get("satisfied", None),
    }


def extract_llm_response_fields_dspy(example):
    # Assumes llm_response is already a dict, not a stringified JSON
    resp = example["dspy_response"]
    return {
        "explanation": resp.get("reasoning", None),
        "satisfied": resp.get("satisfied", None).lower(),
    }


def concat_company_and_conversation(example):
    return {
        "company_and_transcript": f"Company: {example['company']}\nTranscript so far: {example['truncated_conversation']}"
    }


def concat_latest_response(example):
    return {
        "output_transcript": f"{example['company_and_transcript']}\nSupport: {example['llm_response']}"
    }


def concat_latest_response_dspy(example):
    return {
        "output_transcript": f"{example['company_and_transcript']}\nSupport: {example['dspy_response']['llm_response']}"
    }


def build_company_and_conversation_cols(example):
    company_name = example["llm_response"].split("\n\n")[0].split(":")[1].strip()
    conversation_string = ":".join(
        "\n".join(example["llm_response"].split("\n\n")[1:]).split(":")[1:]
    ).strip()
    return {"company": company_name, "conversation": conversation_string}


def convert_dataset_to_dspy_examples(dataset, field_mapping, input_field):
    """
    Convert dataset to a list of dspy.Example objects.

    Parameters:
        dataset: The dataset object (e.g., Hugging Face Dataset)
        field_mapping: dict mapping from dspy attribute name to dataset column name
            e.g., {'transcript': 'transcript', 'satisfied': 'satisfied'}
        input_field: str
    Returns:
        List of dspy.Example objects
    """

    dataset_pd = dataset.to_pandas()
    examples = []

    for idx, row in dataset_pd.iterrows():
        example_fields = {
            key: str(row[value]).strip() if isinstance(row[value], str) else row[value]
            for key, value in field_mapping.items()
        }

        # Add an ID to help with hashing or caching
        example_fields["_id"] = f"example_{idx}"

        # Create the example with dynamic arguments
        example = dspy.Example(**example_fields).with_inputs(input_field)
        examples.append(example)

    logger.info(f"Processed {len(examples)} training examples")
    return examples
