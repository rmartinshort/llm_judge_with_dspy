import dspy
from dspy_judge.prompts.base_prompts import (
    baseline_customer_response_support_system_prompt,
    baseline_judge_system_prompt,
)


class SupportTranscriptJudge(dspy.Signature):
    transcript: str = dspy.InputField(desc="Input transcript to judge")
    satisfied: bool = dspy.OutputField(
        desc="Whether the agent satisfied the customer query"
    )


SupportTranscriptJudge.__doc__ = baseline_judge_system_prompt.strip()


class SupportTranscriptNextResponse(dspy.Signature):
    transcript: str = dspy.InputField(desc="Input transcript to judge")
    llm_response: str = dspy.OutputField(desc="The support agent's next utterance")


SupportTranscriptNextResponse.__doc__ = (
    baseline_customer_response_support_system_prompt.strip()
)
