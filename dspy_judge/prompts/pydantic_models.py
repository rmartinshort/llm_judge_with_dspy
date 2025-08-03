from pydantic import BaseModel, Field

class JudgeResponse(BaseModel):
    """Response schema for the judge model."""
    satisfied: bool = Field(description="Whether the agent satisfied the customer query")
    explanation: str = Field(description="Explanation for the decision")