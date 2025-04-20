from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Any, Optional

class PromptMatch(BaseModel):
    """Result of matching a user query to a prompt template."""
    prompt_name: str = Field(description="Name of the matched prompt template (e.g., 'essay_prompt')")
    confidence: float = Field(description="Confidence score between 0-100 for this match")
    reasoning: str = Field(description="Brief explanation of why this template is appropriate")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Values extracted for template variables")
    
    @field_validator('confidence')
    @classmethod
    def normalize_confidence(cls, v):
        """Normalize confidence to 0-100 scale."""
        # If value is a decimal between 0-1 (exclusive), convert to percentage
        if 0 < v < 1:
            return v * 100
        # If value is already in percentage range, keep it
        elif 1 <= v <= 100:
            return v
        # Otherwise, it's invalid
        else:
            raise ValueError('Confidence must be between 0 and 100 (or 0 and 1 as decimal)')

class ValidationResult(BaseModel):
    """Result of validating an enhanced prompt."""
    valid: bool = Field(description="Whether the enhanced prompt is valid and maintains the original intent")
    issues: List[str] = Field(default_factory=list, description="List of identified issues or problems")

class AdjustedPrompt(BaseModel):
    """Result of adjusting a prompt to fix issues."""
    adjusted_prompt: str = Field(description="The improved, adjusted prompt that fixes previous issues")
    explanation: str = Field(description="Brief explanation of what was adjusted and why")