from enum import Enum
from typing import List, Literal
from pydantic import BaseModel, Field

class ExtractionMethod(str, Enum):
    REGEX = "regex"
    FAST_LLM = "fast_llm"
    PRO_FALLBACK = "pro_fallback"

class ReferenceType(str, Enum):
    CIRCULAR = "circular"
    REGULATION = "regulation"
    ACT = "act"
    UNKNOWN = "unknown"

class LLMReference(BaseModel):
    """What the LLM is asked to fill in. Page number and extraction method are set by the pipeline, not the model."""
    title: str = Field(description="The title or identifier of the document/clause being referenced.")
    reference_type: ReferenceType = Field(description="The type of the referenced document.")
    context_snippet: str = Field(description="The surrounding text snippet where the reference appeared.")
    relation: Literal["references", "modifies"] = Field(
        default="references",
        description="'modifies' only if THIS circular itself changes, substitutes, inserts or omits text in "
        "that document; otherwise 'references'. 'As amended from time to time' is NOT a modification.",
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence score of the extraction between 0.0 and 1.0."
    )

class LLMExtractionResult(BaseModel):
    references: List[LLMReference] = Field(description="List of extracted legal references.", default_factory=list)

class ExtractedReference(LLMReference):
    page_number: int = Field(description="The page number where the reference was found.")
    extraction_method: ExtractionMethod = Field(description="The method used to extract this reference.")
