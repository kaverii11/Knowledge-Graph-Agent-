from enum import Enum
from typing import List, Optional
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

class ExtractedReference(BaseModel):
    title: str = Field(description="The title or identifier of the document/clause being referenced.")
    reference_type: ReferenceType = Field(description="The type of the referenced document.")
    page_number: int = Field(description="The page number where the reference was found.")
    context_snippet: str = Field(description="The surrounding text snippet where the reference appeared.")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence score of the extraction between 0.0 and 1.0."
    )
    extraction_method: ExtractionMethod = Field(description="The method used to extract this reference.")

class ExtractionResult(BaseModel):
    references: List[ExtractedReference] = Field(description="List of extracted legal references.", default_factory=list)
