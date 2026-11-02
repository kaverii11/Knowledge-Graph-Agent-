import re
import os
import base64
from typing import TypedDict, List, Literal, Optional
from langgraph.graph import StateGraph, END
from schemas.models import ExtractedReference, LLMExtractionResult, ExtractionMethod, ReferenceType
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Below this, an extraction is sent to the stronger model for a second pass
CONFIDENCE_THRESHOLD = 0.7
FAST_MODEL = "gpt-4o-mini"
PRO_MODEL = "gpt-4o"

class GraphState(TypedDict):
    chunk_text: str
    chunk_image_bytes: Optional[bytes]
    context_text: str
    page_number: int
    references: List[ExtractedReference]
    validation_status: str
    errors: List[str]

def extract_regex(state: GraphState) -> GraphState:
    text = state["chunk_text"]
    references = state.get("references", [])
    
    if not text:
        return {"references": references}
        
    # SEBI circular IDs like SEBI/HO/MIRSD/MIRSD-PoD-1/P/CIR/2023/71 or SEBI/HO/OIAE/OIAE_IAD-1/P/CIR/2023/156.
    # Department codes mix case ("PoD") and underscores, so the prefix accepts [\w-], not just capitals.
    sebi_pattern = r"(SEBI/[\w/\-]*?(?:CIR|Master)[/\w\-]*\w)"
    matches = re.finditer(sebi_pattern, text)
    
    for match in matches:
        ref_id = match.group(1)
        # Find some surrounding context
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 50)
        snippet = text[start:end].replace("\n", " ")
        
        # Avoid duplicate regex extractions
        if not any(r.title == ref_id for r in references):
            references.append(ExtractedReference(
                title=ref_id,
                reference_type=ReferenceType.CIRCULAR,
                page_number=state["page_number"],
                context_snippet=f"...{snippet}...",
                confidence_score=1.0,  # High confidence for regex matches
                extraction_method=ExtractionMethod.REGEX
            ))
            
    return {"references": references}

def get_llm_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Only transient errors (rate limits, timeouts, 5xx) are worth retrying; bad requests or auth errors fail fast
@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)),
    reraise=True,
)
def generate_content_with_retry(client: OpenAI, model: str, messages: list, response_format):
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=0.1,
    )
    return completion.choices[0].message.parsed

def _to_references(parsed: Optional[LLMExtractionResult], page_number: int, method: ExtractionMethod) -> List[ExtractedReference]:
    if not parsed:
        return []
    return [
        ExtractedReference(**ref.model_dump(), page_number=page_number, extraction_method=method)
        for ref in parsed.references
    ]

def _build_messages(prompt: str, image_bytes: Optional[bytes]) -> list:
    """Build OpenAI message list, optionally including a base64-encoded image."""
    if image_bytes:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                ],
            }
        ]
    return [{"role": "user", "content": prompt}]

def extract_fast_llm(state: GraphState) -> GraphState:
    client = get_llm_client()
    text = state["chunk_text"]
    context = state["context_text"]
    image_bytes = state.get("chunk_image_bytes")
    references = list(state.get("references", []))
    
    prompt = f"""
    You are a legal AI assistant. Extract all mentions of other legal documents, circulars, acts, or regulations from the document.
    Do not duplicate standard SEBI circular IDs if you see them, but extract names of Acts, Regulations, or non-standard references.
    
    Context from previous sections (use for resolving pronouns like 'the aforementioned act'):
    {context}
    """
    
    if text:
        prompt += f"\n\nCurrent Text Chunk to process:\n{text}"
    else:
        prompt += "\n\nPlease extract references from the attached document image."
    
    messages = _build_messages(prompt, image_bytes)
    # Scanned pages are images, which need the vision-capable model
    model = PRO_MODEL if image_bytes else FAST_MODEL
    
    try:
        parsed_result = generate_content_with_retry(
            client=client,
            model=model,
            messages=messages,
            response_format=LLMExtractionResult,
        )
        references.extend(_to_references(parsed_result, state["page_number"], ExtractionMethod.FAST_LLM))
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Fast LLM Error: {str(e)}"]}
        
    return {"references": references}

def validate_extraction(state: GraphState) -> GraphState:
    references = state.get("references", [])
    errors = list(state.get("errors", []))
    
    low_confidence_count = 0
    for ref in references:
        if ref.confidence_score < CONFIDENCE_THRESHOLD:
            low_confidence_count += 1
            errors.append(f"Low confidence ({ref.confidence_score}) for reference: {ref.title}")
            
    if low_confidence_count > 0 or len(errors) > 0:
        return {"validation_status": "invalid", "errors": errors}
        
    return {"validation_status": "valid"}

def refine_pro(state: GraphState) -> GraphState:
    """If the fast LLM failed or had low confidence, retry with the stronger model (pro fallback)."""
    client = get_llm_client()
    text = state["chunk_text"]
    context = state["context_text"]
    image_bytes = state.get("chunk_image_bytes")
    
    prompt = f"""
    The previous extraction had validation issues or low confidence. Please carefully extract all legal references from this document.
    Ensure high accuracy.
    
    Context: {context}
    """
    
    if text:
        prompt += f"\n\nText: {text}"
    else:
        prompt += "\n\nPlease carefully extract references from the attached document image."
    
    messages = _build_messages(prompt, image_bytes)
    
    try:
        parsed_result = generate_content_with_retry(
            client=client,
            model=PRO_MODEL,
            messages=messages,
            response_format=LLMExtractionResult,
        )
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Pro Fallback Error: {str(e)}"]}

    if not parsed_result:
        # Model refused or returned nothing: keep the fast-model results rather than dropping them
        return {"errors": state.get("errors", []) + ["Pro Fallback returned no result"]}

    # Keep the regex hits, replace everything the fast model produced with the pro results
    new_refs = [r for r in state.get("references", []) if r.extraction_method == ExtractionMethod.REGEX]
    new_refs.extend(_to_references(parsed_result, state["page_number"], ExtractionMethod.PRO_FALLBACK))
    return {"references": new_refs, "validation_status": "valid", "errors": []}

def route_after_validation(state: GraphState) -> Literal["refine_pro", "__end__"]:
    if state.get("validation_status") == "invalid":
        return "refine_pro"
    return "__end__"

def build_graph():
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("extract_regex", extract_regex)
    workflow.add_node("extract_fast_llm", extract_fast_llm)
    workflow.add_node("validate_extraction", validate_extraction)
    workflow.add_node("refine_pro", refine_pro)
    
    # Build edges
    workflow.set_entry_point("extract_regex")
    workflow.add_edge("extract_regex", "extract_fast_llm")
    workflow.add_edge("extract_fast_llm", "validate_extraction")
    workflow.add_conditional_edges("validate_extraction", route_after_validation)
    workflow.add_edge("refine_pro", END)
    
    return workflow.compile()
