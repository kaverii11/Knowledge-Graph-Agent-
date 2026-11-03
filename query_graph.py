import os
import re
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from utils.llm import QUERY_MODEL, get_client
from utils.neo4j_client import Neo4jConnection
from utils.tracing import setup_tracing, trace

# The LLM writes the Cypher, so refuse anything that could change the graph
WRITE_CLAUSES = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL\s+apoc)\b", re.IGNORECASE)

STRINGS_AND_COMMENTS = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|//[^\n]*|/\*.*?\*/", re.S)

def is_read_only(cypher: str) -> bool:
    """Keyword check on the query with string literals and comments removed, so a search for the
    word 'set' is allowed. The query also runs in a read transaction, which Neo4j enforces."""
    return not WRITE_CLAUSES.search(STRINGS_AND_COMMENTS.sub("''", cypher))

@trace(span_type="TOOL")
def run_generated_query(conn, cypher: str) -> list:
    """Runs LLM-generated Cypher in a read transaction, after rejecting write clauses."""
    if not is_read_only(cypher):
        raise ValueError("Generated query tries to modify the graph; only read queries are allowed.")
    return conn.execute_query(cypher, read_only=True)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60), reraise=True,
       retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)))
def llm_call(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """Single LLM call on the query model."""
    response = client.chat.completions.create(
        model=QUERY_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

def get_live_schema(conn) -> str:
    schema_parts = []
    try:
        rels = conn.execute_query("MATCH ()-[r]->() RETURN DISTINCT type(r) AS type", read_only=True)
        schema_parts.append(f"Relationship types: {[r['type'] for r in rels]}")
    except Exception:
        schema_parts.append("Relationship types: ['REFERENCES', 'MODIFIES', 'CONTAINS']")
    try:
        circulars = conn.execute_query(
            "MATCH (c:Circular) RETURN c.id AS id, c.date AS date, c.subject AS subject ORDER BY id", read_only=True)
        schema_parts.append("Circulars (c.id | c.date | c.subject):\n" + "\n".join(
            f"  - {c['id']} | {c['date']} | {c['subject']}" for c in circulars))
    except Exception:
        pass
    try:
        entities = conn.execute_query(
            "MATCH (e:Entity) RETURN e.name AS name, e.type AS type, e.number AS number, e.aliases AS aliases ORDER BY name",
            read_only=True)
        lines = []
        for e in entities:
            extra = f" (number: {e['number']})" if e.get("number") else ""
            if e.get("aliases"):
                extra += f" (also called: {'; '.join(e['aliases'])})"
            lines.append(f"  - [{e['type']}] {e['name']}{extra}")
        schema_parts.append("Entities (e.name):\n" + "\n".join(lines))
    except Exception:
        pass
    return "\n".join(schema_parts)

@trace(span_type="LLM")
def get_cypher_query(question: str, client: OpenAI, live_schema: str) -> str:
    system_prompt = """You are an expert Neo4j Cypher query writer for a SEBI compliance knowledge graph.
Database Schema:
- (c:Circular {id, date, subject})          one node per circular that was ingested
- (cl:Clause {text, page_number})           the sentence where a document is mentioned
- (e:Entity {name, type, number, aliases})  a document that circulars cite: act, regulation or circular;
                                            number = its circular number, aliases = other names for it
- Relationships: (c)-[:CONTAINS]->(cl), (cl)-[:REFERENCES]->(e), (cl)-[:MODIFIES]->(e)

Rules (NEVER BREAK THESE):
- Return ONLY the raw Cypher query string. No markdown, no explanation, no backticks.
- Identify circulars from the inventory: by topic use c.subject, by date use c.date, or use the exact c.id.
- Match an entity with 1-3 distinctive keywords, never every word of its title, and check aliases too:
  (toLower(e.name) CONTAINS 'kw' OR any(a IN coalesce(e.aliases, []) WHERE toLower(a) CONTAINS 'kw'))
- "cite / refer to / mention" means any relationship: (cl)-[r]->(e). Use :MODIFIES only when asked about changes.
- For "both" / "all" questions, collect per document and filter on the count of distinct circulars.
- If the answer is a property of a node (number, date, subject), return that property.
- Always RETURN useful fields: c.id, c.subject, cl.page_number, e.name, type(r) as appropriate."""

    user_prompt = f"""LIVE DATABASE INVENTORY (use these EXACT names as search keywords):
{live_schema}

User Question: {question}"""

    try:
        result = llm_call(client, system_prompt, user_prompt)
        # Strip accidental markdown fences
        result = re.sub(r"^```(cypher)?\s*", "", result, flags=re.IGNORECASE)
        result = re.sub(r"\s*```$", "", result)
        return result.strip()
    except Exception as e:
        print(f"\n  Cypher generation failed: {e}")
        return ""

@trace(span_type="LLM")
def synthesize_answer(question: str, results: list, client: OpenAI) -> str:
    if not results:
        return "No results found for your question."
    
    system_prompt = """You are an AI assistant for a major Indian Bank's Compliance Team.
Based on the following JSON results from a SEBI Compliance Knowledge Graph, answer the user's question.
Give a professional, concise, and highly readable summary. Use bullet points for clarity where appropriate."""

    user_prompt = f"""User Question: {question}

Neo4j Results:
{results}"""

    try:
        return llm_call(client, system_prompt, user_prompt)
    except Exception as e:
        return f"Error synthesizing answer: {e}"

@trace(span_type="AGENT")
def answer_question(question: str, conn, client: OpenAI, live_schema: str | None = None) -> dict:
    """Text-to-Cypher agent: write a query, check it is read-only, run it, then answer from the rows.

    Shared by the CLI, the Streamlit app, the MCP server and the evals.
    """
    if live_schema is None:
        live_schema = get_live_schema(conn)
    cypher = get_cypher_query(question, client, live_schema)
    if not cypher:
        return {"question": question, "cypher": "", "results": [], "answer": "", "error": "no query generated"}
    try:
        results = run_generated_query(conn, cypher)
    except Exception as e:
        return {"question": question, "cypher": cypher, "results": [], "answer": "", "error": str(e)}
    return {"question": question, "cypher": cypher, "results": results,
            "answer": synthesize_answer(question, results, client), "error": None}

def chat_interface():
    load_dotenv()
    
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    try:
        client = get_client()
    except RuntimeError as e:
        print(f"Error: {e}")
        return
    setup_tracing()

    try:
        conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return
    try:
        live_schema = get_live_schema(conn)
        print("Ready! Welcome to the SEBI Compliance Knowledge Graph.")
        print("Type 'exit' or 'quit' to stop.\n")
        
        while True:
            question = input("\n[Compliance Officer] Question: ")
            
            if question.lower() in ['exit', 'quit']:
                break
            if not question.strip():
                continue
            
            out = answer_question(question, conn, client, live_schema)
            if not out["cypher"]:
                print("\n  Sorry, I couldn't understand that question. Try rephrasing.\n")
                continue
            print(f"\n  [Generated Cypher]\n  {out['cypher']}\n")
            if out["error"]:
                print(f"\n  Sorry, there was a database error: {out['error']}\n")
            else:
                print(f"\n{out['answer']}\n")

    except KeyboardInterrupt:
        print("\n\n[System] Shutting down gracefully. Goodbye!")
                
    finally:
        conn.close()

if __name__ == "__main__":
    chat_interface()
