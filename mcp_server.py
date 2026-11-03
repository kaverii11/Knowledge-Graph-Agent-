"""MCP server: exposes the SEBI knowledge graph as tools any MCP client (Claude Desktop, Cursor,
an agent built on LangGraph or the OpenAI Agents SDK) can call.

Run it:
    python mcp_server.py                         # stdio, for Claude Desktop / MCP Inspector
    python mcp_server.py --http --port 8765      # Streamable HTTP

Every tool is read-only: fixed queries use parameters (no string building), and the one tool that
accepts raw Cypher rejects write clauses and runs inside a read transaction. Each call is traced in
MLflow (span type TOOL) with its inputs, outputs and latency.
"""
import argparse
import os
from typing import Annotated

from dotenv import load_dotenv
from pydantic import Field

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from query_graph import answer_question, get_live_schema, is_read_only
from utils.neo4j_client import Neo4jConnection
from utils.tracing import setup_tracing, trace

load_dotenv()

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True)
MAX_ROWS = 50

server = MCPServer(
    name="sebi-knowledge-graph",
    instructions=(
        "Knowledge graph of SEBI (India's securities regulator) circulars and the acts, regulations and "
        "other circulars they reference or modify. Start with list_circulars or search_documents; use "
        "who_cites to find every circular that references or modifies a document; use ask for questions "
        "that need a custom query."
    ),
)

_conn = None


def get_conn():
    global _conn
    if _conn is None:
        _conn = Neo4jConnection(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            os.getenv("NEO4J_USERNAME", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "password"),
        )
    return _conn


def set_connection(conn):
    """Lets tests swap in a fake connection."""
    global _conn
    _conn = conn


def _read(query: str, params: dict | None = None) -> list[dict]:
    try:
        return get_conn().execute_query(query, params or {}, read_only=True)
    except Exception as e:
        raise ToolError(f"Graph database error: {e}") from e


Keyword = Annotated[str, Field(min_length=2, max_length=200, description="Words to search for, case-insensitive")]


@server.tool(annotations=READ_ONLY)
@trace(span_type="TOOL")
def list_circulars() -> list[dict]:
    """List every SEBI circular in the graph: circular number, subject, date and how many documents it cites."""
    return _read(
        "MATCH (c:Circular) OPTIONAL MATCH (c)-[:CONTAINS]->(:Clause)-[]->(e:Entity) "
        "RETURN c.id AS circular_id, c.subject AS subject, c.date AS date, count(DISTINCT e) AS documents_cited "
        "ORDER BY c.date DESC"
    )


@server.tool(annotations=READ_ONLY)
@trace(span_type="TOOL")
def search_documents(keyword: Keyword) -> list[dict]:
    """Find acts, regulations and circulars whose name, short name or circular number contains the keyword
    (e.g. 'FPI', 'Debenture', 'IT Act'). Returns how many circulars cite each one."""
    return _read(
        "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($k) "
        "OR any(a IN coalesce(e.aliases, []) WHERE toLower(a) CONTAINS toLower($k)) "
        "OPTIONAL MATCH (c:Circular)-[:CONTAINS]->(:Clause)-[]->(e) "
        "RETURN e.name AS name, e.type AS type, e.number AS circular_number, e.aliases AS other_names, "
        "count(DISTINCT c) AS cited_by_circulars ORDER BY cited_by_circulars DESC LIMIT $limit",
        {"k": keyword, "limit": MAX_ROWS},
    )


@server.tool(annotations=READ_ONLY)
@trace(span_type="TOOL")
def circular_references(
    circular_id: Annotated[str, Field(min_length=4, max_length=120, description="Full circular number, or its last part, e.g. 'I/24814/2026'")],
) -> list[dict]:
    """Everything one circular references or modifies, with the page and the sentence it appears in."""
    rows = _read(
        "MATCH (c:Circular)-[:CONTAINS]->(cl:Clause)-[r]->(e:Entity) "
        "WHERE c.id = $id OR c.id ENDS WITH $id "
        "RETURN c.id AS circular_id, e.name AS document, e.type AS type, type(r) AS relation, "
        "cl.page_number AS page, cl.text AS evidence ORDER BY page LIMIT $limit",
        {"id": circular_id.strip(), "limit": MAX_ROWS},
    )
    if not rows:
        raise ToolError(f"No circular matching '{circular_id}'. Call list_circulars to see the numbers.")
    return rows


@server.tool(annotations=READ_ONLY)
@trace(span_type="TOOL")
def who_cites(
    document: Keyword,
    relation: Annotated[str, Field(pattern="^(any|references|modifies)$", description="any, references or modifies")] = "any",
) -> list[dict]:
    """Which circulars reference or modify a document (matched on its name, short name or circular number)."""
    rel = {"any": "", "references": ":REFERENCES", "modifies": ":MODIFIES"}[relation]
    return _read(
        f"MATCH (c:Circular)-[:CONTAINS]->(cl:Clause)-[r{rel}]->(e:Entity) "
        "WHERE toLower(e.name) CONTAINS toLower($k) "
        "OR any(a IN coalesce(e.aliases, []) WHERE toLower(a) CONTAINS toLower($k)) "
        "RETURN DISTINCT c.id AS circular_id, c.subject AS subject, c.date AS date, e.name AS document, "
        "type(r) AS relation, cl.page_number AS page ORDER BY date DESC LIMIT $limit",
        {"k": document, "limit": MAX_ROWS},
    )


@server.tool(annotations=READ_ONLY)
@trace(span_type="TOOL")
def run_cypher(
    query: Annotated[str, Field(min_length=6, max_length=2000, description="A read-only Cypher query")],
) -> list[dict]:
    """Run a custom read-only Cypher query. Schema: (:Circular {id, subject, date})-[:CONTAINS]->
    (:Clause {text, page_number})-[:REFERENCES|MODIFIES]->(:Entity {name, type, aliases, number}).
    Queries that create, change or delete data are rejected. Returns at most 50 rows."""
    if not is_read_only(query):
        raise ToolError("Rejected: only read queries are allowed (no CREATE, MERGE, SET, DELETE, REMOVE, DROP).")
    return _read(query)[:MAX_ROWS]


@server.tool(annotations=READ_ONLY)
@trace(span_type="TOOL")
def ask(question: Annotated[str, Field(min_length=5, max_length=500)]) -> dict:
    """Answer a plain-English question about the circulars. An LLM writes a Cypher query, it is checked
    to be read-only and run, and the answer is written only from the rows returned."""
    from utils.llm import get_client
    out = answer_question(question, get_conn(), get_client())
    if out["error"]:
        raise ToolError(f"Could not answer: {out['error']}")
    return {"answer": out["answer"], "cypher": out["cypher"], "rows": out["results"][:MAX_ROWS]}


@server.resource("graph://schema", name="schema", description="Node labels, relationships and the documents in the graph")
def schema() -> str:
    return get_live_schema(get_conn())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="Serve over Streamable HTTP instead of stdio")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    setup_tracing()
    if args.http:
        server.run(transport="streamable-http", port=args.port)
    else:
        server.run()
