"""Talks to the MCP server in-process through a real MCP client, with a fake graph connection."""
import anyio
import pytest
from mcp import Client

import mcp_server


class FakeConn:
    def __init__(self):
        self.calls = []

    def execute_query(self, query, params=None, read_only=False):
        self.calls.append((query, params, read_only))
        if "MATCH (c:Circular) OPTIONAL MATCH" in query:
            return [{"circular_id": "HO/1/I/1/2026", "subject": "Test", "date": "October 1, 2026", "documents_cited": 2}]
        return []


@pytest.fixture
def conn():
    c = FakeConn()
    mcp_server.set_connection(c)
    yield c
    mcp_server.set_connection(None)


def call(name, args=None):
    async def go():
        async with Client(mcp_server.server) as client:
            return await client.call_tool(name, args or {})
    return anyio.run(go)


def test_all_tools_are_read_only():
    async def go():
        async with Client(mcp_server.server) as client:
            return (await client.list_tools()).tools
    tools = anyio.run(go)
    assert {t.name for t in tools} == {"list_circulars", "search_documents", "circular_references",
                                       "who_cites", "run_cypher", "ask"}
    assert all(t.annotations.read_only_hint and not t.annotations.destructive_hint for t in tools)


def test_queries_run_in_read_transactions_with_parameters(conn):
    assert not call("list_circulars").is_error
    call("search_documents", {"keyword": "FPI'; DETACH DELETE n //"})
    query, params, read_only = conn.calls[-1]
    assert read_only and "FPI'; DETACH DELETE n //" == params["k"]  # passed as a parameter, never pasted into Cypher


def test_write_cypher_is_rejected_before_reaching_the_database(conn):
    r = call("run_cypher", {"query": "MATCH (n) DETACH DELETE n"})
    assert r.is_error and "only read queries" in r.content[0].text
    assert conn.calls == []


def test_input_validation(conn):
    assert call("search_documents", {"keyword": "x"}).is_error
    assert call("who_cites", {"document": "SEBI Act", "relation": "deletes"}).is_error
    assert conn.calls == []


def test_unknown_circular_gives_a_helpful_error(conn):
    r = call("circular_references", {"circular_id": "I/0000/1999"})
    assert r.is_error and "list_circulars" in r.content[0].text
