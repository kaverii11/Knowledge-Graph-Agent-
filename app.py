import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from query_graph import get_cypher_query, get_live_schema, run_generated_query, synthesize_answer
from utils.llm import get_client
from utils.tracing import setup_tracing
from utils.neo4j_client import Neo4jConnection

load_dotenv()

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="SEBI Compliance Agent", page_icon="🏦", layout="centered")

# ── Custom Styling ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #1a73e8, #6e3ef5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #6b7280;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏦 SEBI Compliance Agent</h1>
    <p>Ask natural-language questions about SEBI circulars, acts & regulations</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Cached Resources (persist across reruns) ─────────────────────────────────
@st.cache_resource
def get_neo4j_connection():
    try:
        return Neo4jConnection(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USERNAME", "neo4j"),
            pwd=os.getenv("NEO4J_PASSWORD", "password"),
        )
    except Exception as e:
        st.error(f"Could not connect to Neo4j: {e}")
        st.stop()

@st.cache_resource
def get_openai_client():
    try:
        setup_tracing()
        return get_client()
    except RuntimeError as e:
        st.error(f"🔑 {e}")
        st.stop()

conn = get_neo4j_connection()
client = get_openai_client()

# Fetch live schema once per session
if "live_schema" not in st.session_state:
    st.session_state.live_schema = get_live_schema(conn)

# ── Chat History ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! I'm your SEBI Compliance assistant. Ask me anything about circulars, regulations, or acts in the knowledge graph. 🏛️"}
    ]

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat Input ───────────────────────────────────────────────────────────────
if question := st.chat_input("e.g. Which circulars modify the SEBI Act?"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate Cypher & query graph
    with st.chat_message("assistant"):
        with st.spinner("Generating Cypher & Querying Graph..."):
            cypher_query = get_cypher_query(question, client, st.session_state.live_schema)

        if not cypher_query:
            reply = "Sorry, I couldn't understand that question. Try rephrasing."
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            # Show generated Cypher
            st.code(cypher_query, language="cypher")

            try:
                raw_results = run_generated_query(conn, cypher_query)
            except Exception as e:
                reply = f"⚠️ Database error: `{e}`"
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.stop()

            # Synthesize answer
            with st.spinner("Synthesizing Answer..."):
                answer = synthesize_answer(question, raw_results, client)

            st.markdown(answer)

            # Store full reply (Cypher + answer) in history
            full_reply = f"```cypher\n{cypher_query}\n```\n\n{answer}"
            st.session_state.messages.append({"role": "assistant", "content": full_reply})
