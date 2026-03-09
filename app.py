import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from query_graph import Neo4jConnection, get_cypher_query, get_live_schema, synthesize_answer

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
    .cypher-box {
        background: #1e1e2e;
        color: #89b4fa;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-family: 'Fira Code', monospace;
        font-size: 0.8rem;
        margin: 0.5rem 0;
        overflow-x: auto;
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
    return Neo4jConnection(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USERNAME", "neo4j"),
        pwd=os.getenv("NEO4J_PASSWORD", "password"),
    )

@st.cache_resource
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("🔑 `OPENAI_API_KEY` not found in `.env`. Please add it and restart.")
        st.stop()
    return OpenAI(api_key=api_key)

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
        st.markdown(msg["content"], unsafe_allow_html=True)

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
            st.markdown(f'<div class="cypher-box">{cypher_query}</div>', unsafe_allow_html=True)

            try:
                raw_results = conn.execute_query(cypher_query)
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
            full_reply = f'<div class="cypher-box">{cypher_query}</div>\n\n{answer}'
            st.session_state.messages.append({"role": "assistant", "content": full_reply})
