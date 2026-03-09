import os
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__password = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__password), max_connection_lifetime=200)
            self.__driver.verify_connectivity()
        except Exception as e:
            print(f"Error connecting to Neo4j: {e}")
            
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
            
    def execute_query(self, query, parameters=None):
        if self.__driver is None:
            raise Exception("Driver not initialized.")
        try:
            with self.__driver.session() as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except Exception as e:
            self.close()
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__password), max_connection_lifetime=200)
            self.__driver.verify_connectivity()
            with self.__driver.session() as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]

def llm_call(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """Single unified LLM call using OpenAI gpt-4o-mini."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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
        rels = conn.execute_query("MATCH ()-[r]->() RETURN DISTINCT type(r) AS type")
        schema_parts.append(f"Relationship types: {[r['type'] for r in rels]}")
    except:
        schema_parts.append("Relationship types: ['REFERENCES', 'MODIFIES', 'CONTAINS']")
    try:
        circulars = conn.execute_query("MATCH (c:Circular) RETURN c.id AS id")
        schema_parts.append(f"Circular IDs: {[c['id'] for c in circulars]}")
    except:
        pass
    try:
        entities = conn.execute_query("MATCH (e:Entity) RETURN e.name AS name ORDER BY name")
        schema_parts.append(f"Entity names:\n{chr(10).join('  - ' + e['name'] for e in entities)}")
    except:
        pass
    return "\n".join(schema_parts)

def get_cypher_query(question: str, client: OpenAI, live_schema: str) -> str:
    system_prompt = """You are an expert Neo4j Cypher query writer for a SEBI compliance knowledge graph.
Database Schema:
- (c:Circular {id})
- (cl:Clause {text, page_number})
- (e:Entity {name, type})
- Relationships: (c)-[:CONTAINS]->(cl), (cl)-[:REFERENCES]->(e), (cl)-[:MODIFIES]->(e)

Rules (NEVER BREAK THESE):
- Return ONLY the raw Cypher query string. No markdown, no explanation, no backticks.
- NEVER use exact string matching (=) for entity names. ALWAYS use: toLower(e.name) CONTAINS 'keyword'
- For multi-word searches, split into multiple AND conditions on toLower(e.name)
- ONLY use relationship types listed in the LIVE INVENTORY below.
- Pick the closest matching entity from the inventory; if nothing matches, search cl.text instead.
- Always RETURN useful fields: c.id, cl.text, cl.page_number, e.name as appropriate."""

    user_prompt = f"""LIVE DATABASE INVENTORY (use these EXACT names as search keywords):
{live_schema}

User Question: {question}"""

    try:
        result = llm_call(client, system_prompt, user_prompt)
        # Strip accidental markdown fences
        result = re.sub(r"^```(cypher)?\n", "", result, flags=re.IGNORECASE)
        result = re.sub(r"\n```$", "", result)
        return result.strip()
    except Exception as e:
        print(f"\n  [DEBUG] OpenAI Cypher generation failed: {e}")
        return ""

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

def chat_interface():
    load_dotenv()
    
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("Error: OPENAI_API_KEY not found in .env file.")
        return
        
    conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    client = OpenAI(api_key=openai_key)
    
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
            
            cypher_query = get_cypher_query(question, client, live_schema)
            
            if not cypher_query:
                print("\n  Sorry, I couldn't understand that question. Try rephrasing.\n")
                continue
            
            print(f"\n  [Generated Cypher]\n  {cypher_query}\n")
            
            try:
                raw_results = conn.execute_query(cypher_query)
                final_answer = synthesize_answer(question, raw_results, client)
                print(f"\n{final_answer}\n")
            except Exception as e:
                print(f"\n  Sorry, there was a database error: {e}\n")

    except KeyboardInterrupt:
        print("\n\n[System] Shutting down gracefully. Goodbye!")
                
    finally:
        conn.close()

if __name__ == "__main__":
    chat_interface()
