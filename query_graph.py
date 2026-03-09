import os
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types
from neo4j import GraphDatabase
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Using the same Neo4jConnection we built for graph_db.py
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
            # If the connection was reset by peer (e.g. timeout), try to reconnect once
            print(f"  [Neo4j Driver] Connection dropped ({e}). Attempting to reconnect...")
            self.close()
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__password), max_connection_lifetime=200)
            self.__driver.verify_connectivity()
            with self.__driver.session() as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]

@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=2, min=15, max=70),
    retry=retry_if_exception_type(Exception)
)
def generate_content_with_retry(client, model, contents):
    try:
        return client.models.generate_content(
            model=model,
            contents=contents,
        )
    except Exception as e:
        print(f"  [Gemini API] Retryable Error: {e}")
        raise

def get_live_schema(conn) -> str:
    """
    Dynamically fetches the actual schema from the live Neo4j database
    so the LLM knows exactly what nodes, properties, relationships, and
    entity names exist. This eliminates all hallucination.
    """
    schema_parts = []
    
    # Get relationship types
    try:
        rels = conn.execute_query("MATCH ()-[r]->() RETURN DISTINCT type(r) AS type")
        rel_types = [r['type'] for r in rels]
        schema_parts.append(f"Relationship types that ACTUALLY EXIST: {rel_types}")
    except:
        rel_types = ['REFERENCES']
        schema_parts.append("Relationship types: ['REFERENCES']")
    
    # Get circular IDs
    try:
        circulars = conn.execute_query("MATCH (c:Circular) RETURN c.id AS id")
        circular_ids = [c['id'] for c in circulars]
        schema_parts.append(f"Circular IDs: {circular_ids}")
    except:
        schema_parts.append("Circular IDs: unknown")
    
    # Get all entity names
    try:
        entities = conn.execute_query("MATCH (e:Entity) RETURN e.name AS name ORDER BY name")
        entity_names = [e['name'] for e in entities]
        schema_parts.append(f"Entity names in database:\n{chr(10).join('  - ' + n for n in entity_names)}")
    except:
        schema_parts.append("Entity names: unknown")
    
    return "\n".join(schema_parts)

def get_cypher_query(question: str, client: genai.Client, live_schema: str) -> str:
    """
    Translates a natural language question into a Cypher query using Gemini Flash.
    Uses a live schema snapshot so the LLM knows exactly what data exists.
    """
    prompt = f"""
    You are an expert Neo4j Cypher query writer.
    Translate the user's natural language question into a valid Cypher query.
    
    Database Schema:
    - Nodes: 
      - (c:Circular {{id}})
      - (cl:Clause {{text, page_number}})
      - (e:Entity {{name, type}})
    - Relationships: (c)-[:CONTAINS]->(cl), (cl)-[:REFERENCES]->(e)
    
    LIVE DATABASE INVENTORY (use these EXACT names when matching):
    {live_schema}
    
    User Question: {question}
    
    Instructions:
    - Only return the raw Cypher query string. No markdown formatting.
    - ONLY use relationship types from the LIVE DATABASE INVENTORY above. If MODIFIES is not listed, do NOT use it.
    - When matching entity names, use fuzzy matching with toLower() CONTAINS on individual keywords.
    - When matching circular IDs, use toLower(c.id) CONTAINS 'keyword'.
    - Pick the closest matching entity name from the inventory above. If no entity closely matches the question, search cl.text instead.
    - Always RETURN useful fields: c.id, cl.text, cl.page_number, e.name as appropriate.
    """
    
    try:
        response = generate_content_with_retry(
            client=client,
            model='gemini-2.5-flash-lite',
            contents=prompt,
        )
        
        query = response.text.strip()
        query = re.sub(r"^```(cypher)?\n", "", query, flags=re.IGNORECASE)
        query = re.sub(r"\n```$", "", query)
        
        return query.strip()
    except Exception as e:
        print(f"Error generating Cypher: {e}")
        return ""

def synthesize_answer(question: str, results: list, client: genai.Client) -> str:
    """
    Takes the raw Neo4j results and uses Gemini Flash to write a clean summary for a compliance officer.
    """
    if not results:
        return "The database query yielded no results matching your question."
        
    prompt = f"""
    You are an AI assistant for a major Indian Bank's Compliance Team.
    Based on the following JSON results from our Knowledge Graph, answer the user's question.
    Give a professional, concise, and highly readable summary.
    
    User Question: {question}
    
    Neo4j Results:
    {results}
    """
    
    try:
        response = generate_content_with_retry(
            client=client,
            model='gemini-2.5-flash-lite',
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"Error synthesizing answer: {e}"

def chat_interface():
    load_dotenv()
    
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file.")
        return
        
    print("\nConnecting to Neo4j database...")
    conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    client = genai.Client(api_key=api_key)
    
    try:
        # Fetch live schema once at startup
        print("Fetching live schema from Neo4j...")
        live_schema = get_live_schema(conn)
        print("Done. Welcome to the SEBI Compliance Knowledge Graph!")
        print("Type 'exit' or 'quit' to stop.\n")
        
        while True:
            question = input("\n[Compliance Officer] Question: ")
            
            if question.lower() in ['exit', 'quit']:
                break
            if not question.strip():
                continue
                
            print("\n  [Agent] Generating Cypher query (Gemini Flash)...")
            cypher_query = get_cypher_query(question, client, live_schema)
            
            if not cypher_query:
                print("  [Agent] Failed to generate a valid Cypher query. Try rephrasing.")
                continue
                
            print(f"  [Agent] Executing: {cypher_query}")
            
            try:
                raw_results = conn.execute_query(cypher_query)
                print(f"  [Agent] Fetched {len(raw_results)} records from Neo4j.")
                
                print("  [Agent] Synthesizing final answer (Gemini Flash)...")
                final_answer = synthesize_answer(question, raw_results, client)
                
                print("\n================ ANSWER ================\n")
                print(final_answer)
                print("\n========================================\n")
            except Exception as e:
                print(f"  [Agent] Neo4j Execution Error: {e}")
                
    finally:
        conn.close()

if __name__ == "__main__":
    chat_interface()
