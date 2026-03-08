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

def get_cypher_query(question: str, client: genai.Client) -> str:
    """
    Translates a natural language question into a Cypher query using Gemini Flash.
    """
    prompt = f"""
    You are an expert Neo4j Cypher query writer.
    Translate the user's natural language question into a valid Cypher query.
    
    Database Schema:
    - Nodes: 
      - (c:Circular {{id}})  // Note: Circular nodes ONLY have an 'id' property. Do NOT use 'file_name' or other guessed properties.
      - (cl:Clause {{text, page_number}})
      - (e:Entity {{name, type, canonical_name}})
    - Relationships: 
      - (c)-[:CONTAINS]->(cl)
      - (cl)-[:REFERENCES]->(e)
      - (cl)-[:MODIFIES]->(e)
    
    User Question: {question}
    
    Instructions:
    - Only return the raw Cypher query string.
    - Do not include markdown formatting like ```cypher.
    - For fuzzy text/name matching, splitting into multiple CONTAINS is MUST: `toLower(e.name) CONTAINS 'word1' AND toLower(e.name) CONTAINS 'word2'`. NEVER use exact equality or a single long substring for multi-word entities.
    - Never assume the exact `id` of a Circular. Always fuzzy match it: `toLower(c.id) CONTAINS 'custodian' AND toLower(c.id) CONTAINS 'guidelines'`.
    - CRITICAL NODE SELECTION: If the user searches for generic regulations, topics, acts, or entities (e.g. "Depositories and Participants Regulations"), you MUST search the `Entity` node using `e.name`. DO NOT search `c.id` unless the user explicitly asks for a "circular named X".
    - IMPORTANT SYNTAX: If querying multiple relationships, use the `|` OR operator without repeating the colon. Use `[:REFERENCES|MODIFIES]` NOT `[:REFERENCES|:MODIFIES]`.
    - CRITICAL GRAPH PATH: When asked about what a circular "references", "modifies", or a specific named topic (like 'custodian guidelines'), you MUST query the Entity node `(e:Entity)` using `toLower(e.name) CONTAINS toLower('keyword')`. Do NOT just full-text search the `cl:Clause` text.
    """
    
    try:
        response = generate_content_with_retry(
            client=client,
            model='gemini-2.5-flash-lite',
            contents=prompt,
        )
        
        # Strip potential markdown formatting if the model disobeys instructions
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
        print("Done. Welcome to the SEBI Compliance Knowledge Graph!")
        print("Type 'exit' or 'quit' to stop.\n")
        
        while True:
            question = input("\n[Compliance Officer] Question: ")
            
            if question.lower() in ['exit', 'quit']:
                break
            if not question.strip():
                continue
                
            print("\n  [Agent] Generating Cypher query (Gemini Flash)...")
            cypher_query = get_cypher_query(question, client)
            
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
