import os
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types
from neo4j import GraphDatabase

# Using the same Neo4jConnection we built for graph_db.py
class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__password = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__password))
            self.__driver.verify_connectivity()
        except Exception as e:
            print(f"Error connecting to Neo4j: {e}")
            
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
            
    def execute_query(self, query, parameters=None):
        if self.__driver is None:
            raise Exception("Driver not initialized.")
        with self.__driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

def get_cypher_query(question: str, client: genai.Client) -> str:
    """
    Translates a natural language question into a Cypher query using Gemini Pro.
    """
    prompt = f"""
    You are an expert Neo4j Cypher query writer.
    Translate the user's natural language question into a valid Cypher query.
    
    Database Schema:
    - Nodes: Circular (properties: id), Clause (properties: text, page_number), Entity (properties: name, type)
    - Relationships: (Circular)-[:CONTAINS]->(Clause), (Clause)-[:REFERENCES]->(Entity), (Clause)-[:MODIFIES]->(Entity)
    
    User Question: {question}
    
    Instructions:
    - Only return the raw Cypher query string.
    - Do not include markdown formatting like ```cypher.
    - Use case-insensitive matching where appropriate (e.g. toLower(e.name) CONTAINS toLower('...')).
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
        )
        
        # Strip potential markdown formatting if the model disobeys instructions
        query = response.text.strip()
        query = re.sub(r"^```(cypher)?\n", "", query, flags=re.IGNORECASE)
        query = re.sub(r"\n```$", "", query)
        
        return query.strip()
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print("  [Agent] Rate limit hit on Gemini Pro. Falling back to Gemini Flash for Cypher generation...")
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                )
                query = response.text.strip()
                query = re.sub(r"^```(cypher)?\n", "", query, flags=re.IGNORECASE)
                query = re.sub(r"\n```$", "", query)
                
                return query.strip()
            except Exception as flash_e:
                print(f"Error generating Cypher with fallback: {flash_e}")
                return ""
        else:
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
        response = client.models.generate_content(
            model='gemini-2.5-flash',
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
                
            print("\n  [Agent] Generating Cypher query (Gemini Pro)...")
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
