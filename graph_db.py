import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase, exceptions

class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__password = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__password))
            self.__driver.verify_connectivity()
            print("Successfully connected to Neo4j!")
        except exceptions.ServiceUnavailable as e:
            print(f"Failed to connect to Neo4j: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
            print("Neo4j connection closed.")
            
    def execute_query(self, query, parameters=None):
        if self.__driver is None:
            print("Driver not initialized.")
            return None
        try:
            with self.__driver.session() as session:
                result = session.run(query, parameters)
                return [record for record in result]
        except Exception as e:
            print(f"Query failed: {e}")
            return None

def ingest_data(conn: Neo4jConnection, json_file: str, circular_id: str):
    """
    Reads the extracted references and populates the Neo4j Graph.
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File {json_file} not found. Have you run the extractor yet?")
        return

    # Create the root Circular node
    circular_query = """
    MERGE (c:Circular {id: $circular_id})
    """
    conn.execute_query(circular_query, {"circular_id": circular_id})

    for item in data:
        page_number = item.get("page_number", -1)
        title = item.get("title", "Unknown")
        ref_type = item.get("reference_type", "unknown")
        context = item.get("context_snippet", "")
        
        # We model a Clause as the specific occurrence of the text on a page
        # that belongs to the Circular.
        clause_id = f"{circular_id}_page_{page_number}_{hash(context)}"
        
        # 1. Create/Merge the distinct Entity being referenced
        entity_query = """
        MERGE (e:Entity {name: $title})
        SET e.type = $ref_type
        """
        conn.execute_query(entity_query, {"title": title, "ref_type": ref_type})
        
        # 2. Create the Clause
        clause_query = """
        MERGE (cl:Clause {id: $clause_id})
        SET cl.text = $context,
            cl.page_number = $page_number
        """
        conn.execute_query(clause_query, {
            "clause_id": clause_id, 
            "context": context, 
            "page_number": page_number
        })
        
        # 3. Link Circular to Clause
        circ_clause_query = """
        MATCH (c:Circular {id: $circular_id})
        MATCH (cl:Clause {id: $clause_id})
        MERGE (c)-[:CONTAINS]->(cl)
        """
        conn.execute_query(circ_clause_query, {"circular_id": circular_id, "clause_id": clause_id})
        
        # 4. Link Clause to Entity based on modifies verb or generally references
        # Very simple heuristic: if 'amend', 'modify', 'substitute' in context snippet, it Modifies.
        modifying_verbs = ['amend', 'modify', 'substitute', 'insert', 'repeal', 'replace']
        is_modifying = any(verb in context.lower() for verb in modifying_verbs)
        
        if is_modifying:
            link_query = """
            MATCH (cl:Clause {id: $clause_id})
            MATCH (e:Entity {name: $title})
            MERGE (cl)-[:MODIFIES]->(e)
            """
        else:
            link_query = """
            MATCH (cl:Clause {id: $clause_id})
            MATCH (e:Entity {name: $title})
            MERGE (cl)-[:REFERENCES]->(e)
            """
        conn.execute_query(link_query, {"clause_id": clause_id, "title": title})

    print(f"Finished ingesting {len(data)} references into Neo4j for {circular_id}!")

if __name__ == "__main__":
    load_dotenv()
    
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # We use a hardcoded circular ID for the dummy JSON we tested earlier
    ingest_data(conn, "sebi_references.json", "sample_sebi_circular")
    
    conn.close()
