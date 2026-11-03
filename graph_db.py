import os
import re
import sys
import json
import hashlib
import argparse
import difflib
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from utils.llm import EMBED_MODEL, get_client
from utils.neo4j_client import Neo4jConnection
from utils.entity_resolution import resolve_circulars

def is_strict_sebi_format(title: str) -> bool:
    """
    Checks if a title strictly follows SEBI circular patterns to bypass fuzzy matching.
    """
    pattern = r"^(SEBI|CIR|PR|SMDRP)\/.*[\d]{4}.*"
    return bool(re.match(pattern, title.upper())) or bool(re.fullmatch(r"HO/\S+/I/\d+/\d{4}", title))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def _normalize_title(title: str) -> str:
    t = re.sub(r"[^a-z0-9 ]", " ", title.lower())
    return " ".join(w for w in t.split() if w not in {"the", "of", "and"})

def lexical_similarity(a: str, b: str) -> float:
    """Fallback when no embedding model is available (e.g. Groq has none).

    Only merges near-identical spellings, and never merges titles with different years,
    so 'SEBI (ICDR) Regulations, 2018' and 'SEBI (LODR) Regulations, 2015' stay separate.
    """
    if set(re.findall(r"\b(?:19|20)\d{2}\b", a)) != set(re.findall(r"\b(?:19|20)\d{2}\b", b)):
        return 0.0
    return difflib.SequenceMatcher(None, _normalize_title(a), _normalize_title(b)).ratio()

def resolve_entities(extracted_references: list, client: OpenAI, threshold: float = 0.88) -> list:
    """
    Takes a list of reference dictionaries, separates strict SEBI circular IDs,
    and clusters the remaining natural language entities using OpenAI text embeddings
    (text-embedding-3-small) and cosine similarity.
    """
    unique_titles = set()
    for ref in extracted_references:
        title = ref.get("title")
        if title:
            unique_titles.add(title)
            
    unique_titles = list(unique_titles)
    
    # Track which titles map to which canonical title
    canonical_mapping = {}
    natural_titles = []
    
    # 1. Filter out strict SEBI format — these are already canonical
    for title in unique_titles:
        if is_strict_sebi_format(title):
            canonical_mapping[title] = title
        else:
            natural_titles.append(title)
            
    if not natural_titles:
        return extracted_references

    # 2. Get embeddings for all natural language titles; fall back to string similarity
    #    if the provider has no embedding model (EMBED_MODEL=none) or the call fails
    embeddings = None
    if EMBED_MODEL.lower() != "none":
        print(f"Generating embeddings for {len(natural_titles)} entities...")
        try:
            response = client.embeddings.create(input=natural_titles, model=EMBED_MODEL)
            embeddings = [np.array(item.embedding) for item in response.data]
        except Exception as e:
            print(f"Embeddings unavailable ({e}); using string similarity instead")
    if embeddings is None:
        threshold = 0.9

    def similarity(i, j):
        if embeddings is not None:
            return cosine_similarity(embeddings[i], embeddings[j])
        return lexical_similarity(natural_titles[i], natural_titles[j])

    # 3. Cluster using cosine similarity
    resolved = set()
    
    for i, title in enumerate(natural_titles):
        if title in resolved:
            continue
            
        cluster = [title]
        resolved.add(title)
        
        # Compare against remaining unresolved titles
        for j in range(i + 1, len(natural_titles)):
            other_title = natural_titles[j]
            if other_title in resolved:
                continue
                
            score = similarity(i, j)
            
            if score >= threshold:
                cluster.append(other_title)
                resolved.add(other_title)
                
        # Determine the canonical name for this cluster (the longest/most descriptive)
        canonical_name = max(cluster, key=len)
        
        # Map all variants in the cluster to the canonical name
        for variant in cluster:
            canonical_mapping[variant] = canonical_name
            if variant != canonical_name:
                print(f"  [Semantic Merge] '{variant}' -> '{canonical_name}'")
                
    # Update the references array in-place
    for ref in extracted_references:
        original_title = ref.get("title")
        if original_title in canonical_mapping:
            ref["title"] = canonical_mapping[original_title]
            
    return extracted_references

def ingest_data(conn: Neo4jConnection, client: OpenAI, json_file: str, circular_id: str):
    """
    Reads the extracted references and populates the Neo4j Graph.
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File {json_file} not found. Have you run the extractor yet?")
        return

    # 0a. Circulars: link names, short names and circular numbers of the same document,
    #     including documents already in the graph from earlier circulars
    existing = conn.execute_query(
        "MATCH (e:Entity {type: 'circular'}) RETURN e.name AS name, e.aliases AS aliases, "
        "e.number AS number, e.date AS date", read_only=True)
    existing = [dict(e, date=("master" if "master circular" in e["name"].lower() else "circular", e["date"]) if e["date"] else None)
                for e in existing]  # same (kind, date) shape as new references
    resolved = resolve_circulars(data, existing)
    for item in data:
        if item.get("title") in resolved["mapping"]:
            canonical = resolved["mapping"][item["title"]]
            if canonical != item["title"]:
                print(f"  [Alias Merge] '{item['title']}' -> '{canonical}'")
            item["title"] = canonical

    # 0b. Acts / regulations: merge spelling variants (embeddings, or string similarity as a fallback)
    print("Running Entity Resolution on extracted references...")
    data = resolve_entities(data, client, threshold=0.88)

    # Create the root Circular node (subject/date come from the .meta.json main.py writes)
    meta_path = json_file.replace(".json", ".meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    conn.execute_query(
        "MERGE (c:Circular {id: $circular_id}) SET c.subject = $subject, c.date = $date, c.source_pdf = $pdf",
        {"circular_id": circular_id, "subject": meta.get("subject"), "date": meta.get("date"), "pdf": meta.get("source_pdf")})

    for item in data:
        page_number = item.get("page_number", -1)
        title = item.get("title", "Unknown")
        ref_type = item.get("reference_type", "unknown")
        context = item.get("context_snippet", "")
        
        # We model a Clause as the specific occurrence of the text on a page
        # that belongs to the Circular.
        # Stable hash (Python's built-in hash() changes every run), so re-ingesting the same file
        # MERGEs onto the existing clause instead of creating duplicates
        context_hash = hashlib.sha1(context.encode("utf-8")).hexdigest()[:12]
        clause_id = f"{circular_id}_page_{page_number}_{context_hash}"
        
        # 1. Create/Merge the distinct Entity being referenced
        entity_query = """
        MERGE (e:Entity {name: $title})
        SET e.type = $ref_type
        """
        conn.execute_query(entity_query, {"title": title, "ref_type": ref_type})
        # Documents are usually cited by their short name: "SEBI Act, 1992" for the full title
        short = re.sub(r"(?i)securities and exchange board of india", "SEBI", title)
        if short != title:
            conn.execute_query(
                "MATCH (e:Entity {name: $title}) SET e.aliases = [a IN coalesce(e.aliases, []) WHERE a <> $short] + $short",
                {"title": title, "short": short})
        if title in resolved["meta"]:
            m = resolved["meta"][title]
            conn.execute_query(
                "MATCH (e:Entity {name: $title}) SET e.aliases = [a IN coalesce(e.aliases, []) WHERE NOT a IN $aliases] + $aliases, "
                "e.number = $number, e.date = $date",
                {"title": title, "aliases": m["aliases"], "number": m["number"], "date": m["date"]})
        
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
        
        # 4. Link Clause to Entity. The extractor labels each reference 'references' or 'modifies'.
        #    Older JSON files without the label fall back to amendment verbs, ignoring the stock
        #    phrase "as amended from time to time", which does not modify anything.
        if "relation" in item:
            is_modifying = item["relation"] == "modifies"
        else:
            text = context.lower().replace("as amended from time to time", "")
            is_modifying = any(v in text for v in ['amend', 'modify', 'substitute', 'insert', 'repeal', 'replace'])
        
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

    parser = argparse.ArgumentParser(description="Load extracted SEBI references into Neo4j")
    parser.add_argument("json_file", nargs="?", default="sebi_references.json", help="Output of main.py")
    parser.add_argument("--circular-id", default=None, help="ID for the root Circular node (default: the number main.py detected)")
    args = parser.parse_args()
    if not args.circular_id:
        meta_path = args.json_file.replace(".json", ".meta.json")
        detected = json.load(open(meta_path)).get("circular_id") if os.path.exists(meta_path) else None
        args.circular_id = detected or "sample_sebi_circular"

    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    try:
        client = get_client()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    try:
        conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        sys.exit(1)
    
    ingest_data(conn, client, args.json_file, args.circular_id)
    
    conn.close()
