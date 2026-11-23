"""Shows the graph evidence for every expected answer: runs each question's verify_cypher (read-only)
against the live graph and prints the rows next to the expected answer. No LLM calls.

    python evals/verify_datasets.py                 # both datasets
    python evals/verify_datasets.py --only m01 q04  # some questions
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from utils.neo4j_client import Neo4jConnection

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["qa_dataset.json", "multihop_dataset.json"]

if __name__ == "__main__":
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    conn = Neo4jConnection(os.getenv("NEO4J_URI"), os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    for name in DATASETS:
        for item in json.load(open(os.path.join(HERE, name))):
            if a.only and item["id"] not in a.only:
                continue
            print(f"\n[{item['id']}] ({item.get('type', 'single-step')}, {item['split']}) {item['question']}")
            print(f"  expected: {item['expected']}")
            if item.get("min_tool_path") is not None:
                print(f"  minimum tool path: {' -> '.join(item['min_tool_path']) or '(no tool: refuse)'}")
            if not item.get("verify_cypher"):
                print("  graph check: none (unsafe request: the expected behaviour is a refusal)")
                continue
            print(f"  cypher: {item['verify_cypher']}")
            for row in conn.execute_query(item["verify_cypher"], read_only=True):
                print(f"    {json.dumps(row, default=str, ensure_ascii=False)}")
    conn.close()
