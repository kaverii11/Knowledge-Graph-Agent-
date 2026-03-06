import argparse
import json
import difflib

def normalize_text(text: str) -> str:
    """Lowercase and remove extra whitespaces."""
    return " ".join(text.lower().split())

def is_match(agent_title: str, gt_title: str, threshold: float = 0.8) -> bool:
    """
    Check if the agent title matches the ground truth title.
    Supports exact (normalized) match, subset match, and fuzzy matching.
    """
    agent_norm = normalize_text(agent_title)
    gt_norm = normalize_text(gt_title)
    
    # Exact match after normalization
    if agent_norm == gt_norm:
        return True
        
    # Subset match (e.g. "sebi act 1992" vs "the sebi act 1992")
    if agent_norm in gt_norm or gt_norm in agent_norm:
        return True
        
    # Fuzzy match for typos
    ratio = difflib.SequenceMatcher(None, agent_norm, gt_norm).ratio()
    return ratio >= threshold

def run_evals(ground_truth_path: str, agent_output_path: str):
    try:
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
    except Exception as e:
        print(f"Error loading {ground_truth_path}: {e}")
        return

    try:
        with open(agent_output_path, 'r', encoding='utf-8') as f:
            agent_output = json.load(f)
    except Exception as e:
        print(f"Error loading {agent_output_path}: {e}")
        return

    print(f"Loaded {len(ground_truth)} ground truth references.")
    print(f"Loaded {len(agent_output)} agent extracted references.\n")

    true_positives = []
    false_positives = []
    
    # Track which ground truth items have been matched
    gt_matched = [False] * len(ground_truth)
    
    for agent_item in agent_output:
        agent_title = agent_item.get("title", "")
        agent_page = agent_item.get("page_number", -1)
        
        match_found = False
        for i, gt_item in enumerate(ground_truth):
            if gt_matched[i]:
                continue
                
            gt_title = gt_item.get("title", "")
            gt_page = gt_item.get("page_number", -1)
            
            # Match condition: Same page AND fuzzy title match
            if agent_page == gt_page and is_match(agent_title, gt_title):
                match_found = True
                gt_matched[i] = True
                true_positives.append(agent_item)
                break
                
        if not match_found:
            false_positives.append(agent_item)
            
    false_negatives_count = sum(1 for m in gt_matched if not m)
    
    tp_count = len(true_positives)
    fp_count = len(false_positives)
    fn_count = false_negatives_count
    
    # Metrics
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Confidence Score Analysis
    avg_conf_tp = sum(item.get("confidence_score", 0.0) for item in true_positives) / tp_count if tp_count > 0 else 0.0
    avg_conf_fp = sum(item.get("confidence_score", 0.0) for item in false_positives) / fp_count if fp_count > 0 else 0.0

    print("="*40)
    print(" EVALUATION METRICS REPORT ")
    print("="*40)
    print(f"True Positives (TP):  {tp_count}")
    print(f"False Positives (FP): {fp_count}")
    print(f"False Negatives (FN): {fn_count}")
    print("-" * 40)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")
    print("="*40)
    print(" CONFIDENCE SCORE ANALYSIS ")
    print("="*40)
    if tp_count > 0:
        print(f"Avg Confidence of True Positives:  {avg_conf_tp:.4f}")
    else:
        print("Avg Confidence of True Positives:  N/A (No TPs)")
        
    if fp_count > 0:
        print(f"Avg Confidence of False Positives: {avg_conf_fp:.4f}")
    else:
        print("Avg Confidence of False Positives: N/A (No FPs)")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Agent Extractions against Ground Truth")
    parser.add_argument("ground_truth", type=str, help="Path to ground_truth.json")
    parser.add_argument("agent_output", type=str, help="Path to agent_output.json (from main.py)")
    
    args = parser.parse_args()
    run_evals(args.ground_truth, args.agent_output)
