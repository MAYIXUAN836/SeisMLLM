import os
from pathlib import Path
import pickle
import numpy as np
import random
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re
from tqdm import tqdm
from split_guard import image_names, load_splits, validate_index

# Configuration
INDEX_PATH = os.environ.get("SEISMLLM_CASE_INDEX", str(Path(__file__).resolve().parent / "v5-knowledge_base_stage1.pkl"))
MODEL_NAME = 'BAAI/bge-base-en-v1.5'

# Soft Constraint Weights - Three Configurations
# Conservative (Default): Prioritize visual similarity, subtle CT influence
ALPHA = 0.45  # Dense
BETA = 0.45   # Sparse
GAMMA = 0.10  # Metadata (CT impact: 0.10×0.6=0.06)

# Balanced (Recommended): Double CT influence while keeping visual dominant
# ALPHA = 0.40  # Dense
# BETA = 0.40   # Sparse
# GAMMA = 0.20  # Metadata (CT impact: 0.20×0.6=0.12) - Stronger CT correction

# Aggressive (High CT Accuracy Only): Strong structural preference
# ALPHA = 0.35  # Dense
# BETA = 0.35   # Sparse  
# GAMMA = 0.30  # Metadata (CT impact: 0.30×0.6=0.18) - Near hard-filter behavior

CONDITIONS = [
    "1_collapse_partial_collapse_or_building_off_foundation",
    "2_building_or_story_leaning",
    "3_racking_damage_to_walls_other_structural_damage",
    "4_chimney_parapet_or_other_falling_hazard",
    "5_ground_slope_movement_or_cracking",
    "6_others"
]

def load_knowledge_base(path, evaluation_records=()):
    with open(path, 'rb') as f:
        kb = pickle.load(f)
    # Validate existing indices too; a filename/comment is not evidence of isolation.
    validate_index(kb, load_splits(), evaluation_records)
    return kb

def construct_stage1_text(item):
    """
    V5: Dense Embedding (BGE).
    Strategy: Combine local visual details (SID) with global damage summary (CDD).
    """
    # 1. SID: 现有的多图描述
    descriptions = item.get("9_image_descriptions", {})
    sid_text = " ".join(descriptions.values()) if descriptions else ""
    
    # 2. CDD: 新增整体损坏描述 (从 6_evaluation 提取 description)
    # 注意: Stage 1 的 CDD 可能是结构化的字典，也可能是简化版 key
    # 如果是字典，提取所有不为空的 description
    cdd_text = ""
    evals = item.get("6_evaluation", {})
    if isinstance(evals, dict):
        cdd_parts = []
        for key, val in evals.items():
            if isinstance(val, dict) and val.get("description"):
                cdd_parts.append(val.get("description"))
        cdd_text = " ".join(cdd_parts)
    elif isinstance(item.get("CDD"), str):
        # 兼容 main_pipeline 里传入来的简化版 key
        cdd_text = item.get("CDD")
    
    # 3. 合并
    final_text = f"Global Damage: {cdd_text}. Visual Details: {sid_text}"
    return final_text.strip()

def construct_full_text(item):
    """
    V5: Sparse Search (BM25).
    Strategy: Full Context (Labels + SID + CDD).
    """
    # ... (前面的 CT, Stories, PO, View 保持不变，都是 Label) ...
    ct = item.get("4_type_of_construction", {}).get("category", "Unknown")
    stories = item.get("5_stories", {}).get("over_ground", "Unknown")
    po = item.get("3_primary_occupation", {}).get("category", "Unknown")
    view = item.get("2_view_type", "Unknown")
    
    # 获取 SID 和 CDD (复用上面的逻辑)
    descriptions = item.get("9_image_descriptions", {})
    sid_text = " ".join(descriptions.values()) if descriptions else "No description available."
    
    cdd_text = ""
    evals = item.get("6_evaluation", {})
    if isinstance(evals, dict):
        cdd_parts = []
        for key, val in evals.items():
            if isinstance(val, dict) and val.get("description"):
                cdd_parts.append(val.get("description"))
        cdd_text = " ".join(cdd_parts)
    elif isinstance(item.get("CDD"), str):
        cdd_text = item.get("CDD")
    
    # ...
    return f"Construction: {ct}. Stories: {stories}. Occupation: {po}. View: {view}. Description: {sid_text}. Damage Summary: {cdd_text}"

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def get_condition_status(item, condition_key):
    status = item.get("6_evaluation", {}).get(condition_key, {}).get("status", "Not visible")
    if "severe" in status.lower(): return "Severe"
    if "moderate" in status.lower(): return "Moderate"
    if "minor" in status.lower() or "none" in status.lower(): return "Minor/None"
    return "None"

def calculate_metadata_score(query, candidate):
    """Soft Constraint Metadata Scoring
    
    Weight Distribution:
    - Construction Type: 60% (soft constraint, not hard filter)
    - View Type: 30% (visual perspective)
    - Stories: 5% (building height)
    - Occupation: 5% (building usage)
    
    Total: normalized to [0, 1]
    """
    score = 0.0
    
    # 1. Construction Type (Soft Constraint - 60%)
    q_ct = query.get("4_type_of_construction", {}).get("category", "Unknown")
    c_ct = candidate.get("4_type_of_construction", {}).get("category", "Unknown")
    if q_ct == c_ct:
        score += 3.0  # Full match
    # Note: No penalty for mismatch - allows visual similarity to compensate
    
    # 2. View Type (30%)
    q_view = query.get("2_view_type", "Unknown")
    c_view = candidate.get("2_view_type", "Unknown")
    if q_view == c_view:
        score += 1.5  # Full match
    elif "Exterior" in q_view and "Exterior" in c_view:
        score += 0.75  # Partial match
        
    # 3. Stories (5%)
    q_stories = query.get("5_stories", {}).get("over_ground", "Unknown")
    c_stories = candidate.get("5_stories", {}).get("over_ground", "Unknown")
    if q_stories == c_stories:
        score += 0.25
        
    # 4. Occupation (5%)
    q_occ = query.get("3_primary_occupation", {}).get("category", "Unknown")
    c_occ = candidate.get("3_primary_occupation", {}).get("category", "Unknown")
    if q_occ == c_occ:
        score += 0.25
        
    return score / 5.0  # Normalize to [0, 1] (max score = 5.0)

def retrieve_hybrid_structured(query_case, kb, model, top_k=1):
    query_images = image_names(query_case)
    library_embeddings = kb['embeddings']
    library_meta = kb['metadata']
    bm25 = kb['bm25_model']
    
    # 1. Dense (Description + CDD)
    query_desc = construct_stage1_text(query_case)
    query_embedding = model.encode([query_desc], normalize_embeddings=True)
    dense_scores = np.dot(query_embedding, library_embeddings.T)[0]
    
    # 2. Sparse (Full Context with CDD)
    query_full = construct_full_text(query_case)
    tokenized_query = tokenize(query_full)
    sparse_scores = np.array(bm25.get_scores(tokenized_query))
    if sparse_scores.max() > 0 and sparse_scores.max() > sparse_scores.min():
        sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min())
    else:
        sparse_scores = np.zeros_like(dense_scores)
        
    # 3. Metadata Scoring (On-the-fly with Soft Constraints)
    # No hard filtering on CT - allow visual similarity to compensate for CT errors
    meta_scores = np.zeros_like(dense_scores)
    
    candidates = []
    
    for idx, meta in enumerate(library_meta):
        # Exclude the same building case even when IDs/folders differ or views are a subset.
        if image_names(meta['original_data']) & query_images:
            continue
        # Removed hard CT filter - now using soft constraint in metadata score
        
        # Calculate Metadata Score (includes CT as soft constraint)
        m_score = calculate_metadata_score(query_case, meta['original_data'])
        
        # Fusion: Dense + Sparse + Metadata
        final_score = (ALPHA * dense_scores[idx]) + (BETA * sparse_scores[idx]) + (GAMMA * m_score)
        
        candidates.append({
            "id": meta['id'],
            "meta": meta,
            "score": float(final_score)
        })
        
    if not candidates: return []
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:top_k]

def main():
    print("Loading Knowledge Base...")
    kb = load_knowledge_base(INDEX_PATH)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    random.seed(42)
    total_cases = len(kb['metadata'])
    test_indices = random.sample(range(total_cases), 100)
    
    print(f"\n=== Version 5: Structured Hybrid with CDD Evaluation (N=100) ===")
    print(f"Weights: Dense={ALPHA}, Sparse={BETA}, Metadata={GAMMA}")
    print(f"Strategy: Dense[SID+CDD] + Sparse[Labels+SID+CDD] + Metadata")
    
    total_cond = 0
    total_safe = 0
    
    for idx in tqdm(test_indices):
        query_meta = kb['metadata'][idx]
        query_data = query_meta['original_data']
        
        results = retrieve_hybrid_structured(query_data, kb, model, top_k=1)
        if not results: continue
        
        res_data = results[0]['meta']['original_data']
        
        # Condition
        match = 0
        for key in CONDITIONS:
            if get_condition_status(query_data, key) == get_condition_status(res_data, key):
                match += 1
        total_cond += (match / 6)
        
        # Safety
        q_sp = query_data.get("8_final_status", {}).get("status", "Unknown")
        r_sp = res_data.get("8_final_status", {}).get("status", "Unknown")
        if q_sp == r_sp: total_safe += 1
            
    avg_cond = (total_cond / 100) * 100
    avg_safe = (total_safe / 100) * 100
    
    print("\n" + "="*40)
    print(f"{'Metric':<25} | {'V4':<12} | {'V5 (with CDD)':<15}")
    print("-" * 60)
    print(f"{'Condition Match Rate':<25} | {'71.67%':<12} | {avg_cond:.2f}%")
    print(f"{'Safety Posting Match':<25} | {'69.00%':<12} | {avg_safe:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
