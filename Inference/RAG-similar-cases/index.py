import os
from pathlib import Path
import json
import numpy as np
import pickle
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re
from split_guard import dataset_directory, load_splits, select_training_records, validate_index

# Configuration
JSON_PATH = os.environ.get("SEISMLLM_CASES_JSON", str(dataset_directory() / "Full_GT.json"))
INDEX_OUTPUT_PATH = os.environ.get("SEISMLLM_CASE_INDEX", str(Path(__file__).resolve().parent / "v5-knowledge_base_stage1.pkl"))
MODEL_NAME = 'BAAI/bge-base-en-v1.5'

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

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
    """Simple tokenizer for BM25"""
    return re.findall(r'\w+', text.lower())

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    
    print(f"Loading raw data from {JSON_PATH}...")
    raw_data = load_data(JSON_PATH)
    # Full_GT supplies annotations; only cases in train.json may enter the index.
    splits = load_splits()
    raw_data = select_training_records(raw_data, splits)
    print(f"Selected {len(raw_data)} training cases; released test cases are excluded.")
    
    documents_dense = []  # For dense embedding (descriptions + CDD)
    documents_full = []   # For BM25 (full text)
    tokenized_corpus = []
    metadata_list = []
    
    print("Processing cases for V5 Structured Hybrid Index (with CDD)...")
    for item in raw_data:
        try:
            text_dense = construct_stage1_text(item)
            text_full = construct_full_text(item)
            
            documents_dense.append(text_dense)
            documents_full.append(text_full)
            tokenized_corpus.append(tokenize(text_full))
            
            meta = {
                "id": item.get("case_id", "Unknown"),
                "construction_type": item.get("4_type_of_construction", {}).get("category", "Unknown"),
                "final_status": item.get("8_final_status", {}),
                "damage_conditions": item.get("6_evaluation", {}),
                "view_type": item.get("2_view_type", "Unknown"),
                "stories": item.get("5_stories", {}),
                "occupation": item.get("3_primary_occupation", {}),
                "original_data": item
            }
            metadata_list.append(meta)
            
        except Exception as e:
            continue
            
    print(f"Processed {len(documents_dense)} valid cases.")

    # --- 1. Dense Index (BGE on descriptions + CDD) ---
    print(f"Building Dense Index ({MODEL_NAME}) with SID+CDD...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    embeddings = model.encode(documents_dense, normalize_embeddings=True, show_progress_bar=True)
    
    # --- 2. Sparse Index (BM25 on full text) ---
    print("Building Sparse Index (BM25) with Labels+SID+CDD...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    knowledge_base = {
        "model_name": MODEL_NAME,
        "embeddings": embeddings,       # Dense Vectors (SID + CDD)
        "bm25_model": bm25,             # Sparse Model (full context)
        "metadata": metadata_list       # Enhanced metadata for structured scoring
    }
    
    validate_index(knowledge_base, splits)
    print(f"Saving V5 Structured Hybrid Knowledge Base to {INDEX_OUTPUT_PATH}...")
    with open(INDEX_OUTPUT_PATH, 'wb') as f:
        pickle.dump(knowledge_base, f)
        
    print("✅ V5 Structured Hybrid Index (Dense[SID+CDD] + Sparse[Labels+SID+CDD] + Metadata) Built Successfully!")

if __name__ == "__main__":
    main()
