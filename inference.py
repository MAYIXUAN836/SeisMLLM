"""SeisMLLM multi-view inference with similar-case and codebook retrieval."""

import sys
import os
import json
import asyncio
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import importlib.util
import pickle
from PIL import Image
import re
import time
from datetime import datetime

# 导入性能监控模块（来自作者内部工具，开源仓库中可能不存在）


# --- Configuration ---
# 默认使用当前仓库作为 PROJECT_ROOT；如需兼容外部仓库结构，可通过
# 环境变量 SEISMLLM_PROJECT_ROOT / SEISMLLM_RAG_* 覆盖。
PROJECT_ROOT = os.environ.get(
    "SEISMLLM_PROJECT_ROOT",
    str(Path(__file__).resolve().parent),
)

INFERENCE_DIR = Path(PROJECT_ROOT) / "Inference"

OPEN_SOURCE_DIR = os.environ.get(
    "SEISMLLM_OPEN_SOURCE_DIR",
    str(INFERENCE_DIR),  # 默认仅使用当前 Inference 目录
)

RAG_SIMILAR_DIR = os.environ.get(
    "SEISMLLM_RAG_SIMILAR_DIR",
    str(INFERENCE_DIR / "RAG-similar-cases"),
)

RAG_ATC_DIR = os.environ.get(
    "SEISMLLM_RAG_ATC_DIR",
    str(INFERENCE_DIR / "RAG-ATC"),
)

RAG_ANYTHING_DIR = os.environ.get(
    "SEISMLLM_RAG_ANYTHING_DIR",
    str(Path(RAG_ATC_DIR) / "RAG-Anything"),
)

# Add paths
for _p in [OPEN_SOURCE_DIR, RAG_SIMILAR_DIR, RAG_ATC_DIR, RAG_ANYTHING_DIR]:
    if _p and _p not in sys.path:
        sys.path.append(_p)

# --- Imports ---
from performance_monitor import PerformanceMonitor, estimate_tokens_from_json_output

# 1. Stage 1: Qwen Inference
try:
    from qwen import QwenLoRAInference, normalize_prediction, BASE_MODEL_PATH, LORA_PATH
except ImportError:
    print("Error importing QwenLoRAInference. Check paths.")
    sys.exit(1)

# 2. Stage 2a: Similar Case Retrieval
# We need to import this dynamically or carefully because of filename
spec = importlib.util.spec_from_file_location("v5_retrieve", os.path.join(RAG_SIMILAR_DIR, "retrieve.py"))
v5_retrieve = importlib.util.module_from_spec(spec)
sys.modules["v5_retrieve"] = v5_retrieve
spec.loader.exec_module(v5_retrieve)

retrieve_hybrid_structured = v5_retrieve.retrieve_hybrid_structured
SentenceTransformer = v5_retrieve.SentenceTransformer

# 3. Stage 2b: ATC Rule Retrieval
# We need to setup RAGAnything
from raganything import RAGAnything, RAGAnythingConfig
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
import openai

# Load local runtime configuration; credentials come from environment variables.
from config import (
    API_KEY,
    BASE_URL,
    LLM_MODEL,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    KNOWLEDGE_BASE_DIR,
)

# --- Helper Functions ---

def extract_case_info_from_image(image_path: str):
    """从图片路径中解析 earthquake_folder 和 case_id，用于日志和评估对齐。

    兼容两种路径：
    - .../dataset/<earthquake_folder>/<case_id>/<filename>
    - .../VLM finetune dataset/<earthquake_folder>/<case_id>/<filename>
    """
    if not image_path:
        return "Unknown", "Unknown"

    parts = image_path.replace("\\", "/").split("/")
    try:
        if "dataset" in parts:
            idx = parts.index("dataset")
        else:
            idx = parts.index("VLM finetune dataset")
        if idx + 2 < len(parts):
            folder = parts[idx + 1]
            case_id = parts[idx + 2]
            return folder, case_id
    except ValueError:
        pass
    return "Unknown", "Unknown"

def contains_chinese(text):
    """检测文本是否包含中文字符"""
    if not isinstance(text, str):
        return False
    import re
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def validate_and_fix_stage1_output(output):
    """
    验证并修正Stage 1输出
    - 检测中文输出
    - 标准化分类值（IS/PO/CT）
    - 返回修正后的输出和问题列表
    """
    issues = []
    fixed_output = output.copy()
    
    # 标准分类映射（常见中文 → 英文标准类目）
    STANDARD_MAPPINGS = {
        'CT': {
            '玻璃幕墙结构': 'Steel frame',
            '钢筋砌体': 'Reinforced masonry',
            '混凝土框架': 'Concrete frame',
            '钢框架': 'Steel frame',
            '木结构': 'Wood frame',
        },
        'PO': {
            '办公室': 'Office',
            '办公': 'Office',
            '商业': 'Commercial',
            '住宅': 'Dwelling',
            '学校': 'School',
            '公共集会': 'Public assembly',
        },
        'IS': {
            '商业建筑': 'Commercial',
            '住宅建筑': 'Residential',
            '学校': 'School',
            '办公楼': 'Office',
        }
    }

    # 检查并修正IS
    if 'IS' in output and contains_chinese(output['IS']):
        issues.append(f"IS contains Chinese: {output['IS']}")
        for cn, en in STANDARD_MAPPINGS['IS'].items():
            if cn in output['IS']:
                fixed_output['IS'] = en
                issues.append(f"  → Fixed to: {en}")
                break

    # 检查并修正CT
    if 'CT' in output and contains_chinese(output['CT']):
        issues.append(f"CT contains Chinese: {output['CT']}")
        for cn, en in STANDARD_MAPPINGS['CT'].items():
            if cn in output['CT']:
                fixed_output['CT'] = en
                issues.append(f"  → Fixed to: {en}")
                break

    # 检查并修正PO
    if 'PO' in output:
        po_text = output['PO']
        if contains_chinese(po_text):
            issues.append(f"PO contains Chinese: {po_text}")
            for cn, en in STANDARD_MAPPINGS['PO'].items():
                if cn in po_text:
                    fixed_output['PO'] = en
                    issues.append(f"  → Fixed to: {en}")
                    break
        elif isinstance(po_text, str) and len(po_text) > 50:
            # 太长的描述性文本会影响检索，统一压成 Unknown
            issues.append(f"PO is too descriptive: {po_text[:50]}...")
            fixed_output['PO'] = 'Unknown'

    # 检查SN
    if 'SN' in output and contains_chinese(output['SN']):
        issues.append(f"SN contains Chinese: {output['SN']}")
        if '未提供' in output['SN'] or '不清楚' in output['SN']:
            fixed_output['SN'] = 'Over ground: Unknown'

    # DC 和 SID 先只标记问题，翻译放到后面的异步流程里
    if 'DC' in output and contains_chinese(output['DC']):
        issues.append(f"DC contains Chinese: {output['DC']}")

    if 'SID' in output and isinstance(output['SID'], str) and contains_chinese(output['SID']):
        issues.append("SID contains Chinese")
        if '无明显损坏' in output['SID'] or '无' in output['SID']:
            fixed_output['SID'] = 'No visible damage from images'

    return fixed_output, issues

async def translate_to_english_if_needed(text: str) -> str:
    """如包含中文，则调用配置的语言模型翻译为英文，仅用于内部检索/推理。

    - 如果没有中文，直接返回原文
    - 翻译时强制要求只返回英文内容
    """
    if not isinstance(text, str) or not text.strip():
        return text
    if not contains_chinese(text):
        return text

    system_prompt = (
        "You are a professional translator. "
        "Translate the user content to concise English only. "
        "DO NOT explain, DO NOT add any extra text. "
        "Return ONLY the translated English sentence(s)."
    )

    try:
        translated = await gptsapi_llm_enhanced(text, system_prompt=system_prompt, max_tokens=256)
        # 防止模型又加解释，这里做一次简单截断
        return translated.strip()
    except Exception:
        # 翻译失败就退回原文，至少不中断流程
        return text

async def gptsapi_llm_enhanced(prompt, system_prompt=None, **kwargs):
    client = openai.AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    if isinstance(prompt, dict) and 'content' in prompt:
        messages.append({"role": "user", "content": prompt['content']})
    else:
        messages.append({"role": "user", "content": prompt})
    
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 
                              'presence_penalty', 'stop', 'n', 'stream']}
    
    response = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        **filtered_kwargs
    )
    return response.choices[0].message.content

async def gptsapi_embed(texts):
    client = openai.AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    if isinstance(texts, str):
        texts = [texts]
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    # Return as numpy array for LightRAG compatibility
    return np.array(embeddings)

embedding_func = EmbeddingFunc(
    embedding_dim=EMBEDDING_DIM,
    max_token_size=8192,
    func=gptsapi_embed
)

async def english_only_llm(prompt, system_prompt=None, **kwargs):
    """包装底层LLM，强制所有回答为英文。

    这是提供给 LightRAG / RAGAnything 使用的 llm_model_func，
    以避免ATC规则回答里出现中文。
    """
    enforced_system = "You MUST answer in English only. Do NOT use Chinese in any form."
    if system_prompt:
        enforced_system += "\n" + str(system_prompt)
    return await gptsapi_llm_enhanced(prompt, system_prompt=enforced_system, **kwargs)


def full_json_to_stage1_fields(full_json: dict) -> dict:
    """将Qwen输出的完整ATC-20 JSON压缩成 Stage1 需要的字段(IS, PO, CT, SN, DC, SID等)。"""

    view_type = full_json.get("2_view_type", "Unknown")

    po_obj = full_json.get("3_primary_occupation", {}) or {}
    po_cat = po_obj.get("category", "Unknown")

    ct_obj = full_json.get("4_type_of_construction", {}) or {}
    ct_cat = ct_obj.get("category", "Unknown")

    stories = full_json.get("5_stories", {}) or {"over_ground": None, "under_ground": None}
    over_g = stories.get("over_ground")
    if over_g is not None:
        sn = f"Over ground: {over_g}"
    else:
        sn = "Over ground: None"

    evals = full_json.get("6_evaluation", {}) or {}
    dc_parts = []
    cond_map = [
        ("collapse", "1_collapse_partial_collapse_or_building_off_foundation"),
        ("building", "2_building_or_story_leaning"),
        ("racking", "3_racking_damage_to_walls_other_structural_damage"),
        ("chimney", "4_chimney_parapet_or_other_falling_hazard"),
        ("ground", "5_ground_slope_movement_or_cracking"),
        ("others", "6_others"),
    ]
    for short_key, long_key in cond_map:
        cond = evals.get(long_key, {}) or {}
        status = cond.get("status", "Unknown")
        desc = cond.get("description", "")
        # The DC string includes condition statuses and supporting descriptions.
        if desc:
            dc_parts.append(f"{short_key}: {status} ({desc})")
        else:
            dc_parts.append(f"{short_key}: {status}")
    dc = "; ".join(dc_parts)

    img_desc = full_json.get("9_image_descriptions", {}) or {}
    sid_list = []
    for fname, desc in img_desc.items():
        sid_list.append(f"{fname}: {desc}")

    stage1 = {
        "IS": view_type,
        "PO": po_cat,
        "CT": ct_cat,
        "SN": sn,
        "DC": dc,
        "SID": sid_list,
        "5_stories": stories,
        "6_evaluation": evals,
    }
    return stage1

# --- Main Pipeline Class ---

class Pipeline:
    def __init__(self):
        self.qwen_engine = None
        self.similar_case_kb = None
        self.similar_case_model = None
        self.rag_atc = None
        
    async def initialize(self):
        print("Initializing Stage 1: Qwen Model...")
        self.qwen_engine = QwenLoRAInference(BASE_MODEL_PATH, LORA_PATH)
        self.qwen_engine.load_model()
        
        print("Initializing Stage 2a: Similar Case KB (Stage1-only)...")
        # 使用只包含 Stage1 GT（并且排除 130 个 test case）的新 KB
        kb_path = os.environ.get("SEISMLLM_CASE_INDEX", os.path.join(RAG_SIMILAR_DIR, "v5-knowledge_base_stage1.pkl"))
        with open(kb_path, 'rb') as f:
            self.similar_case_kb = pickle.load(f)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.similar_case_model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)
        
        print("Initializing Stage 2b: ATC RAG...")
        # Initialize LightRAG separately first，强制其LLM只输出英文
        self.lightrag_atc = LightRAG(
            working_dir=KNOWLEDGE_BASE_DIR,
            llm_model_func=english_only_llm,
            embedding_func=embedding_func,
        )
        
        # Initialize LightRAG storages (required for async operations)
        await self.lightrag_atc.initialize_storages()
        
        # Then create RAGAnything with the pre-initialized LightRAG
        config = RAGAnythingConfig(
            working_dir=KNOWLEDGE_BASE_DIR,
            enable_image_processing=False, # We only query text rules
        )
        self.rag_atc = RAGAnything(
            lightrag=self.lightrag_atc,  # Pass pre-initialized LightRAG
            config=config,
            llm_model_func=english_only_llm,
            embedding_func=embedding_func
        )
        
    async def run_stage1(self, images, system_prompt, user_prompt):
        """Stage 1: 调用 Qwen 输出完整 ATC-20 JSON，并提取 Stage1 精简字段。

        返回值为 (full_pred, stage1_fixed)：
        - full_pred: 归一化后的完整 Qwen JSON（用于评估 Stage1，保持与 baseline 一致）
        - stage1_fixed: 仅包含 IS/PO/CT/SN/DC/SID 等字段，用于后续 RAG 检索
        """
        # Qwen inference
        # Note: generate_json is synchronous in the original script, but we are in async loop.
        # It's fine to call it directly if it doesn't block too long, or run in executor.
        # Since it uses GPU, it blocks. For simplicity, we call it directly.
        print("\n" + "="*80)
        print("STAGE 1: Qwen-7B Inference")
        print("="*80)
        print(f"Processing {len(images)} images...")
        
        # 使用与 inference_qwen_7b_finetuned.py 一致的调用方式：
        # 直接传入 utils_qwen_finetune.get_inference_system_prompt() 给出的 system_prompt，
        # 和类似 get_inference_dataset 构造的 user_prompt，让模型输出完整ATC-20 JSON。
        output_text = self.qwen_engine.generate_json(images, system_prompt, user_prompt)
        
        # 估算token数
        token_count = estimate_tokens_from_json_output(output_text)
        
        # Parse JSON
        output_text = output_text.strip()
        if not output_text.startswith('{'):
            start_idx = output_text.find('{')
            if start_idx != -1:
                output_text = output_text[start_idx:]
        if not output_text.endswith('}'):
            end_idx = output_text.rfind('}')
            if end_idx != -1:
                output_text = output_text[:end_idx+1]
        
        try:
            full_pred = json.loads(output_text)
            full_pred = normalize_prediction(full_pred)

            # 从完整JSON中提取Stage1所需的精简字段（仅供 RAG 使用，不改变 full_pred 本身）
            stage1_raw = full_json_to_stage1_fields(full_pred)

            # Validate and fix Stage1 精简字段（主要是语言/分类归一化）
            fixed_stage1, issues = validate_and_fix_stage1_output(stage1_raw)

            if issues:
                print("\n⚠ Stage 1 Output Issues Detected:")
                for issue in issues:
                    print(f"  {issue}")

            print("\n✓ Stage 1 Output (after validation, Stage1 fields only):")
            print(json.dumps(fixed_stage1, indent=2, ensure_ascii=False))

            # 返回完整 Qwen 预测 + 精简 Stage1 字段 + token数
            return full_pred, fixed_stage1, token_count
        except Exception as e:
            print(f"✗ Stage 1 JSON Parse Error: {e}")
            return None

    def run_stage2a(self, stage1_json, test_case_id="Unknown"):
        # Similar Case Retrieval
        print("\n" + "="*80)
        print("STAGE 2a: Similar Case Retrieval (v4-structured)")
        print("="*80)
        
        # Convert Stage 1 simplified format to full format expected by retrieval
        # Stage 1 outputs: IS, PO, CT, SN, DC, SID
        # Retrieval expects: 2_view_type, 3_primary_occupation, 4_type_of_construction, 5_stories, 9_image_descriptions
        
        converted_case = {
            "case_id": test_case_id,  # Add case_id to prevent self-retrieval
            "2_view_type": stage1_json.get("IS", "Unknown"),
            "3_primary_occupation": {
                "category": stage1_json.get("PO", "Unknown")
            },
            "4_type_of_construction": {
                "category": stage1_json.get("CT", "Unknown")
            },
            "5_stories": stage1_json.get("5_stories", {"over_ground": None, "under_ground": None}),
            "9_image_descriptions": {}
        }
        
        # Convert SID list to dict format
        sid = stage1_json.get("SID", [])
        if isinstance(sid, list):
            for i, desc in enumerate(sid, 1):
                converted_case["9_image_descriptions"][f"image{i}"] = desc
        
        # Parse SN (e.g., "Over ground: 4") to extract number
        sn = stage1_json.get("SN", "")
        if ":" in str(sn):
            try:
                num = int(sn.split(":")[-1].strip())
                converted_case["5_stories"]["over_ground"] = num
            except:
                pass
        
        # Debug: print what we're searching with
        print(f"\nConverted Stage 1 format for retrieval:")
        print(f"  - View Type: {converted_case['2_view_type']}")
        print(f"  - Primary Occupation: {converted_case['3_primary_occupation']['category']}")
        print(f"  - Construction Type: {converted_case['4_type_of_construction']['category']}")
        print(f"  - Stories (over ground): {converted_case['5_stories']['over_ground']}")
        print(f"  - Image descriptions: {len(converted_case['9_image_descriptions'])} items")
        
        results = retrieve_hybrid_structured(converted_case, self.similar_case_kb, self.similar_case_model, top_k=3)
        if results:
            similar_case = results[0]
            print(f"\n✓ Found similar case (score: {similar_case.get('score', 'N/A'):.4f})")
            print(f"  Earthquake: {similar_case['meta']['original_data'].get('1_earthquake_id', {}).get('name', 'Unknown')}")
            print(f"  Case ID: {similar_case['meta']['original_data'].get('case_id', 'Unknown')}")
            print(f"  Final Status: {similar_case['meta']['original_data'].get('8_final_status', {}).get('status', 'Unknown')}")
            print(f"\n  Similar case key fields:")
            print(f"  - Primary Occupation: {similar_case['meta']['original_data'].get('3_primary_occupation', {}).get('category', 'Unknown')}")
            print(f"  - Construction Type: {similar_case['meta']['original_data'].get('4_type_of_construction', {}).get('category', 'Unknown')}")
            print(f"  - Stories: {similar_case['meta']['original_data'].get('5_stories', {})}") 
            print(f"  - Estimated Damage: {similar_case['meta']['original_data'].get('7_estimated_building_damage', 'Unknown')}")
            return similar_case
        
        print("\n✗ No similar case found")
        return None

    async def run_stage2b(self, stage1_json):
        # ATC Rule Retrieval
        print("\n" + "="*80)
        print("STAGE 2b: ATC-20 Rules Retrieval")
        print("="*80)

        # Construct query from Stage 1 JSON - use actual Stage 1 field names
        # Stage 1 outputs: IS, PO, CT, SN, DC, SID
        ct_raw = stage1_json.get("CT", "Unknown")  # Construction Type
        po_raw = stage1_json.get("PO", "Unknown")  # Primary Occupation
        dc_raw = stage1_json.get("DC", "Unknown")  # Damage Conditions
        sid_raw = stage1_json.get("SID", [])  # Image descriptions

        # Stage3 Qwen模型输出已经是英文，直接使用
        ct = ct_raw
        po = po_raw
        dc = dc_raw
        sid = sid_raw

        # Build a comprehensive query based on Stage 1 results (now English-only as far as possible)
        damage_details = ", ".join(sid[:2]) if isinstance(sid, list) and sid else dc

        query = f"""Safety evaluation for {ct} building used as {po}. 
        Observed damage: {dc} severity. 
        Specific observations: {damage_details}. 
        What are the ATC-20 evaluation criteria and posting requirements?"""

        print(f"Query: {query[:200]}...")

        # Query RAG
        try:
            # Use the asynchronous query method
            result = await self.rag_atc.aquery(query, mode="hybrid")  # Use hybrid mode with async query
            if result:
                print(f"\n✓ Retrieved ATC-20 rules ({len(result)} chars):")
                # Show first 600 chars
                print(f"{result[:600]}..." if len(result) > 600 else result)
                return result
            else:
                print(f"\n✗ Stage 2b: No results returned")
                return ""
        except Exception as e:
            print(f"\n✗ Stage 2b Error: {e}")
            return ""

    async def run_stage3(self, stage1_full_json, stage1_for_gpt, similar_cases_list, atc_rules):
        """Stage 3: Qwen基于description + similar cases + ATC判断status和SP。"""
        """Stage 3: 只预测 DC (6_evaluation) 和 SP (8_final_status)，其余字段保持 Qwen Stage1 不变。"""

        # 构造相似cases文本摘要（只用 Stage1 相关字段）
        similar_case_text = "No similar cases retrieved."
        if similar_cases_list:
            sim_lines = []
            for rank, similar_case in enumerate(similar_cases_list[:2], 1):  # 只用top1和top2
                sc_meta = similar_case.get("meta", {})
                sc_data = sc_meta.get("original_data", {})
                sc_eq = sc_meta.get("earthquake_folder", "Unknown")
                sc_id = sc_meta.get("case_id", sc_meta.get("id", "Unknown"))
                sc_score = similar_case.get("score", "N/A")
                sc_po = sc_data.get("3_primary_occupation", {}).get("category", "Unknown")
                sc_ct = sc_data.get("4_type_of_construction", {}).get("category", "Unknown")
                sc_stories = sc_data.get("5_stories", {}).get("over_ground", "Unknown")
                sc_descriptions = sc_data.get("9_image_descriptions", {}) or {}
                
                if rank == 1:
                    sim_lines.append(f"=== Top {rank} Similar Case (MOST SIMILAR, score: {sc_score:.4f}) ===")
                else:
                    sim_lines.append(f"=== Top {rank} Similar Case (score: {sc_score:.4f}) ===")
                
                sim_lines.extend([
                    f"  - Earthquake: {sc_eq}",
                    f"  - Case ID: {sc_id}",
                    f"  - Primary Occupation: {sc_po}",
                    f"  - Construction Type: {sc_ct}",
                    f"  - Stories: {sc_stories}",
                    "  - Image descriptions:",
                ])
                for fname, desc in sc_descriptions.items():
                    sim_lines.append(f"    * {fname}: {desc}")
                sim_lines.append("")  # 空行分隔

            similar_case_text = "\n".join(sim_lines)

        if not atc_rules:
            atc_rules = "No ATC-20 rules retrieved."

        prompt_content = f"""
You are an expert structural engineer performing ATC-20 post-earthquake safety evaluation.

You are given:
1. Stage 1 understanding of the target building (view type, primary occupation, construction type, stories, image descriptions).
2. Top 1 and Top 2 similar cases from the training dataset (with GT labels).
   - Top 1 is the MOST SIMILAR case (highest similarity score)
   - Top 2 is the second most similar case (lower similarity score)
   - Use Top 1 as the primary reference, and Top 2 as a secondary reference when patterns are ambiguous.
3. ATC-20-1 Field Manual rules and criteria (text snippets).

Your tasks:
- Infer the six damage conditions (DC) in field "6_evaluation".
- Infer the final safety posting (SP) in field "8_final_status".
- The similar case can be used as a reference for typical damage patterns and posting decisions.
- The ATC-20 rules must be used as the decisive criteria for SP (Green / Yellow / Red) and the reasoning.
- NOTE: The top2 retrieved cases are ranked by similarity score. Reference with decreasing confidence 
  as similarity decreases. The first case should be more heavily weighted in your judgment than the second.

STRICT RULES FOR 6_evaluation (Damage Conditions):
- For EACH of the six conditions, the field "status" MUST be EXACTLY ONE of:
    "Minor/none" | "Moderate" | "Severe" | "Not visible from the provided images".
- You do NOT need to rewrite the descriptions in 6_evaluation. Focus only on choosing the correct status
    for each condition based on Stage 1 observations, the similar case and ATC-20 rules.

ATC-20 TABLE 3-1 STYLE RULES FOR 8_final_status.status:
- The field "status" MUST be EXACTLY ONE of:
        "(1) Inspected (Green)" | "(2) Restricted Use (Yellow)" | "(3) Unsafe (Red)".
- The decision MUST follow ATC-20 / Table 3-1 style logic:
    * "(3) Unsafe (Red)":
            - If ANY of conditions 1, 2, 3, or 5 in 6_evaluation has status "Severe", OR
            - If condition 6 (others) indicates serious life-safety hazards (e.g., falling hazards, gas leaks).
    * "(2) Restricted Use (Yellow)":
            - If damage is localized or moderate but may limit safe occupancy, OR
            - If condition 4 (chimney/parapet/falling hazard) is "Severe" but the overall structural system is still stable.
    * "(1) Inspected (Green)":
            - If there are NO "Severe" conditions in 6_evaluation, and
            - All observed damage is at most "Moderate" and does not compromise global stability.

- The field "reasoning" in 8_final_status MUST provide a step-by-step explanation
    of how the statuses in 6_evaluation lead to the chosen posting,
    explicitly referencing ATC-20 style criteria (consistent with Table 3-1 logic).

=== Stage 1 (Target Building) ===
{json.dumps(stage1_for_gpt, indent=2, ensure_ascii=False)}

=== Similar Cases (GT from Training Set) ===
{similar_case_text}

=== ATC-20 Rules Context ===
{atc_rules}

Now output ONLY a JSON object with the following structure (nothing else):
{{
    "6_evaluation": {{
        "1_collapse_partial_collapse_or_building_off_foundation": {{"status": "<Minor/none | Moderate | Severe | Not visible from the provided images>"}},
        "2_building_or_story_leaning": {{"status": "<Minor/none | Moderate | Severe | Not visible from the provided images>"}},
        "3_racking_damage_to_walls_other_structural_damage": {{"status": "<Minor/none | Moderate | Severe | Not visible from the provided images>"}},
        "4_chimney_parapet_or_other_falling_hazard": {{"status": "<Minor/none | Moderate | Severe | Not visible from the provided images>"}},
        "5_ground_slope_movement_or_cracking": {{"status": "<Minor/none | Moderate | Severe | Not visible from the provided images>"}},
        "6_others": {{"status": "<Minor/none | Moderate | Severe | Not visible from the provided images>"}}
    }},
    "8_final_status": {{
        "status": "<(1) Inspected (Green) | (2) Restricted Use (Yellow) | (3) Unsafe (Red)",
        "reasoning": "<concise reasoning consistent with ATC-20 Table 3-1 and the above 6_evaluation>"
    }}
}}

Strict requirements:
- Do NOT include any other fields (no 1_earthquake_id, no 2_view_type, no 7_estimated_building_damage, etc.).
- Make sure the posting status strictly follows ATC-20 logic based on the severity of the six conditions.
"""

        # 使用Qwen进行Stage 3推理
        system_prompt = "You are an expert structural engineer performing ATC-20 post-earthquake safety evaluation based on damage descriptions, similar reference cases, and ATC-20 rules."
        user_prompt = prompt_content
        
        response = self.qwen_engine.generate_json([], system_prompt, user_prompt)

        # Parse JSON
        response = response.strip()
        if response.startswith("```json"):
            response = response[len("```json"):]
        elif response.startswith("```"):
            response = response[len("```"):]

        if response.endswith("```"):
            response = response[:-len("```")]

        response = response.strip()

        enhanced_eval = None
        enhanced_final = None

        try:
            final_json = json.loads(response)
            print("\n✓ Stage 3 Final Output (DC/SP only):")
            print(json.dumps(final_json, indent=2, ensure_ascii=False))
            enhanced_eval = final_json.get("6_evaluation")
            enhanced_final = final_json.get("8_final_status")
        except Exception as e:
            print(f"\n✗ Stage 3 JSON Parse Error: {e}")
            print(f"Response after cleanup (first 300 chars): {response[:300]}")

        # 组合最终输出：以 Qwen 原始预测为基础
        # - 6_evaluation: 只覆盖各 condition 的 status，description 完全保留 Stage1 结果
        # - 8_final_status: 完全采用 Stage3 结果（status + reasoning）
        combined = dict(stage1_full_json) if isinstance(stage1_full_json, dict) else {}

        if isinstance(enhanced_eval, dict):
            base_eval = combined.get("6_evaluation", {}) or {}
            for key, cond_obj in enhanced_eval.items():
                if not isinstance(cond_obj, dict):
                    continue
                new_status = cond_obj.get("status")
                if not new_status:
                    continue
                base_cond = base_eval.get(key, {}) or {}
                base_cond["status"] = new_status
                base_eval[key] = base_cond
            combined["6_evaluation"] = base_eval

        if isinstance(enhanced_final, dict):
            combined["8_final_status"] = enhanced_final

        print("\n✓ Combined Final Output (Qwen Stage1 + RAG-enhanced DC/SP):")
        print(json.dumps(combined, indent=2, ensure_ascii=False))

        return combined

    async def process_case(self, case_item, infer_item, index: int):
        """处理单个样本。

        完全复用 utils_qwen_finetune.get_inference_dataset 构造的 system_prompt / user_prompt / images，
        这样 Stage1 的行为与原始 inference_qwen_7b_finetuned.py 一致。
        """
        monitor = PerformanceMonitor(index)
        monitor.start()

        # 使用 inference_dataset 中已经归一化过的本地图片路径
        local_images = infer_item.get("images", [])
        local_images = [p for p in local_images if os.path.exists(p)]

        if not local_images:
            print(f"No images found for index {index}")
            return None

        system_prompt = infer_item.get("system_prompt", "")
        user_prompt = infer_item.get("user_prompt", "")

        # Stage 1（与原始脚本一致：full JSON -> normalize），再额外提取 Stage1 精简字段给 RAG 使用
        stage1_ret = await self.run_stage1(local_images, system_prompt, user_prompt)
        if not stage1_ret:
            return None

        stage1_full, stage1_res, qwen_tokens = stage1_ret
        monitor.set_token_count(qwen_tokens)
            
        # Stage 2（只基于 Stage1 精简字段做检索与规则查询）
        test_case_id = case_item.get("test_case_info", {}).get("case_id", "Unknown")
        # 检索top2 cases
        similar_cases = retrieve_hybrid_structured(
            {
                "case_id": test_case_id,
                "2_view_type": stage1_res.get("IS", "Unknown"),
                "3_primary_occupation": {"category": stage1_res.get("PO", "Unknown")},
                "4_type_of_construction": {"category": stage1_res.get("CT", "Unknown")},
                "5_stories": stage1_res.get("5_stories", {"over_ground": None, "under_ground": None}),
                "9_image_descriptions": {f"image{i+1}": desc for i, desc in enumerate(stage1_res.get("SID", []))}
            },
            self.similar_case_kb, self.similar_case_model, top_k=3
        )
        atc_rules = await self.run_stage2b(stage1_res)

        # 保存检索日志 - v4+ATC专用目录
        retrieval_dir = os.path.join(os.path.dirname(__file__), "retrieval_logs", "v5_atc")
        os.makedirs(retrieval_dir, exist_ok=True)
        retrieval_log = {
            "case_id": test_case_id,
            "topk_similar_cases": [
                {
                    "id": c["id"],
                    "score": c.get("score"),
                    "meta": {
                        "3_primary_occupation": c["meta"]["original_data"].get("3_primary_occupation", {}),
                        "4_type_of_construction": c["meta"]["original_data"].get("4_type_of_construction", {}),
                        "5_stories": c["meta"]["original_data"].get("5_stories", {}),
                        "8_final_status": c["meta"]["original_data"].get("8_final_status", {})
                    }
                } for c in (similar_cases or [])
            ],
            "atc_rules": atc_rules
        }
        with open(os.path.join(retrieval_dir, f"case_{test_case_id}_retrieve.json"), "w", encoding="utf-8") as f:
            json.dump(retrieval_log, f, indent=2, ensure_ascii=False)

        # Stage 3：在 Qwen 的完整预测基础上，用 RAG + Qwen 优化损伤评级和安全标签，使用top1和top2 cases
        
        # 创建只包含description的6_evaluation
        stage1_for_stage3 = dict(stage1_full)
        if "6_evaluation" in stage1_for_stage3:
            eval_no_status = {}
            for key, cond in stage1_for_stage3["6_evaluation"].items():
                if isinstance(cond, dict):
                    eval_no_status[key] = {"description": cond.get("description", "")}
                else:
                    eval_no_status[key] = cond
            stage1_for_stage3["6_evaluation"] = eval_no_status
        
        final_res = await self.run_stage3(stage1_full, stage1_for_stage3, similar_cases, atc_rules)

        monitor.end()
        metrics = monitor.get_metrics()

        # Return in format expected by evaluate.py
        result = {
            "prediction": final_res,
            # 使用 inference_dataset 里的本地路径，让 evaluate.py 能正确解析 earthquake_folder / case_id
            "images": infer_item.get("images", []),
            "test_case_info": case_item.get("test_case_info", {}),
            "index": index,
            "_timing": metrics  # 新的性能指标结构
        }
        return result

async def main(data_path, output_path, limit=None):
    pipeline_start = time.time()
    
    pipeline = Pipeline()
    await pipeline.initialize()  # Changed to await since initialize is now async
    
    # 使用新的 full_test.json 作为测试集（与 LVLM-inference-2 保持一致）
    test_json_path = data_path

    # 完全复用 utils_qwen_finetune.get_inference_dataset 生成的 system_prompt / user_prompt / images
    from utils_qwen_finetune import get_inference_dataset
    inference_data = get_inference_dataset(test_json_path)
    
    # 结果输出路径（会在运行过程中不断覆盖写入，保存当前已完成的样本）
    # Use a combined Similar+ATC output name to avoid collisions with other pipelines
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if limit is not None:
        inference_data = inference_data[:limit]

    # 跑全部 test case，index 与 inference_data 对齐
    total_to_run = len(inference_data)
    results = []

    print("\n" + "#"*80)
    print("#" + " "*20 + f"RUNNING {total_to_run} TEST CASES" + " "*20 + "#")
    print("#"*80 + "\n")

    for idx in tqdm(range(total_to_run), desc="Pipeline Cases", ncols=100):
        infer_item = inference_data[idx]

        images = infer_item.get('images', []) or []
        if images:
            folder, case_id = extract_case_info_from_image(images[0])
        else:
            folder, case_id = "Unknown", "Unknown"

        case_info = {"earthquake_folder": folder, "case_id": case_id}

        print(f"\n{'='*80}")
        print(f"Processing Case [{idx}]: {folder} / {case_id}")
        print(f"Images: {len(images)} files")
        print(f"{'='*80}")

        # 构造一个最小的 case_item，只携带 test_case_info，供后续保存结果使用
        case_item = {"test_case_info": case_info}

        res = await pipeline.process_case(case_item, infer_item, idx)
        if res:
            results.append(res)

            # 每完成一个样本就增量保存一次，避免中途停止时完全没有结果
            partial_summary = {
                "total_samples": len(results),
                "successful": len(results),
                "failed": (idx + 1) - len(results),
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"results": results, "summary": partial_summary}, f, indent=2, ensure_ascii=False)

    # 构建 summary，供 evaluator 打印
    summary = {
        "total_samples": len(results),
        "successful": len(results),
        "failed": total_to_run - len(results),
    }

    # 计算性能指标
    pipeline_total_time = time.time() - pipeline_start
    case_timings = [r.get("_timing", {}).get("elapsed_seconds", 0) for r in results if "_timing" in r]
    token_counts = [r.get("_timing", {}).get("token_count", 0) for r in results if "_timing" in r]
    gpu_utils = [r.get("_timing", {}).get("gpu_util_percent_avg", 0) for r in results if "_timing" in r and r.get("_timing", {}).get("gpu_util_percent_avg")]
    
    if case_timings:
        import statistics
        total_tokens = sum(token_counts)
        avg_case_time = statistics.mean(case_timings)
        cases_per_second = len(results) / pipeline_total_time if pipeline_total_time > 0 else 0
        tokens_per_second = total_tokens / pipeline_total_time if pipeline_total_time > 0 else 0
        avg_gpu_util = statistics.mean(gpu_utils) if gpu_utils else 0
        
        performance_summary = {
            "pipeline_name": "SeisMLLM",
            "total_time_seconds": pipeline_total_time,
            "total_cases": len(results),
            "cases_per_second": cases_per_second,
            "avg_case_latency_seconds": avg_case_time,
            "min_case_latency_seconds": min(case_timings),
            "max_case_latency_seconds": max(case_timings),
            "total_tokens": total_tokens,
            "tokens_per_second": tokens_per_second,
            "avg_tokens_per_case": statistics.mean(token_counts) if token_counts else 0,
            "avg_gpu_utilization_percent": avg_gpu_util,
        }
        summary["performance"] = performance_summary

    # Save final Results（覆盖最后一次增量保存，summary 按总数重写）
    with open(output_path, 'w', encoding='utf-8') as f:
        # Save in the format expected by evaluate.py: {"results": [...], "summary": {...}}
        json.dump({"results": results, "summary": summary}, f, indent=2, ensure_ascii=False)

    # 额外保存性能指标到单独的 JSON 文件
    if "performance" in summary:
        perf_path = output_path.replace(".json", "_performance.json")
        with open(perf_path, 'w', encoding='utf-8') as f:
            json.dump(summary["performance"], f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Conversation-format input JSON")
    parser.add_argument("--output", default="results/predictions.json")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    asyncio.run(main(args.data, args.output, args.limit))
