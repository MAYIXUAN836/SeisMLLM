"""
Query knowledge base with multimodal support.

Supports text and image queries with intelligent document retrieval.
Features:
- Multimodal query support (text and images)
- Context-aware response generation  
- Detailed source citation
- Multiple query modes (naive/local/global/hybrid)
- Formatted result display

Usage:
    python 2_query_knowledge_base_enhanced.py
"""

import asyncio
import os
import sys
import json
from pathlib import Path



from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
import openai

from config import (
    API_KEY, BASE_URL, LLM_MODEL, EMBEDDING_MODEL, EMBEDDING_DIM, 
    KNOWLEDGE_BASE_DIR, QUERY_MODE, ENABLE_IMAGE_PROCESSING, 
    ENABLE_TABLE_PROCESSING, ENABLE_EQUATION_PROCESSING
)

async def gptsapi_llm_enhanced(prompt, system_prompt=None, **kwargs):
    """Enhanced LLM function supporting multimodal queries."""
    from openai import RateLimitError
    
    client = openai.AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    history_messages = kwargs.pop('history_messages', [])
    messages = []
    
    if history_messages:
        messages.extend(history_messages)
    
    enhanced_system_prompt = """You are a professional document analysis assistant.
When answering questions, provide:
1. Direct answer to the user's question
2. Complete quotes from source material marked with Source Quote
3. Specific source location marked with Source

Provide detailed and accurate information without omitting key details."""

    if system_prompt:
        enhanced_system_prompt = f"{enhanced_system_prompt}\n\n{system_prompt}"
    
    messages.append({"role": "system", "content": enhanced_system_prompt})
    
    if isinstance(prompt, dict) and 'content' in prompt:
        messages.append({"role": "user", "content": prompt['content']})
    else:
        messages.append({"role": "user", "content": prompt})
    
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 
                              'presence_penalty', 'stop', 'n', 'stream']}
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                **filtered_kwargs
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 60
                print(f" API{wait_time}... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                raise

async def gptsapi_vision_model(prompt, image_data=None, **kwargs):
    """Vision model function for image analysis."""
    if image_data:
        content = [{"type": "text", "text": prompt}]
        
        if isinstance(image_data, str) and os.path.exists(image_data):
            import base64
            with open(image_data, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
            })
        elif isinstance(image_data, str) and image_data.startswith('data:'):
            content.append({
                "type": "image_url", 
                "image_url": {"url": image_data}
            })
        
        return await gptsapi_llm_enhanced({"content": content}, **kwargs)
    else:
        return await gptsapi_llm_enhanced(prompt, **kwargs)

async def gptsapi_embed(texts):
    """GPTsAPI embedding function with retry logic."""
    from openai import RateLimitError
    
    client = openai.AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    if isinstance(texts, str):
        texts = [texts]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts
            )
            return [item.embedding for item in response.data]
        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 60
                print(f"Rate limited. Retrying in {wait_time}s... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                raise

embedding_func = EmbeddingFunc(
    embedding_dim=EMBEDDING_DIM,
    max_token_size=8192,
    func=gptsapi_embed
)

def load_chunks_data(knowledge_base_dir: str):
    """Load text chunks data."""
    chunks_file = os.path.join(knowledge_base_dir, "kv_store_text_chunks.json")
    if os.path.exists(chunks_file):
        with open(chunks_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def display_query_statistics(knowledge_base_dir: str):
    """Knowledge base"""
    print(f"\n Knowledge base ({knowledge_base_dir}):")
    print("-" * 50)
    
    try:
        stats = {}
        
        vdb_chunks = os.path.join(knowledge_base_dir, "vdb_chunks.json")
        if os.path.exists(vdb_chunks):
            with open(vdb_chunks, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
                if isinstance(chunks_data, dict) and 'data' in chunks_data:
                    stats[''] = len(chunks_data['data'])
                else:
                    stats[''] = len(chunks_data) if isinstance(chunks_data, list) else 1
        
        vdb_entities = os.path.join(knowledge_base_dir, "vdb_entities.json")
        if os.path.exists(vdb_entities):
            with open(vdb_entities, 'r', encoding='utf-8') as f:
                entities_data = json.load(f)
                if isinstance(entities_data, dict) and 'data' in entities_data:
                    stats[''] = len(entities_data['data'])
                else:
                    stats[''] = len(entities_data) if isinstance(entities_data, list) else 1
        
        vdb_relationships = os.path.join(knowledge_base_dir, "vdb_relationships.json")
        if os.path.exists(vdb_relationships):
            with open(vdb_relationships, 'r', encoding='utf-8') as f:
                relations_data = json.load(f)
                if isinstance(relations_data, dict) and 'data' in relations_data:
                    stats[''] = len(relations_data['data'])
                else:
                    stats[''] = len(relations_data) if isinstance(relations_data, list) else 1
        
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print(f"   Query mode: {QUERY_MODE}")
        print(f"   : {'' if ENABLE_IMAGE_PROCESSING else ''} "
              f"{'' if ENABLE_TABLE_PROCESSING else ''} "
              f"{'' if ENABLE_EQUATION_PROCESSING else ''}")
        
    except Exception as e:
        print(f"   : {e}")

async def enhanced_query_knowledge_base(query: str, image_path: str = None, show_context: bool = False):
    """Query knowledge base with multimodal support."""
    
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"Error: Knowledge base not found: {KNOWLEDGE_BASE_DIR}")
        print("Run 1_create_knowledge_base_enhanced.py to create knowledge base first")
        return
    
    print("="*70)
    print("Querying knowledge base")
    print("="*70)
    print(f"Query: {query}")
    if image_path:
        print(f"Image: {image_path}")
    print(f"Knowledge base: {KNOWLEDGE_BASE_DIR}")
    print(f"Query mode: {QUERY_MODE}")
    print("="*70)
    
    display_query_statistics(KNOWLEDGE_BASE_DIR)
    

    config = RAGAnythingConfig(
        working_dir=KNOWLEDGE_BASE_DIR,
        enable_image_processing=ENABLE_IMAGE_PROCESSING,
        enable_table_processing=ENABLE_TABLE_PROCESSING,
        enable_equation_processing=ENABLE_EQUATION_PROCESSING,
    )
    

    context_info = {"chunks": [], "entities": [], "relations": []}
    
    async def llm_with_context_capture(prompt, **kwargs):
        """LLM"""
        if "# Entities" in prompt:
            try:
                entities_section = prompt.split("# Entities")[1].split("#")[0]
                context_info["entities"] = entities_section.strip()
            except:
                pass
        
        if "# Relationships" in prompt:
            try:
                relations_section = prompt.split("# Relationships")[1].split("#")[0]
                context_info["relations"] = relations_section.strip()
            except:
                pass
        
        if "# Sources" in prompt or "# Context" in prompt:
            try:
                if "# Sources" in prompt:
                    chunks_section = prompt.split("# Sources")[1].split("---")[0]
                elif "# Context" in prompt:
                    chunks_section = prompt.split("# Context")[1].split("---")[0]
                context_info["chunks"] = chunks_section.strip()
            except:
                pass
        
        return await gptsapi_llm_enhanced(prompt, **kwargs)
    
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_with_context_capture,
        vision_model_func=gptsapi_vision_model,
        embedding_func=embedding_func
    )
    
    await rag._ensure_lightrag_initialized()
    
    try:
        print(f"\n ...")
        
        if image_path and os.path.exists(image_path):
            print(f" : {image_path}")
            query_input = f"{query}\n[]"
        else:
            query_input = query
        
        result = await rag.aquery(query_input, mode=QUERY_MODE)
        
        print("\n" + "="*70)
        print(" ")
        print("="*70)
        

        if result:
            lines = [line.strip() for line in result.strip().split('\n') if line.strip()]
            
            answer_lines = []
            citation_lines = []
            source_lines = []
            current_section = None
            
            for line in lines:
                if '' in line or '' in line:
                    current_section = "citation"
                    content = line.replace('', '').replace('', '').strip()
                    if content:
                        citation_lines.append(content)
                elif '' in line:
                    current_section = "source"
                    source_lines.append(line)
                else:
                    if current_section == "citation":
                        if line not in citation_lines:
                            citation_lines.append(line)
                    elif current_section == "source":
                        if line not in source_lines:
                            source_lines.append(line)
                    else:
                        if line not in answer_lines:
                            answer_lines.append(line)
            
            # Answer:
            if answer_lines:
                print(f" :")
                print("-" * 50)
                for line in answer_lines:
                    print(f"   {line}")
            
            # Source quotes:
            if citation_lines:
                print(f"\n :")
                print("-" * 50)
                print(f"   ")
                for line in citation_lines:
                    print(f"   {line}")
            
            # Sources:
            if source_lines:
                print(f"\n :")
                print("-" * 50)
                for line in source_lines:
                    print(f"   {line}")
        
        else:
            print(" ")
            print(" :")
            print("   - ")
            print("   - Knowledge base")
            print(f"   - Query mode (: {QUERY_MODE})")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n : {e}")
        import traceback
        traceback.print_exc()
    
    finally:

        try:
            await rag.close_storages()
        except Exception:
            pass

async def interactive_query_session():
    """Interactive query session"""
    print(" Query mode")
    print("  'quit'  'exit' ")
    print("  'image:path/to/image.jpg:' ")
    print("-" * 70)
    
    while True:
        try:
            user_input = input("\n Query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print(" ")
                break
            
            if not user_input:
                continue
            
            image_path = None
            if user_input.startswith('image:'):
                parts = user_input.split(':', 2)
                if len(parts) >= 3:
                    image_path = parts[1]
                    query = parts[2]
                    if not os.path.exists(image_path):
                        print(f" : {image_path}")
                        continue
                else:
                    print(" : image::")
                    continue
            else:
                query = user_input
            
            await enhanced_query_knowledge_base(query, image_path)
            
        except KeyboardInterrupt:
            print("\n\n ")
            break
        except Exception as e:
            print(f" : {e}")

async def main():
    """"""
    print("="*70)
    print(" RAG-Anything Knowledge base")
    print("="*70)
    
    sample_queries = [
        "",
        "",
        ""
    ]
    
    print(" :")
    for i, query in enumerate(sample_queries, 1):
        print(f"   {i}. {query}")
    
    print(f"\n Knowledge base: {KNOWLEDGE_BASE_DIR}")
    
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"\n Knowledge base: {KNOWLEDGE_BASE_DIR}")
        print("  1_create_knowledge_base_enhanced.py Knowledge base")
        return
    
    print("\n:")
    print("1. ")
    print("2. ")
    
    try:
        choice = input(" (1/2): ").strip()
        
        if choice == "1":
            query = input("Query: ").strip()
            if query:
                await enhanced_query_knowledge_base(query)
        elif choice == "2":
            await interactive_query_session()
        else:
            print(" ")
    
    except KeyboardInterrupt:
        print("\n\n ")

if __name__ == "__main__":
    asyncio.run(main())