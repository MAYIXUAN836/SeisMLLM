"""
Create enhanced knowledge base with multimodal processing.

Supports PDF processing with complete multimodal document understanding:
- Text, image, table, and equation processing
- Intelligent context extraction  
- Vision model support
- Batch processing capabilities

Usage:
    python 1_create_knowledge_base_enhanced.py
"""

import asyncio
import os
import sys
from pathlib import Path
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
import openai

# 确保可以从本地 RAG-Anything 子模块导入 raganything.config


from config import (
    API_KEY, BASE_URL, LLM_MODEL, EMBEDDING_MODEL, EMBEDDING_DIM, PDF_FILE,
    ENABLE_IMAGE_PROCESSING, ENABLE_TABLE_PROCESSING, ENABLE_EQUATION_PROCESSING,
    PARSER, PARSE_METHOD, DISPLAY_CONTENT_STATS, CONTEXT_WINDOW, CONTEXT_MODE,
    MAX_CONTEXT_TOKENS, INCLUDE_HEADERS, INCLUDE_CAPTIONS, MAX_CONCURRENT_FILES, KNOWLEDGE_BASE_DIR
)

async def gptsapi_llm_multimodal(prompt, system_prompt=None, **kwargs):
    """Multimodal LLM function supporting both text and images."""
    from openai import RateLimitError
    
    client = openai.AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    history_messages = kwargs.pop('history_messages', [])
    messages = []
    if history_messages:
        messages.extend(history_messages)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
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
        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 60
                print(f"Rate limited. Retrying in {wait_time}s... ({attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                raise

async def gptsapi_vision_model(prompt, image_data=None, **kwargs):
    """Vision model function for image analysis."""
    if image_data:
        content = [
            {"type": "text", "text": prompt}
        ]
        
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
        
        return await gptsapi_llm_multimodal({"content": content}, **kwargs)
    else:
        return await gptsapi_llm_multimodal(prompt, **kwargs)

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

def get_safe_dirname(pdf_path: str) -> str:
    """Generate safe directory name from PDF filename."""
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in basename)
    safe_name = safe_name.replace(' ', '_').lower()
    safe_name = "_".join(part for part in safe_name.split('_') if part)
    return safe_name or "default"

async def create_enhanced_knowledge_base(pdf_path: str):
    """Create enhanced knowledge base with multimodal processing."""
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        return False
    
    safe_dirname = get_safe_dirname(pdf_path)
    knowledge_base_dir = KNOWLEDGE_BASE_DIR
    output_dir = f"./output_{safe_dirname}"
    
    print("="*70)
    print("Creating knowledge base with multimodal processing")
    print("="*70)
    print(f"Input file: {pdf_path}")
    print(f"Knowledge base directory: {knowledge_base_dir}")
    print(f"Parse output directory: {output_dir}")
    print("="*70)
    
    config = RAGAnythingConfig(
        working_dir=knowledge_base_dir,
        parser_output_dir=output_dir,
        parser=PARSER,
        parse_method=PARSE_METHOD,
        display_content_stats=DISPLAY_CONTENT_STATS,
        enable_image_processing=ENABLE_IMAGE_PROCESSING,
        enable_table_processing=ENABLE_TABLE_PROCESSING, 
        enable_equation_processing=ENABLE_EQUATION_PROCESSING,
        context_window=CONTEXT_WINDOW,
        context_mode=CONTEXT_MODE,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        include_headers=INCLUDE_HEADERS,
        include_captions=INCLUDE_CAPTIONS,
        max_concurrent_files=MAX_CONCURRENT_FILES
    )
    
    rag = RAGAnything(
        config=config,
        llm_model_func=gptsapi_llm_multimodal,
        vision_model_func=gptsapi_vision_model,
        embedding_func=embedding_func,
        lightrag_kwargs={
            'chunk_token_size': 1200,
            'chunk_overlap_token_size': 100,
            'max_entity_tokens': 512,
            'max_relation_tokens': 512,
            'top_k': 10,
            'chunk_top_k': 5,
        }
    )
    
    try:
        print("\nProcessing document...")
        print("-"*70)
        
        await rag.process_document_complete(
            file_path=pdf_path,
            output_dir=output_dir,
            parse_method=PARSE_METHOD,
            show_progress=True
        )
        
        print("\n" + "="*70)
        print("Knowledge base created successfully")
        print("="*70)
        print(f"Knowledge base location: {knowledge_base_dir}/")
        print(f"Parse results: {output_dir}/")
        
        if DISPLAY_CONTENT_STATS:
            print(f"\nProcessing statistics:")
            try:
                import json
                vdb_chunks = f"{knowledge_base_dir}/vdb_chunks.json"
                vdb_entities = f"{knowledge_base_dir}/vdb_entities.json" 
                vdb_relationships = f"{knowledge_base_dir}/vdb_relationships.json"
                
                if os.path.exists(vdb_chunks):
                    with open(vdb_chunks, 'r', encoding='utf-8') as f:
                        chunks_data = json.load(f)
                        print(f"   Text chunks: {len(chunks_data)}")
                
                if os.path.exists(vdb_entities):
                    with open(vdb_entities, 'r', encoding='utf-8') as f:
                        entities_data = json.load(f)
                        print(f"   Entities: {len(entities_data)}")
                
                if os.path.exists(vdb_relationships):
                    with open(vdb_relationships, 'r', encoding='utf-8') as f:
                        relations_data = json.load(f)
                        print(f"   Relations: {len(relations_data)}")
                        
            except Exception as e:
                print(f"   Statistics read failed: {e}")
        
        print(f"\nNext steps:")
        print(f"   1. Update KNOWLEDGE_BASE_DIR in config.py")
        print(f"      Set to: '{knowledge_base_dir}'")
        print(f"   2. Run query script to test knowledge base")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\nFailed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def batch_process_documents(pdf_files: list):
    """Process multiple PDF documents in batch."""
    print(f"\nBatch processing {len(pdf_files)} documents...")
    
    success_count = 0
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n{'='*70}")
        print(f"Processing document {i}/{len(pdf_files)}: {pdf_file}")
        print(f"{'='*70}")
        
        if await create_enhanced_knowledge_base(pdf_file):
            success_count += 1
        
        if i < len(pdf_files):
            print("\nWaiting 5s before next document...")
            await asyncio.sleep(5)
    
    print(f"\n{'='*70}")
    print(f"Batch processing complete: {success_count}/{len(pdf_files)} documents processed")
    print(f"{'='*70}")

async def main():
    """Main function - configure PDF file in config.py"""
    
    print("="*70)
    print("RAG-Anything Knowledge Base Creator")
    print("="*70)
    print(f"PDF file: {PDF_FILE}")
    print("="*70)
    
    await create_enhanced_knowledge_base(PDF_FILE)

if __name__ == "__main__":
    asyncio.run(main())