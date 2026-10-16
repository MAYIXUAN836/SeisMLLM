"""RAG-ATC / RAG-Anything configuration (open-source friendly).

This file intentionally does NOT contain any real API keys.
All secrets should be provided via environment variables.

Environment variables (recommended):

- RAG_ATC_API_KEY        : LLM API key (if needed)
- RAG_ATC_BASE_URL       : LLM base URL (e.g. http://localhost:11434/v1)
- RAG_ATC_LLM_MODEL      : Chat model name (e.g. qwen2.5:32b)
- RAG_ATC_EMBED_MODEL    : Embedding model name (e.g. bge-m3:latest)
- RAG_ATC_EMBED_DIM      : Embedding dimension (default: 1024)
- RAG_ATC_PDF_FILE       : Path to the source PDF for knowledge-base construction
- RAG_ATC_KB_DIR         : Knowledge base working directory
- RAG_ATC_QUERY_MODE     : naive | local | global | hybrid
"""

from __future__ import annotations

import os
from pathlib import Path


# ==================== 路径与基础配置 ====================

ROOT_DIR = Path(__file__).resolve().parent

# 知识库存储目录与查询模式
KNOWLEDGE_BASE_DIR = os.getenv(
    "RAG_ATC_KB_DIR",
    str(ROOT_DIR / "rag_storage_codebook"),
)

QUERY_MODE = os.getenv("RAG_ATC_QUERY_MODE", "hybrid")  # naive/local/global/hybrid


# ==================== LLM 配置 ====================

LLM_MODEL = os.getenv("RAG_ATC_LLM_MODEL", "qwen2.5:32b")
BASE_URL = os.getenv("RAG_ATC_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("RAG_ATC_API_KEY", "")  # Fill in your API key via RAG_ATC_API_KEY.


# ==================== Embedding 配置 ====================

EMBEDDING_MODEL = os.getenv("RAG_ATC_EMBED_MODEL", "bge-m3:latest")
EMBEDDING_DIM = int(os.getenv("RAG_ATC_EMBED_DIM", "1024"))
EMBEDDING_MAX_TOKEN_SIZE = int(os.getenv("RAG_ATC_EMBED_MAX_TOKENS", "8192"))


# ==================== 源 PDF 路径（用于构建知识库） ====================

PDF_FILE = os.getenv("RAG_ATC_PDF_FILE", "")  # Set your absolute path to 20-1-OCR.pdf.


# ==================== 解析与多模态开关 ====================

PARSER = os.getenv("RAG_ATC_PARSER", "mineru")
PARSE_METHOD = os.getenv("RAG_ATC_PARSE_METHOD", "auto")

DISPLAY_CONTENT_STATS = os.getenv("RAG_ATC_DISPLAY_STATS", "1") == "1"

ENABLE_IMAGE_PROCESSING = os.getenv("RAG_ATC_ENABLE_IMAGE", "1") == "1"
ENABLE_TABLE_PROCESSING = os.getenv("RAG_ATC_ENABLE_TABLE", "1") == "1"
ENABLE_EQUATION_PROCESSING = os.getenv("RAG_ATC_ENABLE_EQUATION", "1") == "1"

# 这些参数对应 LightRAG/RAGAnything 的上下文窗口设置
CONTEXT_WINDOW = int(os.getenv("RAG_ATC_CONTEXT_WINDOW", "4096"))
CONTEXT_MODE = os.getenv("RAG_ATC_CONTEXT_MODE", "sliding")
MAX_CONTEXT_TOKENS = int(os.getenv("RAG_ATC_MAX_CONTEXT_TOKENS", "2048"))

INCLUDE_HEADERS = os.getenv("RAG_ATC_INCLUDE_HEADERS", "1") == "1"
INCLUDE_CAPTIONS = os.getenv("RAG_ATC_INCLUDE_CAPTIONS", "1") == "1"
MAX_CONCURRENT_FILES = int(os.getenv("RAG_ATC_MAX_CONCURRENT", "1"))


__all__ = [
    "ROOT_DIR",
    "KNOWLEDGE_BASE_DIR",
    "QUERY_MODE",
    "LLM_MODEL",
    "BASE_URL",
    "API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "EMBEDDING_MAX_TOKEN_SIZE",
    "PDF_FILE",
    "PARSER",
    "PARSE_METHOD",
    "DISPLAY_CONTENT_STATS",
    "ENABLE_IMAGE_PROCESSING",
    "ENABLE_TABLE_PROCESSING",
    "ENABLE_EQUATION_PROCESSING",
    "CONTEXT_WINDOW",
    "CONTEXT_MODE",
    "MAX_CONTEXT_TOKENS",
    "INCLUDE_HEADERS",
    "INCLUDE_CAPTIONS",
    "MAX_CONCURRENT_FILES",
]
