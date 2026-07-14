# document_processor.py
"""
文档解析与切片工具模块。

支持 PDF、DOCX、TXT、Markdown 格式文件的文本提取与智能切片，
为知识库向量化提供预处理能力。切片功能基于 LangChain 的 RecursiveCharacterTextSplitter。
"""
import os
import re
from typing import List, Dict, Optional
from utils import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 可选依赖：仅在需要时导入
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
    logger.warning("PyPDF2 未安装，PDF 文件解析功能不可用")

try:
    import docx
except ImportError:
    docx = None
    logger.warning("python-docx 未安装，DOCX 文件解析功能不可用")


# ===================== 文本提取 =====================
def extract_text(file_path: str) -> str:
    """根据文件扩展名自动提取文本内容"""
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return ""

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == '.pdf':
            return _extract_from_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            return _extract_from_docx(file_path)
        elif ext in ['.txt', '.md', '.markdown']:
            return _extract_from_text(file_path)
        else:
            logger.error(f"不支持的文件格式: {ext}")
            return ""
    except Exception as e:
        logger.error(f"提取文本失败 [{file_path}]: {e}", exc_info=True)
        return ""


def _extract_from_pdf(file_path: str) -> str:
    if PyPDF2 is None:
        raise ImportError("请安装 PyPDF2: pip install PyPDF2")
    text_parts = []
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def _extract_from_docx(file_path: str) -> str:
    if docx is None:
        raise ImportError("请安装 python-docx: pip install python-docx")
    doc = docx.Document(file_path)
    text_parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)
    return "\n".join(text_parts)


def _extract_from_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


# ===================== 文本清洗 =====================
def clean_text(text: str) -> str:
    """清洗冗余内容：去除页码、多余空行等"""
    if not text:
        return ""
    # 去除单独的页码行
    text = re.sub(r'^\d{1,4}$', '', text, flags=re.MULTILINE)
    # 合并多个空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ===================== 切片处理（基于 LangChain） =====================
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[str]:
    """
    使用 LangChain 的递归字符分割器将长文本切分为语义片段。

    Args:
        text:          待切分的文本
        chunk_size:    切片大小（字符数）
        chunk_overlap: 相邻切片重叠字符数

    Returns:
        文本片段列表
    """
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", "！", "？", "；", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    logger.info(f"文本切分完成：{len(chunks)} 个片段")
    return chunks

# ===================== 一键处理（提取+清洗+切片） =====================
def process_document(
    file_path: str,
    metadata_template: Optional[Dict] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> tuple:
    """完整文档处理流水线：提取 → 清洗 → 切片 → 附带元数据"""
    raw_text = extract_text(file_path)
    if not raw_text:
        logger.error("文本提取为空，处理终止")
        return [], []

    cleaned = clean_text(raw_text)
    chunks = split_text(cleaned, chunk_size, chunk_overlap)
    if not chunks:
        return [], []

    source_name = os.path.basename(file_path)
    base_meta = metadata_template.copy() if metadata_template else {}
    metadatas = []
    for i, chunk in enumerate(chunks):
        meta = base_meta.copy()
        meta["source"] = source_name
        meta["chunk_index"] = i
        metadatas.append(meta)

    return chunks, metadatas