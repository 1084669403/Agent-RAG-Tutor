# vector_store.py
"""
向量数据库管理模块。

基于 Chroma 实现持久化向量存储，支持按标签过滤检索，
与知识库上传、RAG 问答联动。所有向量化通过阿里云 Embedding API 完成，
不依赖 Chroma 内置嵌入模型。
"""
import uuid
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from config import Config
from utils import logger
from embedding_service import get_embeddings


# ===================== Chroma 客户端初始化 =====================
def _get_client() -> chromadb.PersistentClient:
    """获取或创建持久化 Chroma 客户端"""
    return chromadb.PersistentClient(
        path=Config.CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )


# ===================== Collection 管理 =====================
COLLECTION_NAME = "knowledge_base"


def _get_collection() -> chromadb.Collection:
    """获取或创建知识库 collection，不绑定任何嵌入函数"""
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


# ===================== 对外接口 =====================
def add_documents(
    texts: List[str],
    metadatas: List[Dict],
    ids: Optional[List[str]] = None
) -> None:
    """
    批量添加文档到向量库。
    向量通过阿里云 Embedding API 生成，不依赖本地模型。
    """
    if not texts:
        logger.warning("add_documents: 文本列表为空")
        return

    if len(texts) != len(metadatas):
        raise ValueError("texts 与 metadatas 长度不一致")

    if ids is None:
        ids = [str(uuid.uuid4()) for _ in texts]

    logger.info(f"正在为 {len(texts)} 条文档生成向量...")
    embeddings = get_embeddings(texts)
    if not embeddings or len(embeddings) != len(texts):
        logger.error("嵌入生成失败或返回数量不一致，放弃添加")
        return

    collection = _get_collection()
    try:
        collection.add(
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )
        logger.info(f"成功添加 {len(texts)} 条文档到向量库")
    except Exception as e:
        logger.error(f"添加文档失败: {e}", exc_info=True)
        raise


def search(
    query_text: str,
    filter_tags: Optional[Dict[str, str]] = None,
    top_k: int = 5
) -> List[Dict]:
    """
    根据查询文本和标签过滤，检索最相关的文档片段。
    查询向量通过阿里云 Embedding API 生成。

    Args:
        query_text:  查询文本
        filter_tags: 标签过滤条件，例如 {"category": "science", "sub_field": "math"}
        top_k:       返回数量

    Returns:
        结果列表，每项包含 id, text, metadata, distance
    """
    if not query_text.strip():
        return []

    # 1. 生成查询向量
    query_vecs = get_embeddings([query_text])
    if not query_vecs or not query_vecs[0]:
        logger.error("查询嵌入生成失败")
        return []
    query_vector = query_vecs[0]

    # 2. 构建过滤条件（符合 Chroma where 语法）
    where_filter = None
    if filter_tags:
        conditions = []
        for key, value in filter_tags.items():
            conditions.append({key: {"$eq": value}})
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

    # 3. 用向量查询
    collection = _get_collection()
    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter
        )
    except Exception as e:
        logger.error(f"向量检索失败: {e}", exc_info=True)
        return []

    # 4. 整理输出
    output = []
    if results.get("ids") and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            item = {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i] if results.get("documents") else "",
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                "distance": results["distances"][0][i] if results.get("distances") else None
            }
            output.append(item)
    return output


def delete_by_ids(ids: List[str]) -> None:
    """根据 ID 列表删除文档"""
    if not ids:
        return
    collection = _get_collection()
    try:
        collection.delete(ids=ids)
        logger.info(f"成功删除 {len(ids)} 条文档")
    except Exception as e:
        logger.error(f"删除文档失败: {e}", exc_info=True)


def delete_by_filter(filter_dict: Dict[str, str]) -> int:
    """
    根据标签条件删除向量库中的文档切片。
    支持多条件组合（内部自动转换为 Chroma 所需的 $and 语法）。

    Args:
        filter_dict: 过滤条件字典，例如 {"category": "science", "knowledge_name": "高等数学"}

    Returns:
        实际删除的切片数量
    """
    if not filter_dict:
        return 0

    # 构建符合 Chroma 要求的 where 表达式
    conditions = []
    for key, value in filter_dict.items():
        conditions.append({key: {"$eq": value}})
    if len(conditions) == 1:
        where_expr = conditions[0]
    else:
        where_expr = {"$and": conditions}

    collection = _get_collection()
    try:
        # 先查询符合条件的切片 id
        result = collection.get(where=where_expr)
        ids_to_delete = result.get("ids", [])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            logger.info(f"按条件 {filter_dict} 删除 {len(ids_to_delete)} 条切片")
        else:
            logger.info(f"没有找到符合条件 {filter_dict} 的切片")
        return len(ids_to_delete)
    except Exception as e:
        logger.error(f"按条件删除失败: {e}", exc_info=True)
        return 0


def count_documents() -> int:
    """获取向量库中的文档总数"""
    collection = _get_collection()
    return collection.count()


def get_all_documents() -> List[Dict]:
    """获取向量库中所有文档的简要信息（不包含向量）"""
    collection = _get_collection()
    try:
        data = collection.get()
        output = []
        if data and data.get("ids"):
            for i in range(len(data["ids"])):
                text_preview = data["documents"][i][:80] + "..." if data["documents"] else ""
                output.append({
                    "id": data["ids"][i],
                    "text": text_preview,
                    "metadata": data["metadatas"][i] if data["metadatas"] else {}
                })
        return output
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        return []