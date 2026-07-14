# embedding_service.py
"""
文本嵌入服务模块。

调用阿里云 DashScope 文本嵌入 API，将文本转换为向量列表，
用于后续的知识库检索与语义匹配。
"""
import requests
import streamlit as st
from typing import List
from config import Config
from utils import logger


# ===================== HTTP 会话（复用连接） =====================
@st.cache_resource(ttl=3600)
def _get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


_session = _get_session()


# ===================== 核心嵌入函数 =====================
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    调用 DashScope 嵌入 API，将文本列表转换为向量列表。

    Args:
        texts: 待嵌入的文本列表，单次最多 25 条。

    Returns:
        向量列表，每个向量为浮点数列表；失败时返回空列表。
    """
    if not Config.QWEN_API_KEY:
        logger.error("嵌入 API 调用失败：缺少 QWEN_API_KEY")
        return []

    if not texts:
        return []

    # 阿里云嵌入 API 单次最多处理 25 条文本
    if len(texts) > 25:
        logger.warning(f"单次嵌入文本数量超过 25 条（实际 {len(texts)}），将分批处理")
        results = []
        for i in range(0, len(texts), 25):
            batch = texts[i:i+25]
            results.extend(_call_embedding_api(batch))
        return results

    return _call_embedding_api(texts)


def _call_embedding_api(texts: List[str]) -> List[List[float]]:
    """
    底层嵌入 API 调用，返回向量列表。
    """
    headers = {"Authorization": f"Bearer {Config.QWEN_API_KEY.strip()}"}
    payload = {
        "model": Config.EMBEDDING_MODEL,  # 同步版本，v2 需要异步任务，这里先用 v1
        "input": {
            "texts": texts              # 官方要求：input.texts 为字符串数组
        }
    }

    try:
        resp = _session.post(
            "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
            headers=headers,
            json=payload,
            timeout=(Config.QWEN_TIMEOUT_CONNECT, Config.QWEN_TIMEOUT_READ)
        )
        resp.raise_for_status()
        data = resp.json()

        # 检查是否有错误信息
        if data.get("code"):
            error_msg = data.get("message", "未知错误")
            logger.error(f"嵌入 API 返回错误: {error_msg}")
            return []

        # 提取 embedding 列表
        output = data.get("output", {})
        embeddings = output.get("embeddings", [])
        if not embeddings:
            logger.warning("嵌入 API 返回空向量")
            return []

        vectors = []
        for emb in embeddings:
            vec = emb.get("embedding")
            if vec:
                vectors.append(vec)
            else:
                logger.warning("某条文本嵌入为空，跳过")
                vectors.append([])
        return vectors

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        logger.error(f"嵌入 API HTTP 错误 {status}: {e}")
        return []
    except Exception as e:
        logger.error(f"嵌入 API 调用失败: {e}", exc_info=True)
        return []