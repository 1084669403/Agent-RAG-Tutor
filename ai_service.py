# ai_service.py
"""
AI 服务模块。

封装大模型调用、多级路由分类、RAG 检索与回答生成。
所有 AI 能力通过阿里云 DashScope API 实现，支持轻量与专业模型分流。
已集成：
- 查询改写重试（相关度不足时自动改写再检索）
- 联网搜索增强（知识库无结果时自动调用互联网搜索并标注来源）
"""
import json
import re
import streamlit as st
import requests

from config import Config
from utils import logger, rate_limit


# ===================== HTTP 会话（复用连接） =====================
@st.cache_resource(ttl=3600)
def _get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


_session = _get_session()


# ===================== 通用 AI 调用（支持联网搜索，返回字典） =====================
def _call_ai(messages: list, temperature: float = None,
             max_tokens: int = None, model: str = None,
             enable_search: bool = False,
             search_options: dict = None) -> dict:
    """
    底层 AI 调用，支持联网搜索增强。

    通过阿里云 DashScope 兼容模式 API 发起请求，可选择性开启联网搜索。
    会自动从响应中提取生成文本与可能的搜索引用结果。

    Args:
        messages:         对话消息列表，格式 [{"role": "user", "content": "..."}]
        temperature:      生成温度，控制随机性
        max_tokens:       最大输出 Token 数
        model:            模型名称，不传则使用配置默认模型
        enable_search:    是否开启联网搜索增强
        search_options:   搜索选项字典，将合并到默认搜索配置中

    Returns:
        dict: {
            "content": str,              # AI 生成的文本回答
            "search_results": list       # 搜索引用的结构化信息（可能为空）
        }
    """
    if not Config.QWEN_API_KEY:
        return {"content": "", "search_results": []}

    headers = {"Authorization": f"Bearer {Config.QWEN_API_KEY.strip()}"}

    # ---------- 构建请求参数 ----------
    parameters = {
        "temperature": temperature if temperature is not None else Config.AI_TEMPERATURE,
        "max_tokens": max_tokens if max_tokens is not None else Config.AI_MAX_TOKENS,
    }

    # 若需要搜索，添加搜索相关参数
    if enable_search:
        parameters["enable_search"] = True
        # 默认搜索配置：要求返回来源、启用角标、强制触发搜索
        params_search_opts = {
            "enable_source": True,
            "enable_citation": True,
            "citation_format": "[ref_<number>]",
            "forced_search": True,
        }
        # 允许调用方覆盖或补充搜索选项
        if search_options:
            params_search_opts.update(search_options)
        parameters["search_options"] = params_search_opts

    payload = {
        "model": model if model else Config.QWEN_MODEL,
        "messages": messages,
        "parameters": parameters
    }

    # ---------- 发送请求 ----------
    try:
        resp = _session.post(
            Config.QWEN_API_URL, headers=headers, json=payload,
            timeout=(Config.QWEN_TIMEOUT_CONNECT, Config.QWEN_TIMEOUT_READ)
        )
        resp.raise_for_status()
        data = resp.json()

        # ---------- 解析响应 ----------
        choices = data.get("choices") or []
        content = ""
        search_results = []

        if choices:
            # 检查是否因输出长度限制被截断
            finish_reason = choices[0].get("finish_reason", "")
            if finish_reason == "length":
                logger.warning("AI 回答因 max_tokens 限制被截断")

            message = choices[0].get("message", {})
            content = message.get("content", "").strip()

            # 兼容多种可能的搜索结果存放位置
            search_results = message.get("search_results", [])
            if not search_results:
                search_results = message.get("citations", [])

        # 若消息体中没有，再尝试从顶层 output 中获取（原生 API 可能放在这里）
        if not search_results:
            search_results = data.get("output", {}).get("search_results", [])

        return {"content": content, "search_results": search_results}

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        logger.error(f"AI HTTP 错误 {status}: {e}")
        return {"content": "", "search_results": []}
    except Exception as e:
        logger.error(f"AI 调用失败: {e}", exc_info=True)
        return {"content": "", "search_results": []}

# ===================== 纯文本调用（向后兼容） =====================
def _call_ai_text(*args, **kwargs) -> str:
    """与旧版 _call_ai 接口兼容，只返回文本内容"""
    return _call_ai(*args, **kwargs)["content"]


# ===================== 原生联网搜索增强调用 =====================
def _call_ai_search(query: str) -> dict:
    """
    使用 DashScope 原生 API 进行联网搜索，返回结构化搜索结果与模型回答。
    官方文档：https://help.aliyun.com/zh/model-studio/enable-search

    Returns:
        {
            "answer": str,            # 模型基于搜索生成的回答
            "search_results": list,   # 搜索引用的结构化信息（含 title/url/text）
            "success": bool           # 是否成功获取到搜索结果
        }
    """
    if not Config.QWEN_API_KEY:
        return {"answer": "", "search_results": [], "success": False}

    headers = {"Authorization": f"Bearer {Config.QWEN_API_KEY.strip()}"}

    # 使用明确支持搜索增强的模型（qwen-max 或 qwen-plus 均可）
    model = "qwen-max"  # 也可改用 "qwen-plus"

    payload = {
        "model": model,
        "input": {
            "messages": [
                {"role": "user", "content": query}
            ]
        },
        "parameters": {
            "enable_search": True,
            "search_options": {
                "enable_source": True  # 必须为 True 才会返回来源链接
                # 不传其他参数，使用默认策略
            }
        }
    }

    search_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    try:
        resp = _session.post(
            search_url, headers=headers, json=payload,
            timeout=(Config.QWEN_TIMEOUT_CONNECT, Config.QWEN_TIMEOUT_READ)
        )
        resp.raise_for_status()
        data = resp.json()

        logger.debug(f"原生搜索响应: {json.dumps(data, ensure_ascii=False)[:500]}")

        output = data.get("output", {})
        answer = output.get("text", "")
        if not answer:
            # 部分版本可能把回答放在 choices 中
            choices = output.get("choices", [])
            if choices:
                answer = choices[0].get("message", {}).get("content", "")

        search_results = output.get("search_results", [])
        success = len(search_results) > 0

        if success:
            logger.info(f"原生搜索获得 {len(search_results)} 条来源")
        else:
            logger.warning("原生搜索未返回结构化来源，回答可能已生成但无链接")

        return {
            "answer": answer.strip(),
            "search_results": search_results,
            "success": success
        }

    except Exception as e:
        logger.error(f"原生搜索 API 调用失败: {e}", exc_info=True)
        return {"answer": "", "search_results": [], "success": False}


# ===================== 联网搜索独立 API（AI搜索开放平台） =====================
def _call_web_search(query: str, top_k: int = 5) -> dict:
    """
    调用阿里云 AI 搜索开放平台独立搜索 API，获取结构化网页搜索结果。
    该 API 稳定可靠，直接返回 title、link、snippet 等字段，适合用于来源标注。

    Args:
        query:  搜索词
        top_k:  返回结果数（默认 5）

    Returns:
        {
            "search_results": [
                {"title": str, "url": str, "snippet": str, "content": str},
                ...
            ],
            "success": bool
        }
    """
    if not Config.WEB_SEARCH_API_KEY or not Config.WEB_SEARCH_HOST:
        logger.warning("WEB_SEARCH_API_KEY 或 WEB_SEARCH_HOST 未配置，无法使用独立搜索")
        return {"search_results": [], "success": False}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.WEB_SEARCH_API_KEY.strip()}"
    }

    # 构建完整的请求 URL（强制使用 HTTPS）
    url = f"https://{Config.WEB_SEARCH_HOST}/v3/openapi/workspaces/default/web-search/ops-web-search-001"

    payload = {
        "query": query,
        "top_k": top_k,
        "content_type": "snippet"   # 返回摘要，长度适中
    }

    try:
        resp = _session.post(
            url, headers=headers, json=payload,
            timeout=(Config.QWEN_TIMEOUT_CONNECT, Config.QWEN_TIMEOUT_READ)
        )
        resp.raise_for_status()
        data = resp.json()

        # 从返回结构中提取搜索结果
        search_result_list = data.get("result", {}).get("search_result", [])
        formatted_results = []
        for item in search_result_list:
            formatted_results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),          # 注意字段名是 link
                "snippet": item.get("snippet", ""),
                "content": item.get("content", "")
            })

        success = len(formatted_results) > 0
        if success:
            logger.info(f"独立搜索获得 {len(formatted_results)} 条结果")
        else:
            logger.warning("独立搜索未返回任何结果")

        return {
            "search_results": formatted_results,
            "success": success
        }

    except Exception as e:
        logger.error(f"独立搜索 API 调用失败: {e}", exc_info=True)
        return {"search_results": [], "success": False}

# ===================== 原单模型接口（保持向后兼容） =====================
@rate_limit("ai_call", Config.RATE_LIMIT_AI_MAX,
            Config.RATE_LIMIT_AI_WINDOW, key_param="user_id")
def call_qwen_ai(user_question: str, user_id: int = 0) -> str:
    """AI 问答（返回原始 Markdown 文本）"""
    if not Config.QWEN_API_KEY:
        return "❌ 请先配置 QWEN_API_KEY 环境变量"

    question = user_question.strip()
    if not question:
        return "❌ 问题内容不能为空"

    content = _call_ai_text(
        messages=[
            {"role": "system", "content": Config.AI_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    )
    return content if content else "❌ AI 未返回有效回答"


# ===================== 一级路由：意图分类 =====================
CLASSIFY_PROMPT = (
    "你是一个智能路由分类器。请分析用户消息，判断它属于哪一类，并以JSON格式输出，不要输出其他内容。\n"
    '类别定义：\n'
    ' - "academic": 用户正在询问或表达希望学习某个学科知识、学术概念、解题、技术原理、历史事件、科学原理等，即使只是说“我想学XX”，也属于专业问题。此外，任何需要实时信息、数据、新闻、天气、股价等时效性查询，也属于 academic，需要准确、深度的回答或搜索。\n'
    ' - "chitchat": 用户在闲聊、打招呼、表达纯情绪（与学习无关）、询问日常琐事等，只需简单友好的回应。\n'
    '输出格式示例：\n'
    '{{"intent": "academic"}}\n'
    '用户消息：{user_message}'
)


def _classify_intent(user_message: str) -> str:
    """使用轻量模型对用户消息进行意图分类"""
    # 先检查明显需要搜索的时效性关键词
    time_sensitive_keywords = ["天气", "今天", "现在", "最新", "新闻", "股价", "汇率", "疫情", "比赛"]
    if any(kw in user_message for kw in time_sensitive_keywords):
        return "academic"

    prompt = CLASSIFY_PROMPT.format(user_message=user_message)
    raw = _call_ai_text(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # 分类任务需要确定性
        max_tokens=50,
        model=Config.QWEN_MODEL_LIGHT
    )
    if not raw:
        logger.warning("意图分类失败，默认归为 academic")
        return "academic"

    try:
        result = json.loads(raw)
        intent = result.get("intent", "").lower()
        if intent in ("academic", "chitchat"):
            return intent
    except json.JSONDecodeError:
        match = re.search(r'"intent"\s*:\s*"(\w+)"', raw)
        if match:
            intent = match.group(1).lower()
            if intent in ("academic", "chitchat"):
                return intent

    logger.warning(f"意图分类解析失败，原始输出: {raw[:100]}")
    return "academic"


# ===================== 二级路由：学科大类 =====================
CATEGORY_PROMPT = (
    "你是一个学科分类器。请根据用户的问题，判断它属于哪个学科大类，并以JSON格式输出，不要输出其他内容。\n"
    '学科大类选项：\n'
    ' - "science": 数学、物理、化学、生物、地理等理工科\n'
    ' - "humanities": 历史、政治、哲学、文学、经济学等人文社科\n'
    ' - "language": 英语、日语、语文等语言学习\n'
    ' - "programming": Python、Java、算法、数据结构等编程技术\n'
    '输出格式示例：\n'
    '{{"category": "science", "confidence": 0.95}}\n'
    '用户问题：{user_message}'
)


def _classify_category(user_message: str) -> dict:
    """使用轻量模型判断学科大类"""
    prompt = CATEGORY_PROMPT.format(user_message=user_message)
    raw = _call_ai_text(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=80,
        model=Config.QWEN_MODEL_LIGHT
    )
    if not raw:
        logger.warning("二级路由失败，默认归为 science")
        return {"category": "science", "confidence": 0.0}

    try:
        result = json.loads(raw)
        cat = result.get("category", "science")
        conf = result.get("confidence", 0.5)
        return {"category": cat, "confidence": conf}
    except json.JSONDecodeError:
        match = re.search(r'"category"\s*:\s*"(\w+)"', raw)
        if match:
            return {"category": match.group(1), "confidence": 0.5}
    logger.warning(f"二级路由解析失败: {raw[:100]}")
    return {"category": "science", "confidence": 0.0}


# ===================== 三级路由：细分领域+难度+RAG决策 =====================
DETAIL_PROMPT = (
    "你是一个学习助手路由决策器。根据用户问题以及已知的学科大类，输出以下信息，并以JSON格式返回，不要其他内容。\n"
    "已知学科大类：{category}\n\n"
    "请输出：\n"
    ' - "sub_field": 细分领域（中文，如"高等数学"/"大学物理"/"中国古代史"/"Python"等）\n'
    ' - "difficulty": 难度等级（"basic"/"advanced"/"exam"/"competition"）\n'
    ' - "need_rag": 是否需要检索知识库（true/false）\n'
    ' - "rag_tags": 检索标签数组，自动由category、sub_field、difficulty拼接，格式为["{{category}}", "sub_field的具体值", "{{difficulty}}"]\n\n'
    '判断need_rag的规则：\n'
    ' - true：概念定义、定理推导、公式、考点解析、教材原文、真题解答、知识点对比等\n'
    ' - false：学习规划、方法建议、经验分享、发散思考、主观讨论等\n\n'
    '输出示例：\n'
    '{{"sub_field": "高等数学", "difficulty": "basic", "need_rag": true, "rag_tags": ["science", "高等数学", "basic"]}}\n'
    '用户问题：{user_message}'
)


def _classify_detail(user_message: str, category: str) -> dict:
    """判断细分领域、难度、是否触发RAG"""
    prompt = DETAIL_PROMPT.format(user_message=user_message, category=category)
    raw = _call_ai_text(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150,
        model=Config.QWEN_MODEL_LIGHT
    )
    if not raw:
        logger.warning("三级路由失败，使用默认值")
        return _default_detail(category)

    try:
        result = json.loads(raw)
        if all(k in result for k in ("sub_field", "difficulty", "need_rag", "rag_tags")):
            return result
    except json.JSONDecodeError:
        pass

    logger.warning(f"三级路由解析失败: {raw[:100]}")
    return _default_detail(category)


def _default_detail(category: str) -> dict:
    """当三级路由失败时的默认值"""
    return {
        "sub_field": "通用",
        "difficulty": "basic",
        "need_rag": False,
        "rag_tags": [category, "通用", "basic"]
    }


# ===================== 查询改写辅助函数 =====================
def _rewrite_query(original: str) -> str:
    """用轻量模型改写问题，使其更聚焦核心概念"""
    prompt = (
        "请将以下用户问题改写成一个简洁、聚焦核心概念的检索查询，直接返回改写后的问题，不要解释。\n"
        f"原始问题：{original}"
    )
    rewritten = _call_ai_text(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100,
        model=Config.QWEN_MODEL_LIGHT
    )
    if not rewritten or len(rewritten.strip()) < 2:
        logger.warning("查询改写失败，使用原始查询")
        return original
    logger.info(f"查询改写: '{original}' → '{rewritten.strip()}'")
    return rewritten.strip()


# ===================== 生成闲聊回答（轻量模型） =====================
def _generate_chitchat(messages: list) -> str:
    """用轻量模型生成简短友好的闲聊回复"""
    system = {
        "role": "system",
        "content": (
            "你叫小明，是一个免费、耐心、鼓励学生的智能学习助手，说话像真朋友，绝对不背设定。\n"
            "回答规则：\n"
            "1. 简单问题（如“你是谁”“在吗”“你好”）只用 1-2 句自然回应，不要自我介绍一堆。\n"
            "2. 用户分享心情、日常，可多聊几句，但不超过 33 字。仔细体会对方的情绪，感同身受地回应，并给予正面、温暖的鼓励或建议。\n"
            "3. 语气口语化，像发微信一样，适当加表情（😊📚✨）。\n"
            "4. 语气随意一点，像好朋友闲聊一样，若用户说了沉重或者复杂的事情可以适当的体恤用户的心情，并多聊几句，但是不超过333个字。\n"
            "5.  永远保持积极乐观，不输出任何消极、负面、脏话或政治敏感内容（包括领导人、敏感事件等），遇到此类话题友好转移至学习相关。\n\n"
            "正确示范：\n"
            "用户：“你是谁？”\n"
            "小明：“我就是小明呀，一个专属于你的学习搭子～”\n"
            "用户：“今天好累”\n"
            "小明：“那就先别硬撑啦，起来走动走动喝口水，等你缓一缓咱们再一起学 📚”"
        )
    }
    return _call_ai_text(
        messages=[system] + messages,
        temperature=0.73,
        max_tokens=300,
        model=Config.QWEN_MODEL_LIGHT
    )


# ===================== 生成专业回答（强模型，支持RAG注入） =====================
def _generate_academic(messages: list, references: str = "") -> str:
    """用专业大模型生成深度学科解答，可注入参考资料"""
    recent_messages = messages[-10:] if len(messages) > 10 else messages

    system_content = (
        "你是一个专业的学习导师，请对以下学科问题进行深度解答，要求：\n"
        " - 逻辑清晰，分点阐述\n"
        " - 如有公式、代码，请规范输出\n"
        " - 如需使用数学公式，请用 $...$ 包裹行内公式，用 $$...$$ 包裹独立公式\n"
        " - 代码请用 Markdown 代码块\n"
        " - 尽量引用可靠来源或经典教材观点\n"
        " - 若问题存在多种解释，请客观列举"
    )

    if references:
        system_content += (
            "\n\n【重要】请优先基于以下参考资料进行回答，并在回答中适当引用资料内容。"
             "严格禁止在回答中出现“参考资料”“参考来源”“来源”等章节标题，也不要用列表形式列出资料内容。"
            "如果资料不足以回答问题，可以结合你自己的知识补充，但需明确指出哪些内容来自资料，哪些是常识补充。\n"
            f"参考资料：\n{references}"
        )

    system = {"role": "system", "content": system_content}
    return _call_ai_text(
        messages=[system] + recent_messages,
        temperature=0.3,
        max_tokens=4000,
        model="qwen-max",
        enable_search=True,
    )


# ===================== 带路由的问答（供聊天页面调用） =====================
@rate_limit("ai_route", Config.RATE_LIMIT_AI_MAX,
            Config.RATE_LIMIT_AI_WINDOW, key_param="user_id")
def call_qa_with_routing(user_id: int, history_messages: list,
                         new_question: str) -> dict:
    """
    智能路由问答：一级→二级→三级（仅专业），再分流到闲聊/专业模型。
    集成 RAG 检索（查询改写重试）和联网搜索回退（统一来源标注）。
    返回的字典同时包含 answer 和路由信息。
    """
    # 提前声明所有变量，避免后续分支未赋值导致 Unbound 错误
    answer = ""
    references = ""
    search_source = "knowledge_base"  # 默认来源：本地知识库
    local_sources = []                # 本地检索的来源信息
    web_sources = []                  # 联网搜索的来源信息（带链接）

    try:
        if not Config.QWEN_API_KEY:
            return {
                "answer": "❌ 请先配置 QWEN_API_KEY 环境变量",
                "need_rag": False,
                "rag_tags": [],
                "category": "",
                "sub_field": "",
                "difficulty": "",
                "search_source": "none"
            }

        question = new_question.strip()
        if not question:
            return {
                "answer": "❌ 问题内容不能为空",
                "need_rag": False,
                "rag_tags": [],
                "category": "",
                "sub_field": "",
                "difficulty": "",
                "search_source": "none"
            }

        # 1. 一级路由：意图分类
        intent = _classify_intent(question)

        # 2. 闲聊分支
        if intent == "chitchat":
            full_messages = list(history_messages)
            full_messages.append({"role": "user", "content": question})
            answer = _generate_chitchat(full_messages)
            return {
                "answer": answer if answer else "❌ AI 未返回有效回答",
                "need_rag": False,
                "rag_tags": [],
                "category": "",
                "sub_field": "",
                "difficulty": "",
                "search_source": "none"
            }

        # 3. 专业分支：二级+三级路由
        route_info = _classify_category(question)
        category = route_info.get("category", "science")
        detail = _classify_detail(question, category)
        need_rag = detail.get("need_rag", False)
        rag_tags = detail.get("rag_tags", [])
        sub_field = detail.get("sub_field", "通用")
        difficulty = detail.get("difficulty", "basic")

        if not need_rag:
            rag_tags = []

        # 4. 构建完整消息列表（历史 + 当前问题）
        full_messages = list(history_messages)
        full_messages.append({"role": "user", "content": question})

        # 5. RAG 检索（含查询改写重试）
        if need_rag and rag_tags:
            try:
                from vector_store import search
                filter_dict = {}
                if len(rag_tags) >= 1:
                    filter_dict["category"] = rag_tags[0]
                if len(rag_tags) >= 2:
                    filter_dict["sub_field"] = rag_tags[1]
                if len(rag_tags) >= 3:
                    filter_dict["difficulty"] = rag_tags[2]

                results = search(query_text=question, filter_tags=filter_dict, top_k=5)

                # 标签放宽
                if not results or all(r.get("distance", 1) > 0.6 for r in results):
                    if "difficulty" in filter_dict:
                        logger.info("精确检索无结果，尝试放宽条件（移除 difficulty）")
                        relaxed_filter = {k: v for k, v in filter_dict.items() if k != "difficulty"}
                        results = search(query_text=question, filter_tags=relaxed_filter, top_k=5)
                        filter_dict = relaxed_filter

                best_distance = min((r.get("distance", 1) for r in results), default=1) if results else 1
                need_rewrite = (not results) or (best_distance > 0.6)

                if need_rewrite:
                    logger.info(f"当前最佳距离 {best_distance:.3f}，尝试查询改写重试")
                    rewritten_query = _rewrite_query(question)
                    retry_results = search(query_text=rewritten_query, filter_tags=filter_dict, top_k=5)
                    if retry_results:
                        results = retry_results
                        logger.info("改写后检索获得结果")
                    else:
                        logger.info("改写检索仍无结果")

                # 构建参考资料和来源元数据
                if results:
                    filtered_results = [r for r in results if r.get("distance") is not None and r["distance"] < 0.6]
                    if filtered_results:
                        references = "\n\n---\n".join([r["text"] for r in filtered_results])
                        logger.info(f"检索到 {len(filtered_results)} 条参考资料")
                        local_sources = [
                            {
                                "source": r["metadata"].get("source", "未知文件"),
                                "knowledge_name": r["metadata"].get("knowledge_name", ""),
                            }
                            for r in filtered_results
                        ]
                    else:
                        logger.info("检索结果相似度不足，降级为纯模型回答")
                else:
                    logger.info("知识库无相关内容，降级为纯模型回答")
            except ImportError:
                logger.warning("vector_store 模块不可用，跳过检索")
            except Exception as e:
                logger.error(f"RAG 检索异常: {e}")

        # 6. 联网搜索回退（本地知识库无结果且需要知识时触发）
        if not references and need_rag:
            logger.info("本地知识库无结果，尝试联网搜索（独立搜索 API）")

            # 第一级：独立搜索 API
            web_result = _call_web_search(query=question, top_k=5)
            if web_result.get("success"):
                web_results = web_result["search_results"]
                ref_texts = []
                for r in web_results[:5]:
                    ref_texts.append(
                        f"标题：{r.get('title', '')}\n内容：{r.get('snippet', r.get('content', ''))}\n来源：{r.get('url', '')}"
                    )
                references = "\n\n---\n".join(ref_texts)
                answer = _generate_academic(full_messages, references)
                search_source = "web"
                web_sources = web_results[:5]   # 暂存，最后由统一标注添加链接
                # 不再提前 return，让流程继续走到 #7 和末尾的统一来源标注

            if not web_sources:
                # 第二级：原生生成搜索
                logger.info("独立搜索无结果，降级为原生生成搜索")
                native_result = _call_ai_search(query=question)
                if native_result.get("success"):
                    answer = native_result["answer"]
                    web_results = native_result.get("search_results", [])
                    search_source = "web"
                    web_sources = web_results[:5]
                else:
                    # 第三级：兼容模式（带角标）
                    logger.info("原生生成搜索失败，降级为兼容模式联网搜索")
                    search_result = _call_ai(
                        messages=[{"role": "user", "content": question}],
                        temperature=0.3,
                        max_tokens=3000,
                        model="qwen-max",
                        enable_search=True,
                        search_options={
                            "enable_citation": True,
                            "citation_format": "[ref_<number>]"
                        }
                    )
                    answer = search_result.get("content", "")
                    search_source = "web"
                    # 兼容模式不返回结构化来源，web_sources 保持为空（答案自带角标）
                    if not answer:
                        logger.info("联网搜索全部失败，降级为纯模型回答")
                        answer = _generate_academic(full_messages)
                        search_source = "knowledge_base"  # 回退为纯模型，无标注

        # 7. 生成回答（如果以上分支均未生成 answer，则使用本地资料或纯模型）
        if not answer:
            answer = _generate_academic(full_messages, references)

        # ---------- 统一来源标注（只添加一次） ----------
        if search_source == "knowledge_base" and references and local_sources:
            # 去重后追加知识库文件名
            seen = set()
            unique_sources = []
            for s in local_sources:
                key = s["source"]
                if key not in seen:
                    seen.add(key)
                    unique_sources.append(s)
            if unique_sources:
                answer += "\n\n📚 **参考来源（知识库）**：\n"
                for i, s in enumerate(unique_sources, 1):
                    name = s["knowledge_name"] if s["knowledge_name"] else s["source"]
                    answer += f"{i}. {name}\n"
        elif search_source == "web" and web_sources:
            # 追加联网搜索带链接的来源
            answer += "\n\n📚 **参考来源**：\n"
            for i, r in enumerate(web_sources, 1):
                title = r.get('title', '来源')
                url = r.get('url', '')
                if url:
                    answer += f"{i}. [{title}]({url})\n"
                else:
                    answer += f"{i}. {title}\n"
        # 兼容模式且 web_sources 为空时，答案自带角标，不再额外添加来源

        return {
            "answer": answer if answer else "❌ AI 未返回有效回答",
            "need_rag": need_rag,
            "rag_tags": rag_tags,
            "category": category,
            "sub_field": sub_field,
            "difficulty": difficulty,
            "search_source": search_source
        }

    except Exception as e:
        logger.error(f"路由处理异常: {e}", exc_info=True)
        return {
            "answer": "❌ 路由服务内部错误，请稍后再试",
            "need_rag": False,
            "rag_tags": [],
            "category": "",
            "sub_field": "",
            "difficulty": "",
            "search_source": "none"
        }


# ===================== 生成选择题 =====================
@rate_limit("ai_quiz", Config.RATE_LIMIT_AI_MAX,
            Config.RATE_LIMIT_AI_WINDOW, key_param="user_id")
def generate_quiz_by_ai(topic: str, user_id: int = 0,
                        num_questions: int = 5) -> list:
    """调用 AI 生成选择题，返回 list[dict]"""
    if not Config.QWEN_API_KEY:
        return []

    num_questions = max(1, min(num_questions, Config.QUIZ_MAX_COUNT))

    prompt = f"""请为以下知识点生成 {num_questions} 道单选题。
要求：
1. 每题 4 个选项（A/B/C/D），只有一个正确答案
2. 附带解析
3. 难度适中，适合学生自测

知识点：{topic}

请严格按照以下 JSON 数组格式返回，不要返回其他任何内容：
[
  {{
    "topic": "知识点主题",
    "question_text": "题目内容",
    "option_a": "选项A",
    "option_b": "选项B",
    "option_c": "选项C",
    "option_d": "选项D",
    "correct_answer": "A",
    "explanation": "解析"
  }}
]"""

    raw = _call_ai_text(
        messages=[
            {"role": "system", "content": "你是专业出题老师，只返回 JSON 数组，不要任何额外文字。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=4000,
    )
    if not raw:
        return []
    return _parse_json_from_ai(raw)


# ===================== 生成思维导图 / 整理笔记 =====================
@rate_limit("ai_mindmap", Config.RATE_LIMIT_AI_MAX,
            Config.RATE_LIMIT_AI_WINDOW, key_param="user_id")
def generate_mindmap_by_ai(knowledge_text: str, user_id: int = 0) -> str:
    """调用 AI 将零散知识点整理为结构化 Markdown 笔记"""
    if not Config.QWEN_API_KEY:
        return ""

    prompt = f"""请将以下知识点内容整理为结构清晰的学习笔记（Markdown 格式）。
要求：
1. 提取核心概念，建立层次结构，使用标准的 Markdown 标题语法：**每个标题的井号 '#' 后面必须有一个空格**（例如 "### 1. 子标题" 而不是 "###1. 子标题"）。
2. 使用列表、表格等组织信息。
3. 重点内容加粗标注。
4. 末尾添加简要总结。
5. 适合学生快速回顾。

原始内容：
{knowledge_text}"""

    content = _call_ai_text(
        messages=[
            {"role": "system", "content": "你是知识整理专家，擅长将零散内容整理为结构化 Markdown 学习笔记。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=4000,
    )
    return content


# ===================== AI 返回 JSON 解析（健壮版） =====================
def _parse_json_from_ai(raw: str) -> list:
    """从 AI 返回的文本中提取 JSON 数组（兼容多种格式）"""
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    match = re.search(r"$$(.+?)$$", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    logger.error(f"AI JSON 解析失败，原始内容前 200 字: {raw[:200]}")
    return []