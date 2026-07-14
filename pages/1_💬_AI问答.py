"""
AI 对话答疑页面 (聊天流 UI)。

提供连续对话式的智能答疑界面，自动携带上下文。
页面需要登录态，顶部侧边栏由公共组件统一渲染。
用户可连续提问，AI 结合历史生成解答，所有问答持久化到数据库，
但每个用户最多保留最近 N 条记录，超出部分自动删除以节省存储空间。
"""
import streamlit as st
from datetime import datetime

from utils import (
    require_login, render_sidebar, safe_rerun,
    handle_exception, utc_to_local,
)
import db_manager
from ai_service import call_qa_with_routing   # 使用带路由的新接口


# ==================== 辅助函数 ====================
def _format_latex(text: str) -> str:
    """
    将 AI 输出中可能出现的 \(...\) 和 \[...\] 分隔符
    转换为 Streamlit 可识别的 $...$ 和 $$...$$ 格式，使数学公式能够正确渲染。
    """
    text = text.replace("\\(", "$").replace("\\)", "$")
    text = text.replace("\\[", "$$").replace("\\]", "$$")
    return text


# ---- 权限与界面初始化 ----
require_login()      # 强制登录，未登录时跳转
render_sidebar()     # 渲染全局侧边栏（用户信息、导航等）

st.title("💬 AI 智能知识点答疑")
st.info("📖 连续提问，AI 会结合上下文为你精准拆解讲解")
st.info("📖 连续提问，AI 会结合上下文为你精准拆解讲解\n💡 提示：前往「📚 知识库管理」上传教材或笔记，AI 将结合你的专属资料回答")

# ---- 配置项 ----
# 每个用户最多保留的对话轮数（一轮 = 一次提问 + 一次回答）
MAX_CONVERSATIONS = 20

# 初始化缓存字典
if "rag_cache" not in st.session_state:
    st.session_state.rag_cache = {}

# ---- 初始化聊天历史 ----
# 首次进入页面时，从数据库加载最近 N 条问答记录并转换为消息列表
if "messages" not in st.session_state:
    try:
        # 获取用户最近 N 条历史记录（数据库默认按时间倒序返回）
        raw_records = db_manager.get_recent_qa_by_user(
            st.session_state.user_id, limit=MAX_CONVERSATIONS
        )
        # 将倒序记录翻转为正序，以便消息按时间从早到晚显示
        raw_records.reverse()

        messages = []
        for rec in raw_records:
            # 从记录中提取时间，转换为本地时区后格式化为字符串
            time_str = rec.get("asked_at", "")
            local_time = ""
            try:
                local_time = utc_to_local(
                    datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                ).strftime("%Y-%m-%d %H:%M")
            except Exception:
                local_time = time_str  # 转换失败则使用原始字符串

            # 用户消息
            messages.append({
                "role": "user",
                "content": rec.get("question", ""),
                "time": local_time,
            })
            # AI 回答
            messages.append({
                "role": "assistant",
                "content": rec.get("answer", ""),
                "time": local_time,  # 回答与问题使用相同时间戳展示
            })
        st.session_state.messages = messages
    except Exception as e:
        handle_exception(e, "加载历史对话失败")
        st.session_state.messages = []

# ===================== 历史消息渲染 =====================
# 按顺序展示所有聊天消息，每条消息以气泡形式显示
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # 自动转换 LaTeX 分隔符，使公式美观显示
        st.markdown(_format_latex(msg["content"]))
        # 若消息包含时间戳，则在气泡下方添加小字时间提示
        if msg.get("time"):
            st.caption(f"⏰ {msg['time']}")

# ===================== 用户输入处理 =====================
# 使用 st.chat_input 提供固定在底部的输入框，体验类似聊天应用
if prompt := st.chat_input("请输入你的问题..."):
    # 1. 将用户消息追加到会话状态中，并立即显示
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")  # 当前时间用于展示
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "time": now_str,
    })
    with st.chat_message("user"):
        # 用户问题通常不包含公式，但仍可转换（保持健壮性）
        st.markdown(_format_latex(prompt))
        st.caption(f"⏰ {now_str}")

    # 2. 构建历史消息列表（不含刚追加的用户消息），调用带路由的 AI
    with st.spinner("AI 正在生成解答..."):
        recent_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        # 构建缓存键：结合最近3轮对话上下文，避免不同语境下返回错误缓存
        cache_key = str(prompt)

        # 检查缓存
        if cache_key in st.session_state.rag_cache:
            routing_result = st.session_state.rag_cache[cache_key]
            ai_answer = routing_result["answer"]
        else:
            # 未命中缓存，调用路由
            routing_result = call_qa_with_routing(
                user_id=st.session_state.user_id,
                history_messages=recent_history,
                new_question=prompt,
            )
            ai_answer = routing_result["answer"]

            # 如果不是错误回答，存入缓存
            if routing_result and not ai_answer.startswith("❌"):
                st.session_state.rag_cache[cache_key] = routing_result

        # 容错处理：如果路由结果为空，构建默认错误字典
        if routing_result is None:
            ai_answer = "❌ 路由服务异常，请稍后再试"
            routing_result = {
                "answer": ai_answer,
                "need_rag": False,
                "rag_tags": [],
                "category": "",
                "sub_field": "",
                "difficulty": ""
            }

        # 统一更新路由调试信息
        st.session_state.latest_routing = {
            "need_rag": routing_result.get("need_rag", False),
            "rag_tags": routing_result.get("rag_tags", []),
            "category": routing_result.get("category", ""),
            "sub_field": routing_result.get("sub_field", ""),
            "difficulty": routing_result.get("difficulty", "")
        }

    # 3. 错误处理
    if ai_answer is None:
        ai_answer = "⚠️ 请求过于频繁，请稍后再试"
    elif ai_answer.startswith("❌"):
        # AI 服务返回的错误信息直接展示
        pass

    # 4. 将 AI 回答加入会话状态，并显示
    ai_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_answer,
        "time": ai_time,
    })
    with st.chat_message("assistant"):
        # 渲染时自动处理公式
        st.markdown(_format_latex(ai_answer))
        st.caption(f"⏰ {ai_time}")

    # 5. 持久化当前问答对到数据库
    try:
        db_manager.add_qa_record(prompt, ai_answer, st.session_state.user_id)
    except Exception as e:
        handle_exception(e, "保存问答记录失败")

    # 6. 自动清理超出上限的旧记录，释放存储空间
    try:
        if hasattr(db_manager, "cleanup_old_records"):
            db_manager.cleanup_old_records(
                st.session_state.user_id, keep=MAX_CONVERSATIONS
            )
    except Exception as e:
        handle_exception(e, "清理旧记录失败")


# ===================== 可选功能：清空当前会话（不删数据库记录） =====================
with st.sidebar:
    if st.button("🧹 清空当前会话"):
        st.session_state.messages = []
        safe_rerun()