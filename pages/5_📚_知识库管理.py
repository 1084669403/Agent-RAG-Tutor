"""
知识库上传与管理页面。

提供标准化的知识库文档上传、标签指定与自动向量化入库功能。
页面需要登录态，顶部侧边栏由公共组件统一渲染。
支持 PDF / DOCX / TXT / Markdown 格式，上传后自动解析、切片、向量化存储。
支持查看和删除已存储的切片。
"""
import streamlit as st
import os
import tempfile

from utils import require_login, render_sidebar, safe_rerun, handle_exception
from document_processor import process_document
from vector_store import add_documents, count_documents, get_all_documents, delete_by_ids, delete_by_filter
from config import Config
from utils import logger


# ==================== 权限与界面初始化 ====================
require_login()
render_sidebar()

st.title("📚 知识库管理")
st.info("✨ 上传学习资料，自动切分为语义片段并存储到向量库，后续 AI 答疑可直接引用")

# ==================== 学科标签选项（与路由体系对齐） ====================
CATEGORY_OPTIONS = {
    "science": "理工科",
    "humanities": "人文社科",
    "language": "语言类",
    "programming": "编程类"
}

SUB_FIELD_OPTIONS = {
    "science": ["高等数学", "大学物理", "化学基础", "生物科学", "其他理工"],
    "humanities": ["中国古代史", "世界历史", "哲学原理", "经济学基础", "其他文科"],
    "language": ["英语", "日语", "语文基础", "其他语言"],
    "programming": ["Python", "Java", "C/C++", "数据结构", "其他编程"]
}

DIFFICULTY_OPTIONS = ["basic", "advanced", "exam", "competition"]
DIFFICULTY_LABELS = {
    "basic": "基础",
    "advanced": "进阶",
    "exam": "应试/考研",
    "competition": "竞赛"
}


# ==================== 学科标签选择（表单外，可即时更新细分领域） ====================
st.subheader("📤 上传新知识库")

col1, col2 = st.columns(2)
with col1:
    category = st.selectbox(
        "学科大类",
        options=list(CATEGORY_OPTIONS.keys()),
        format_func=lambda x: CATEGORY_OPTIONS[x]
    )
    difficulty = st.selectbox(
        "难度等级",
        options=DIFFICULTY_OPTIONS,
        format_func=lambda x: DIFFICULTY_LABELS[x]
    )
with col2:
    # 根据选择的大类动态显示细分领域（页面刷新后立即更新）
    sub_fields = SUB_FIELD_OPTIONS.get(category, ["其他"])
    sub_field = st.selectbox("细分领域", options=sub_fields)
    knowledge_name = st.text_input("知识库名称（如教材名称）", placeholder="例：同济高数第七版上册")

# ==================== 上传表单（仅文件、名称和提交按钮） ====================
with st.form("upload_form", clear_on_submit=False):
    uploaded_file = st.file_uploader(
        "选择文件（支持 PDF / DOCX / TXT / Markdown）",
        type=["pdf", "docx", "doc", "txt", "md"]
    )

    submit_btn = st.form_submit_button("🚀 开始上传并处理", type="primary", use_container_width=True)

    if submit_btn:
        if not uploaded_file:
            st.warning("⚠️ 请先选择文件")
        elif not knowledge_name.strip():
            st.warning("⚠️ 请填写知识库名称")
        else:
            # 保存上传文件到临时目录
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                with st.spinner("🔄 正在解析文件并生成向量切片，请稍候..."):
                    # 构建元数据模板（使用表单外的选择值）
                    metadata = {
                        "category": category,
                        "sub_field": sub_field,
                        "difficulty": difficulty,
                        "knowledge_name": knowledge_name.strip()
                    }

                    # 调用文档处理流水线
                    texts, metas = process_document(
                        tmp_path,
                        metadata_template=metadata
                    )

                    if not texts:
                        st.error("❌ 文档解析后内容为空，请检查文件是否有效")
                    else:
                        # 存入向量库
                        add_documents(texts, metas)
                        st.success(f"✅ 上传成功！共生成 {len(texts)} 个语义切片，已存入向量库")
                        logger.info(
                            f"用户 {st.session_state.user_id} 上传知识库 '{knowledge_name}'，"
                            f"标签: {category}/{sub_field}/{difficulty}，切片数: {len(texts)}"
                        )
            except Exception as e:
                handle_exception(e, "上传处理失败")
            finally:
                # 清理临时文件
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)


# ==================== 已有知识库概览 ====================
st.divider()
st.subheader("📊 当前向量库概况")

try:
    doc_count = count_documents()
    if doc_count > 0:
        st.metric("向量库总切片数", doc_count)
    else:
        st.info("向量库暂无内容，请上传知识库文件")
except Exception as e:
    handle_exception(e, "获取向量库信息失败")


# ==================== 切片管理与删除 ====================
st.divider()
st.subheader("🗑️ 管理已有切片")

docs = get_all_documents()
if docs:
    # 1. 按学科大类 / 知识库名称批量删除
    st.write("**按分类 / 知识库批量删除**")
    col_del1, col_del2 = st.columns(2)

    with col_del1:
        # 选择要操作的大类（与上传时一致）
        del_category = st.selectbox(
            "选择学科大类",
            options=list(CATEGORY_OPTIONS.keys()),
            format_func=lambda x: CATEGORY_OPTIONS[x],
            key="del_cat"
        )
    with col_del2:
        # 根据当前已选择的大类，列出该大类下已有的知识库名称（去重）
        kb_names = sorted(list(set(
            doc["metadata"].get("knowledge_name", "")
            for doc in docs
            if doc["metadata"].get("category") == del_category and doc["metadata"].get("knowledge_name")
        )))
        if kb_names:
            del_kb = st.selectbox("选择知识库（可选）", options=["全部"] + kb_names, key="del_kb")
        else:
            del_kb = "全部"

    # 删除按钮：点击后根据选择的大类和（可选的）知识库名称执行批量删除
    if st.button("🗑️ 删除该分类 / 知识库下的所有切片"):
        filter_dict = {"category": del_category}
        if del_kb != "全部":
            filter_dict["knowledge_name"] = del_kb
        deleted = delete_by_filter(filter_dict)
        if deleted > 0:
            st.success(f"成功删除 {deleted} 个切片")
            safe_rerun()
        else:
            st.info("没有找到符合条件的切片")

    st.divider()

    # 2. 精确的单切片多选删除（保留原有功能）
    st.write("**按单个切片删除（可多选）**")

    # 构建显示选项：切片 ID 前8位 + 知识库名 + 细分领域
    doc_options = {
        doc["id"]: (
            f"{doc['id'][:8]}... | "
            f"{doc['metadata'].get('knowledge_name', '无名称')} | "
            f"{doc['metadata'].get('sub_field', '无领域')}"
        )
        for doc in docs
    }

    selected_ids = st.multiselect(
        "选择要删除的切片",
        options=list(doc_options.keys()),
        format_func=lambda x: doc_options[x]
    )

    if st.button("🗑️ 删除选中切片"):
        if selected_ids:
            delete_by_ids(selected_ids)
            st.success(f"已删除 {len(selected_ids)} 个切片")
            safe_rerun()
        else:
            st.warning("请至少选择一个切片")

    # 显示当前切片总数
    st.caption(f"当前共有 {len(docs)} 个切片")
else:
    st.info("向量库暂无切片，请上传知识库文件")