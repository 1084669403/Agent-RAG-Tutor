<<<<<<< HEAD
# 智能学习助手
三层路由决策 + RAG 检索增强 + 联网搜索的智能学习助手 | A multi-layer routed RAG tutor with web search
=======
# Agent RAG Tutor 🤖📚

一个基于**三层路由决策**、**向量检索增强**与**联网搜索**的智能学习助手，体现了 Agent 核心的规划、工具使用与自校正能力。

![Python](https://img.shields.io/badge/python-3.13-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red) ![Qwen](https://img.shields.io/badge/Qwen-DashScope-green)

## ✨ 核心亮点

- 🧭 **三层意图路由**：闲聊/专业分流 → 学科大类匹配 → 细分领域+难度判定，模拟 Agent 的分诊与规划。
- 📚 **RAG 增强生成**：向量知识库检索 + 查询改写重试 + 标签放宽，降低幻觉。
- 🌐 **联网搜索回退**：知识库未命中时自动调用独立搜索 API，获取实时信息并附带可点击来源链接。
- 🔄 **自校正与降级**：检索相似度不足时自动改写查询，多级联网搜索降级保障可用性。
- 🧠 **会话缓存**：上下文感知的检索缓存，避免重复计算。
- 🗑️ **知识库管理**：支持文档上传、切片、按分类/知识库/单片删除。
- 📐 **公式渲染**：自动转换 LaTeX 语法，数学公式美观显示。

## 🏗️ 架构图（简化）
用户提问 → 意图分类（闲聊/专业）
├── 闲聊 → 轻量模型直接回复
└── 专业 → 学科路由 → 领域+难度 → RAG决策
├── 需要检索 → 向量库检索（改写重试）
│ ├── 命中 → 注入资料生成 + 来源标注
│ └── 未命中 → 联网搜索（独立API/原生/兼容）
│ └── 成功 → 带链接回答
│ └── 失败 → 纯模型回答
└── 不需要检索 → 专业模型直接生成

text

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/tianming/agent-rag-tutor.git
cd agent-rag-tutor
2. 安装依赖
bash
pip install -r requirements.txt
3. 配置环境变量
复制 .env.example 为 .env，填入你的阿里云 DashScope API Key 和 AI 搜索开放平台 Key：

bash
cp .env.example .env
# 编辑 .env，填入真实密钥
所需的主要环境变量包括：

QWEN_API_KEY（DashScope）

WEB_SEARCH_API_KEY（AI 搜索开放平台）

WEB_SEARCH_HOST（搜索域名）

其他可选变量参见 .env.example

4. 启动应用
bash
streamlit run main.py
📸 演示
https://assets/demo.gif

🛠️ 技术栈
大模型：阿里云 Qwen (qwen-flash / qwen-max) via DashScope

嵌入模型：text-embedding-v2

向量数据库：Chroma

文档处理：LangChain Text Splitters, PyPDF2, python-docx

Web 框架：Streamlit

联网搜索：阿里云 AI 搜索开放平台

📂 项目结构（公开部分）
text
src/
├── ai_service.py          # 核心路由、检索、联网搜索逻辑
├── vector_store.py         # Chroma 向量库增删查
├── document_processor.py   # 文档解析与切片
├── embedding_service.py    # 文本嵌入服务
├── pages/
│   ├── 1_AI_Q&A.py        # 聊天界面
│   └── 5_Knowledge_Base.py # 知识库管理
└── config.py              # 配置模板
📝 许可
MIT License
>>>>>>> 717725f (Initial commit: Agent RAG Tutor)
