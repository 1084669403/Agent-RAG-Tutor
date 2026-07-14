# config.py
"""
应用统一配置模块。

集中管理所有可调参数，支持通过环境变量覆盖默认值。
包含 AI 模型、向量库、联网搜索、数据库、安全、限流等配置。
"""
import os
import logging
from typing import List
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的环境变量


# [OPT] 任务11：安全的环境变量读取函数，防止格式错误导致启动崩溃
def _safe_int(env_key: str, default: int) -> int:
    """安全读取整数类型的环境变量，格式错误时回退到默认值"""
    raw = os.getenv(env_key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logging.getLogger("ai_learning_assistant").warning(
            f"环境变量 {env_key}='{raw}' 不是有效整数，使用默认值 {default}"
        )
        return default


def _safe_float(env_key: str, default: float) -> float:
    """安全读取浮点数类型的环境变量，格式错误时回退到默认值"""
    raw = os.getenv(env_key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logging.getLogger("ai_learning_assistant").warning(
            f"环境变量 {env_key}='{raw}' 不是有效浮点数，使用默认值 {default}"
        )
        return default


class Config:
    """应用统一配置类（所有可调参数集中管理）"""

    # ==================== 基础配置 ====================
    APP_NAME = "AI智能学习助手"                         # 应用名称
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"  # 是否开启调试模式
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # 运行环境：development / production

    # ==================== 数据库配置 ====================
    DB_FILE = os.getenv("DB_FILE", "qa_data.db")          # SQLite 数据库文件路径
    DB_TIMEOUT = _safe_int("DB_TIMEOUT", 30)              # 数据库连接超时（秒）

    # ==================== 密码安全配置 ====================
    PBKDF2_ITERATIONS = _safe_int("PBKDF2_ITERATIONS", 600000)  # PBKDF2 迭代次数（越高越安全，但登录稍慢）
    PASSWORD_MIN_LENGTH = _safe_int("PASSWORD_MIN_LENGTH", 8)   # 密码最小长度
    HASH_ALGORITHM = "sha256"                                   # 密码哈希算法
    HASH_VERSION = "1"                                          # 哈希格式版本（方便未来升级）

    # ==================== AI 模型配置 ====================
    # 注意：所有密钥和敏感信息通过环境变量注入，严禁硬编码到代码中
    QWEN_API_KEY = os.getenv("QWEN_API_KEY")                    # 阿里云 DashScope API Key（必填）
    QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-turbo")          # 默认模型（用于非关键调用）
    QWEN_MODEL_LIGHT = os.getenv("QWEN_MODEL_LIGHT", "qwen-flash")  # 轻量模型（路由、闲聊）
    QWEN_MODEL_PRO = os.getenv("QWEN_MODEL_PRO", "qwen-plus")       # 专业模型（深度回答）
    QWEN_API_URL = os.getenv(
        "QWEN_API_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"  # DashScope 兼容 OpenAI 接口地址
    )
    QWEN_TIMEOUT_CONNECT = _safe_int("QWEN_TIMEOUT_CONNECT", 20)  # API 连接超时（秒）
    QWEN_TIMEOUT_READ = _safe_int("QWEN_TIMEOUT_READ", 90)        # API 读取超时（秒）

    # 嵌入与向量库配置
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v2")  # 文本嵌入模型（用于知识库向量化）
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")  # Chroma 向量库持久化目录

    # 联网搜索独立 API 配置（阿里云 AI 搜索开放平台）
    WEB_SEARCH_API_KEY = os.getenv("WEB_SEARCH_API_KEY", "")  # 搜索 API Key（可在控制台获取）
    WEB_SEARCH_HOST = os.getenv("WEB_SEARCH_HOST", "")        # 搜索服务域名（不含协议头，如 example.com）

    # AI 生成参数
    AI_SYSTEM_PROMPT = os.getenv(
        "AI_SYSTEM_PROMPT",
        "你是一名耐心专业的学生辅导老师，讲解知识点通俗易懂、条理清晰、分点罗列。"  # 全局系统提示词
    )
    AI_MAX_TOKENS = _safe_int("AI_MAX_TOKENS", 2048)    # 默认最大输出 Token 数
    AI_TEMPERATURE = _safe_float("AI_TEMPERATURE", 0.6)  # 默认生成温度（0~1，越高越随机）

    # 出题配置
    QUIZ_DEFAULT_COUNT = _safe_int("QUIZ_DEFAULT_COUNT", 5)  # 默认生成选择题数量
    QUIZ_MAX_COUNT = _safe_int("QUIZ_MAX_COUNT", 10)         # 单次最大生成数量

    # ==================== 复习算法配置 ====================
    # 艾宾浩斯复习间隔（天），索引对应 mastery_level
    # mastery_level=0 时不安排复习（完全不会→重新学习）
    REVIEW_INTERVALS: List[int] = [1, 2, 4, 7, 15, 30]

    # 掌握度等级（0~5）
    MASTERY_LEVELS = {
        0: "完全不会",
        1: "略有印象",
        2: "基本理解",
        3: "熟练掌握",
        4: "融会贯通",
        5: "完全掌握",
    }

    # [OPT] 任务12：掌握度调整系数，从 algorithm.py 移入配置中心
    # 值越大，该掌握度下的复习间隔越长
    MASTERY_FACTORS = {
        0: 0.5,   # 完全不会 → 间隔缩短一半，尽快重学
        1: 0.8,   # 略有印象 → 间隔略短
        2: 1.0,   # 基本理解 → 标准间隔
        3: 1.3,   # 熟练掌握 → 间隔延长 30%
        4: 1.8,   # 融会贯通 → 间隔延长 80%
        5: 2.5,   # 完全掌握 → 间隔延长 150%
    }

    # ==================== 频率限制配置 ====================
    RATE_LIMIT_LOGIN_MAX = _safe_int("RATE_LIMIT_LOGIN_MAX", 5)      # 登录接口每分钟最大请求数
    RATE_LIMIT_LOGIN_WINDOW = _safe_int("RATE_LIMIT_LOGIN_WINDOW", 60)  # 登录限流时间窗口（秒）
    RATE_LIMIT_AI_MAX = _safe_int("RATE_LIMIT_AI_MAX", 10)            # AI 接口每分钟最大请求数
    RATE_LIMIT_AI_WINDOW = _safe_int("RATE_LIMIT_AI_WINDOW", 60)      # AI 限流时间窗口（秒）

    # ==================== 内容长度限制 ====================
    MAX_NOTE_TITLE_LENGTH = _safe_int("MAX_NOTE_TITLE_LENGTH", 200)     # 笔记标题最大字符数
    MAX_NOTE_CONTENT_LENGTH = _safe_int("MAX_NOTE_CONTENT_LENGTH", 50000)  # 笔记内容最大字符数
    MAX_QUESTION_LENGTH = _safe_int("MAX_QUESTION_LENGTH", 1000)         # 问题最大字符数
    MAX_ANSWER_LENGTH = _safe_int("MAX_ANSWER_LENGTH", 50000)            # 回答最大字符数

    # ==================== 日志配置 ====================
    LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if DEBUG else "INFO")  # 日志级别（DEBUG / INFO / WARNING / ERROR）

    @classmethod
    def validate(cls):
        """启动前验证关键配置，确保必需的环境变量已设置"""
        if not cls.QWEN_API_KEY and cls.ENVIRONMENT == "production":
            raise ValueError("生产环境必须配置 QWEN_API_KEY 环境变量")

        if not cls.QWEN_API_KEY and cls.ENVIRONMENT != "production":
            logging.getLogger("ai_learning_assistant").warning(
                "⚠️ QWEN_API_KEY 未配置，AI 功能将不可用"
            )

        if cls.PBKDF2_ITERATIONS < 100000:
            raise ValueError("PBKDF2_ITERATIONS 至少为 100000，当前值过于不安全")
        if cls.PASSWORD_MIN_LENGTH < 6:
            raise ValueError("密码最小长度至少为 6 位")