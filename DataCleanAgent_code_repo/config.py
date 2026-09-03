# -*- coding: utf-8 -*-
"""
Data Cleaning Agent 实验配置
所有配置集中管理
"""
import os

# ============ API 配置 ============
# 注意：API 密钥一律从环境变量读取，请勿在代码中硬编码密钥。
# 使用前请先设置环境变量（见 README.md 的 "Configuration" 一节）：
#   export LOCAL_VLLM_URL="http://<your-gpu-server>:8002/v1"
#   export LOCAL_VLLM_API_KEY="local-dev"
#   export CLAW_API_KEY="your-claw-api-key"
#   export ZHIPU_API_KEY="your-zhipu-api-key"

# 主模型：本地 vLLM GPU 推理（Qwen2.5-7B）
LOCAL_VLLM_URL = os.getenv("LOCAL_VLLM_URL", "http://localhost:8002/v1")
LOCAL_VLLM_API_KEY = os.getenv("LOCAL_VLLM_API_KEY", "local-dev")
LOCAL_VLLM_MODEL = "Qwen2.5-7B"

# 备选：交大 CLAW API（远程，不推荐大规模实验）
CLAW_API_KEY = os.getenv("CLAW_API_KEY", "")
CLAW_BASE_URL = "https://models.sjtu.edu.cn/api/v1"

# 备选智谱 GLM-4-Flash（免费额度）
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

# ============ 模型配置 ============
MODELS = {
    # 主模型：本地 vLLM（Qwen2.5-7B，GPU 推理，高速）
    "local_vllm": {
        "name": "Qwen2.5-7B",
        "call_name": "Qwen2.5-7B",
        "provider": "vllm",
        "size": "7B",
        "quality_rank": 2,
        "speed_rank": 1,  # GPU 本地推理，极快
        "logprobs": True,
        "supports_thinking": False,
    },
    # 备选：DeepSeek-V3.2 (685B) - 用于高质量对比实验
    "deepseek_v32": {
        "name": "DeepSeek-V3.2",
        "call_name": "deepseek-chat",
        "provider": "claw",
        "size": "685B",
        "quality_rank": 1,
        "speed_rank": 3,
        "logprobs": True,
        "supports_thinking": False,
    },
    # 备选：Qwen3-Coder-30B
    "qwen3_coder": {
        "name": "Qwen3-Coder-30B",
        "call_name": "qwen3coder",
        "provider": "claw",
        "size": "30B",
        "quality_rank": 2,
        "speed_rank": 1,
        "logprobs": True,
        "supports_thinking": False,
    },
    # 备选：MiniMax-M2.5
    "minimax": {
        "name": "MiniMax-M2.5",
        "call_name": "minimax",
        "provider": "claw",
        "size": "230B",
        "quality_rank": 2,
        "speed_rank": 2,
        "logprobs": True,
        "supports_thinking": False,
    },
}

# 主模型配置：默认使用本地 vLLM（GPU 高速推理）
DEFAULT_MODEL = "local_vllm"

# ============ 置信度策略配置 ============
STRATEGY_CONFIG = {
    # 策略一：Prompt-based
    "prompt": {
        "prompt_template": """你是一个医疗数据质量审计专家。请仔细审查以下医患对话，判断其是否存在噪声。

【医患对话】
患者问题：{question}
医生回答：{answer}

【噪声类型定义（请逐一排查）】
类型1 - 事实错误：医生回答与医学常识明显相悖（如把相关当因果、混淆疾病症状等）
类型2 - 逻辑错误：回答中的推理过程自相矛盾或违背医学逻辑
类型3 - 语义冲突：患者问题和医生回答在语义上不匹配或明显不一致
类型4 - 不完整数据：医生回答未完整解答患者问题，或回避了核心问题

【判断方法】
先仔细阅读，逐一检查是否存在上述4类噪声。如果发现任意一类，给出低置信度；如果四类都不存在，再判断为"干净"。

【输出格式】
请严格按JSON格式输出，不要有其他内容：
{{"label": 0或1, "confidence": 0-100整数, "reason": "简要说明判断依据"}}

label含义：0=干净数据，1=噪声数据
confidence含义：0=完全没有把握，100=完全确定。""",
        "temperature": 0.3,
        "max_tokens": 150,
    },

    # 策略二：Self-Consistency
    "self_consistency": {
        "n_samples": 5,  # n种不同表述
        "temperature": 0.7,  # 需要随机性
        "max_tokens": 100,
        "prompt_variants": [
            "请判断以下医患对话是否存在质量问题。",
            "以下医患对话数据质量如何？",
            "评估这条医患对话：是否存在数据噪声？",
            "请分析这条医患对话是否值得保留为训练数据。",
            "从数据质量角度，这条医患对话合格吗？",
        ],
    },

    # 策略三：Logit-based
    "logit_based": {
        "temperature": 0.3,
        "max_tokens": 50,
        "answer_tokens": ["0", "1", "是噪声", "非噪声", "clean", "noise"],
        "use_top_logprobs": True,
        "top_k": 5,
    },

    # 策略四：Hybrid（加权融合）
    "hybrid": {
        "weights": {
            "prompt": 0.4,
            "self_consistency": 0.4,
            "logit_based": 0.2,
        },
    },
}

# ============ ECE 校准配置 ============
CALIBRATION_CONFIG = {
    "n_bins": 10,  # 置信度分箱数
    "method": "temperature_scaling",  # or "platt_scaling"
    "validation_ratio": 0.2,  # 验证集比例
}

# ============ 三档阈值配置 ============
THRESHOLD_CONFIG = {
    "high_confidence": 0.8,    # c >= 0.8：直接清洗
    "low_confidence": 0.4,     # c < 0.4：拒绝清洗
    # 0.4 <= c < 0.8：人工确认（实验时视为保留，待人工审核）
}

# ============ 数据配置 ============
DATA_CONFIG = {
    # 医疗对话数据集路径（绝对路径）
    "med_dialogue_dir": r"C:\Users\27878\WorkBuddy\20260408170411\datasets\med_79w\Chinese-medical-dialogue-data-master\Data_数据",

    # 科室文件夹和文件映射（实际目录结构）
    "departments": {
        "IM_内科": "IM_内科/内科5000-33000.csv",
        "OAGD_妇产科": "OAGD_妇产科/妇产科6-28000.csv",
        "Surgical_外科": "Surgical_外科/外科5-14000.csv",
        "Pediatric_儿科": "Pediatric_儿科/儿科5-14000.csv",
        "Andriatria_男科": "Andriatria_男科/男科5-13000.csv",
        "Oncology_肿瘤科": "Oncology_肿瘤科/肿瘤科5-10000.csv",
    },

    # 中文简称映射（用于显示）
    "dept_alias": {
        "IM_内科": "内科",
        "OAGD_妇产科": "妇产科",
        "Surgical_外科": "外科",
        "Pediatric_儿科": "儿科",
        "Andriatria_男科": "男科",
        "Oncology_肿瘤科": "肿瘤科",
    },

    # 实验采样配置
    "pilot_sample_size": 100,    # 小规模验证：100条
    "full_sample_size": 5000,    # 完整实验：5000条
    "noise_injection_ratio": 0.3,  # 注入噪声比例
}

# ============ 中文语义噪声分类体系（4类16种） ============
NOISE_TAXONOMY = {
    "类型一：事实性错误（Factual Errors）": {
        "子类型1.1": "医学知识错误（Medical Knowledge Errors）",
        "子类型1.2": "数值/剂量错误（Numerical/Dosage Errors）",
        "子类型1.3": "过时医学知识（Outdated Medical Knowledge）",
        "子类型1.4": "虚构医疗建议（Fabricated Medical Advice）",
    },
    "类型二：逻辑性错误（Logical Errors）": {
        "子类型2.1": "问答不匹配（Question-Answer Mismatch）",
        "子类型2.2": "逻辑断裂（Logical Discontinuity）",
        "子类型2.3": "自相矛盾（Self-Contradiction）",
        "子类型2.4": "因果关系错误（False Causality）",
    },
    "类型三：语言质量问题（Language Quality Issues）": {
        "子类型3.1": "语言不规范（Non-Standard Language）",
        "子类型3.2": "内容截断（Content Truncation）",
        "子类型3.3": "乱码/格式错误（Garbled/Format Errors）",
        "子类型3.4": "重复冗余（Redundancy）",
    },
    "类型四：专业知识偏差（Domain Knowledge Biases）": {
        "子类型4.1": "命名实体错误（Named Entity Errors）",
        "子类型4.2": "医学术语混用（Terminology Mixing）",
        "子类型4.3": "科室专业性偏差（Departmental Bias）",
        "子类型4.4": "文化/地区性偏差（Cultural/Regional Bias）",
    },
}

# ============ DRQ 评估指标 ============
DRQ_METRICS = {
    "quality": ["precision", "recall", "f1", "accuracy"],
    "validity": ["validity_score", "completeness", "consistency"],
    "efficiency": ["cleaning_time", "cost_per_sample"],
}

# ============ 输出配置 ============
OUTPUT_DIR = "outputs/experiment_results"
FIGURES_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
