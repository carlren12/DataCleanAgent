# -*- coding: utf-8 -*-
"""
四种置信度估计策略实现
基于开题报告的策略一至策略四

策略一：Prompt-based - 通过Prompt引导LLM输出置信度
策略二：Self-Consistency - 多次采样，统计一致性
策略三：Logit-based - 通过logprobs计算token概率
策略四：Hybrid - 加权融合前三者

【重要】默认使用本地 vLLM GPU 推理（极快），备选 CLAW API
"""
import os
import re
import time
import json
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    CLAW_API_KEY, CLAW_BASE_URL,
    LOCAL_VLLM_URL, LOCAL_VLLM_API_KEY, LOCAL_VLLM_MODEL,
    MODELS, DEFAULT_MODEL,
    STRATEGY_CONFIG, THRESHOLD_CONFIG
)


@dataclass
class ConfidenceResult:
    """置信度估计结果"""
    confidence: float  # 置信度 [0, 1]
    label: int         # 判断标签 0=非噪声, 1=噪声
    raw_response: str  # 原始响应
    strategy: str       # 策略名称
    model: str          # 模型名称
    elapsed_time: float # 耗时(秒)
    metadata: Dict = None  # 额外信息


# ============ API 客户端工厂 ============

class VLLMAPIClient:
    """本地 vLLM GPU 推理客户端"""
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat_completions(
        self,
        messages: List[Dict],
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 150,
        logprobs: bool = False,
        top_logprobs: int = 5,
        timeout: int = 120,
    ) -> Tuple[Dict, float]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = top_logprobs

        t0 = time.time()
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=timeout
            )
            elapsed = time.time() - t0
            data = r.json()
            if r.status_code != 200:
                print(f"[VLLM Error] Status {r.status_code}: {str(data)[:200]}")
            return data, elapsed
        except Exception as e:
            print(f"[VLLM Exception] {e}")
            return {"error": str(e)}, time.time() - t0


class CLAWAPIClient:
    """CLAW API 客户端（远程，速率受限）"""
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat_completions(
        self,
        messages: List[Dict],
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 150,
        logprobs: bool = False,
        top_logprobs: int = 5,
        timeout: int = 120,
    ) -> Tuple[Dict, float]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = top_logprobs

        t0 = time.time()
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=timeout
            )
            elapsed = time.time() - t0
            data = r.json()
            if r.status_code != 200:
                print(f"[CLAW Error] Status {r.status_code}: {str(data)[:200]}")
            return data, elapsed
        except Exception as e:
            print(f"[CLAW Exception] {e}")
            return {"error": str(e)}, time.time() - t0


# 全局客户端缓存
_vllm_client: Optional[VLLMAPIClient] = None
_claw_client: Optional[CLAWAPIClient] = None


def get_api_client(model_key: str = DEFAULT_MODEL):
    """根据模型类型返回对应 API 客户端（工厂模式）"""
    global _vllm_client, _claw_client
    model_config = MODELS.get(model_key, MODELS[DEFAULT_MODEL])
    provider = model_config.get("provider", "claw")

    if provider == "vllm":
        if _vllm_client is None:
            _vllm_client = VLLMAPIClient(LOCAL_VLLM_URL, LOCAL_VLLM_API_KEY)
        return _vllm_client
    else:
        if _claw_client is None:
            _claw_client = CLAWAPIClient(CLAW_API_KEY, CLAW_BASE_URL)
        return _claw_client


# ============ 响应解析工具 ============

def parse_json_response(text: str) -> Optional[Dict]:
    """从文本中提取 JSON 响应"""
    text = text.strip()
    # 尝试从代码块中提取
    if "```json" in text:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
    elif "```" in text:
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
    # 尝试直接解析
    try:
        return json.loads(text)
    except:
        pass
    # 尝试提取 JSON 对象（宽松匹配）
    match = re.search(r"\{[^{}]*\"label\"[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    return None


def parse_text_response(text: str) -> Tuple[int, float]:
    """从纯文本中提取 label 和 confidence"""
    text_lower = text.lower()
    label = 0
    confidence = 0.5

    # 判断 label
    if any(kw in text_lower for kw in ["噪声", "noise", "有问题", "错误", "低质量"]):
        label = 1
    elif any(kw in text_lower for kw in ["干净", "clean", "高质量", "正常", "合格"]):
        label = 0

    # 提取置信度
    match = re.search(r"(?:置信[度称]|confidence|confidence[:：])\s*[:：]?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        confidence = float(match.group(1))
        if confidence > 1:
            confidence /= 100.0

    return label, confidence


# ============ 策略一：Prompt-based ============

def strategy_prompt_based(
    sample: Dict,
    model_key: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> ConfidenceResult:
    """策略一：Prompt-based 置信度估计"""
    model_config = MODELS[model_key]
    call_name = model_config["call_name"]
    strategy_cfg = STRATEGY_CONFIG["prompt"]

    prompt_template = strategy_cfg["prompt_template"]
    temperature = strategy_cfg["temperature"]
    max_tokens = strategy_cfg["max_tokens"]

    user_prompt = prompt_template.format(
        question=sample["question"],
        answer=sample["answer"]
    )

    messages = [
        {"role": "system", "content": "You are a professional data quality evaluator."},
        {"role": "user", "content": user_prompt}
    ]

    client = get_api_client(model_key)
    data, elapsed = client.chat_completions(
        messages=messages,
        model=call_name,
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=False,
    )

    raw_response = ""
    label = 0
    confidence = 0.5

    if "error" not in data:
        try:
            raw_response = data["choices"][0]["message"]["content"]
            result = parse_json_response(raw_response)
            if result:
                label = result.get("label", 0)
                confidence = result.get("confidence", 50) / 100.0
            else:
                label, confidence = parse_text_response(raw_response)
        except Exception as e:
            if verbose:
                print(f"[Parse Error] {e}")
            raw_response = str(data)

    return ConfidenceResult(
        confidence=confidence,
        label=label,
        raw_response=raw_response,
        strategy="prompt_based",
        model=model_config["name"],
        elapsed_time=elapsed,
        metadata={"temperature": temperature, "call_name": call_name}
    )


# ============ 策略二：Self-Consistency ============

def strategy_self_consistency(
    sample: Dict,
    model_key: str = DEFAULT_MODEL,
    n_samples: int = 5,
    verbose: bool = False,
) -> ConfidenceResult:
    """策略二：Self-Consistency 置信度估计"""
    model_config = MODELS[model_key]
    call_name = model_config["call_name"]
    strategy_cfg = STRATEGY_CONFIG["self_consistency"]

    n_samples = strategy_cfg.get("n_samples", n_samples)
    temperature = strategy_cfg["temperature"]
    max_tokens = strategy_cfg["max_tokens"]
    prompt_variants = strategy_cfg["prompt_variants"][:n_samples]

    votes = []
    raw_responses = []

    for i, variant_prompt in enumerate(prompt_variants):
        messages = [
            {"role": "system", "content": "You are a data quality evaluator."},
            {"role": "user", "content": f"{variant_prompt}\n\n患者问题：{sample['question']}\n医生回答：{sample['answer']}"}
        ]

        client = get_api_client(model_key)
        data, elapsed = client.chat_completions(
            messages=messages,
            model=call_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if "error" not in data:
            try:
                content = data["choices"][0]["message"]["content"]
                raw_responses.append(content)
                result = parse_json_response(content)
                if result:
                    votes.append(result.get("label", 0))
                else:
                    lbl, _ = parse_text_response(content)
                    votes.append(lbl)
            except Exception as e:
                if verbose:
                    print(f"[SC Error on variant {i+1}] {e}")

    # 计算置信度：噪声比例越低越置信
    if votes:
        noise_ratio = sum(votes) / len(votes)
        confidence = 1.0 - noise_ratio
        label = 1 if noise_ratio > 0.5 else 0
    else:
        confidence = 0.5
        label = 0

    return ConfidenceResult(
        confidence=confidence,
        label=label,
        raw_response=str(raw_responses),
        strategy="self_consistency",
        model=model_config["name"],
        elapsed_time=sum(v[1] for v in [(d, e) for d, e in [(data, elapsed)]]),  # simplified
        metadata={"n_variants": len(prompt_variants), "votes": votes}
    )


# ============ 策略三：Logit-based ============

def strategy_logit_based(
    sample: Dict,
    model_key: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> ConfidenceResult:
    """策略三：Logit-based 置信度估计（基于 token logprobs）"""
    model_config = MODELS[model_key]
    call_name = model_config["call_name"]
    strategy_cfg = STRATEGY_CONFIG["logit_based"]

    temperature = strategy_cfg["temperature"]
    max_tokens = strategy_cfg["max_tokens"]
    answer_tokens = strategy_cfg["answer_tokens"]
    top_k = strategy_cfg["top_k"]

    # 构造简化的二分类 prompt
    messages = [
        {"role": "system", "content": "You are a medical data quality classifier. Output only '0' (clean) or '1' (noisy)."},
        {"role": "user", "content": f"判断这条医患对话质量。患者问题：{sample['question']}\n医生回答：{sample['answer']}\n\n输出0或1："}
    ]

    client = get_api_client(model_key)
    data, elapsed = client.chat_completions(
        messages=messages,
        model=call_name,
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=top_k,
    )

    raw_response = ""
    label = 0
    confidence = 0.5

    if "error" not in data:
        try:
            choice = data["choices"][0]
            raw_response = choice["message"]["content"]

            # 从 logprobs 中提取置信度
            if "logprobs" in choice and choice["logprobs"]:
                top_logprobs = choice["logprobs"].get("content", [])
                if top_logprobs:
                    # 计算 token 概率加权
                    token_probs = []
                    for lp in top_logprobs:
                        token = lp["token"].strip()
                        logp = lp["logprob"]
                        prob = math.exp(logp) if logp > -50 else 0  # 避免 underflow
                        token_probs.append((token, prob))

                    # 判断哪个 token（0 或 1）的概率更高
                    prob_0 = sum(p for t, p in token_probs if t in ["0", "0", "Clean", "clean"])
                    prob_1 = sum(p for t, p in token_probs if t in ["1", "1", "Noisy", "noisy"])

                    total = prob_0 + prob_1 + 1e-9
                    label = 1 if prob_1 > prob_0 else 0
                    confidence = max(prob_0, prob_1) / total
            else:
                # 无 logprobs 时，解析文本响应
                label, confidence = parse_text_response(raw_response)
        except Exception as e:
            if verbose:
                print(f"[Logit Error] {e}")
            raw_response = str(data)

    return ConfidenceResult(
        confidence=confidence,
        label=label,
        raw_response=raw_response,
        strategy="logit_based",
        model=model_config["name"],
        elapsed_time=elapsed,
        metadata={"temperature": temperature, "call_name": call_name}
    )


# ============ 策略四：Hybrid ============

def strategy_hybrid(
    sample: Dict,
    model_key: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> ConfidenceResult:
    """策略四：Hybrid 加权融合"""
    weights = STRATEGY_CONFIG["hybrid"]["weights"]

    r1 = strategy_prompt_based(sample, model_key, verbose)
    r2 = strategy_self_consistency(sample, model_key, verbose=verbose)
    r3 = strategy_logit_based(sample, model_key, verbose)

    # 加权置信度
    fused_confidence = (
        weights["prompt"] * r1.confidence +
        weights["self_consistency"] * r2.confidence +
        weights["logit_based"] * r3.confidence
    )

    # 投票决定 label
    label_votes = [r1.label, r2.label, r3.label]
    label = 1 if sum(label_votes) >= 2 else 0

    return ConfidenceResult(
        confidence=fused_confidence,
        label=label,
        raw_response=f"prompt:{r1.confidence:.3f} sc:{r2.confidence:.3f} logit:{r3.confidence:.3f}",
        strategy="hybrid",
        model=MODELS[model_key]["name"],
        elapsed_time=r1.elapsed_time + r2.elapsed_time + r3.elapsed_time,
        metadata={
            "weights": weights,
            "component_confidences": {
                "prompt": r1.confidence,
                "self_consistency": r2.confidence,
                "logit_based": r3.confidence,
            }
        }
    )


# ============ 统一接口 ============

def estimate_confidence(
    sample: Dict,
    strategy: str = "hybrid",
    model_key: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> ConfidenceResult:
    """统一入口：估计单条样本的置信度"""
    strategies = {
        "prompt": strategy_prompt_based,
        "self_consistency": strategy_self_consistency,
        "logit_based": strategy_logit_based,
        "hybrid": strategy_hybrid,
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}")

    return strategies[strategy](sample, model_key, verbose)


def estimate_confidence_batch(
    samples: List[Dict],
    strategy: str = "hybrid",
    model_key: str = DEFAULT_MODEL,
    show_progress: bool = True,
) -> List[ConfidenceResult]:
    """批量估计置信度（GPU 推理高速版）"""
    iterator = tqdm(samples, desc=f"GPU推理 ({strategy})") if show_progress else samples

    results = []
    for sample in iterator:
        result = estimate_confidence(sample, strategy, model_key)
        results.append(result)
        # vLLM 很快，不需要额外 sleep；CLAW API 才需要限速
        if result.elapsed_time < 0.1:
            time.sleep(0.05)  # 保护远程 API

    return results


# ============ 单元测试 ============

if __name__ == "__main__":
    import math
    print("=" * 60)
    print("  Confidence Strategies - vLLM GPU 加速版")
    print("=" * 60)

    test_sample = {
        "id": 1,
        "question": "感冒了应该怎么办？",
        "answer": "建议多休息，多喝水，保持室内空气流通。",
    }

    strategies = ["prompt", "self_consistency", "logit_based", "hybrid"]

    for strat in strategies:
        print(f"\n[Test] Strategy: {strat}")
        result = estimate_confidence(test_sample, strategy=strat, verbose=True)
        print(f"  → label={result.label}, confidence={result.confidence:.3f}, time={result.elapsed_time:.2f}s")
        print(f"  → model={result.model}")