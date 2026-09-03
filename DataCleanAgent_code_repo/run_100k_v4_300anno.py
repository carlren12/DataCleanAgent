"""
DataCleanAgent 并发实验脚本 - 10万条正式版
- ThreadPoolExecutor 8线程并发
- 四种置信度：P-Conf / SC-Conf / L-Conf / H-Conf
- 数据格式: row_index, department, question, answer, expert_label
- 前300条(row_index 0-299)为专家标注（ECE验证）
- 编码: gbk（与源数据一致）
"""
import os, sys, json, time, re, requests, math, csv, random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== 配置 ====================
MODEL          = "Qwen2.5-7B"
API_URL        = "http://localhost:8002/v1/chat/completions"
HEADERS        = {"Authorization": "Bearer local-dev", "Content-Type": "application/json"}

# 数据文件（10万条，300条专家标注 row_index 0-299）
DATA_FILE      = "/data/users/renhao/experiment/raw_annotation_10w-已标注300.csv"
OUTPUT_FILE    = "/data/users/renhao/experiment/results_100k_v4_300anno.json"
LOG_FILE       = "/data/users/renhao/experiment/run_100k_v4_300anno.log"
SAVE_EVERY     = 100
MAX_SAMPLES    = None   # None=全量100,000条
SC_ROUNDS      = 3
MAX_WORKERS    = 8      # 外层并发数

PROMPT_P = (
    "患者问题：{question}\n"
    "医生回答：{answer}\n\n"
    "请评估该医生回答的质量问题严重程度。\n"
    "0=无质量问题，1=有严重质量问题。\n"
    "请只输出一个0到1之间的数字（可以带1位小数）："
)

PROMPT_SC = (
    "患者问题：{question}\n"
    "医生回答：{answer}\n\n"
    "该医生回答是否存在质量问题？\n"
    "请仅输出 YES 或 NO（不要输出其他内容）："
)

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

# ==================== 置信度解析 ====================
def extract_conf(text):
    """从模型输出中提取0-1数字，找不到则返回None。"""
    text = text.strip()
    m = re.search(r'(?<!\d)([01]?\.\d+|0|1)(?!\d)', text)
    if m:
        v = float(m.group(1))
        if 0 <= v <= 1:
            return round(v, 4)
    m = re.search(r'置信度[：:]\s*([0-9.]+)', text)
    if m:
        v = float(m.group(1))
        if 0 <= v <= 1:
            return round(v, 4)
        if 0 < v <= 100:
            return round(v / 100, 4)
    return None

# ==================== P-Conf ====================
def compute_p_conf(question, answer):
    """策略P1 - Prompt-based：1次采样（temperature=0.1），直接要求输出置信度数字和理由"""
    prompt = PROMPT_P.format(question=question, answer=answer)
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.1,
    }
    t0 = time.time()
    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        elapsed = time.time() - t0
        if resp.status_code != 200:
            return {"confidence": 0.5, "elapsed": elapsed, "text": ""}
        body = resp.json()
        text = body["choices"][0]["message"]["content"]
        conf = extract_conf(text)
        if conf is None:
            return {"confidence": 0.5, "elapsed": elapsed, "text": text}
    except Exception as e:
        return {"confidence": 0.5, "elapsed": time.time() - t0, "text": ""}
    return {"confidence": conf, "elapsed": elapsed, "text": text}

# ==================== SC-Conf（Self-Consistency 多数投票）====================
def compute_sc_conf(question, answer):
    """
    策略P2 - Self-Consistency：N次独立采样（temperature=0.9），
    每次输出 YES/NO，统计多数决策占比作为置信度：conf = 一致决策次数 / N
    """
    prompt = PROMPT_SC.format(question=question, answer=answer)
    messages = [{"role": "user", "content": prompt}]
    decisions = []  # "YES" or "NO"
    sc_time = 0.0
    for _ in range(SC_ROUNDS):
        t0 = time.time()
        try:
            resp = requests.post(API_URL, headers=HEADERS, json={
                "model": MODEL, "messages": messages,
                "max_tokens": 5, "temperature": 0.9
            }, timeout=60)
            sc_time += time.time() - t0
            text = resp.json()["choices"][0]["message"]["content"].strip().upper()
            if "YES" in text or "是" in text or "有" in text:
                decisions.append("YES")
            else:
                decisions.append("NO")
        except:
            decisions.append("NO")
    if not decisions:
        return {"confidence": 0.5, "elapsed": sc_time, "decisions": [], "vote_detail": ""}
    count_yes = decisions.count("YES")
    count_no = decisions.count("NO")
    sc_conf = max(count_yes, count_no) / len(decisions)  # 一致率
    vote_detail = f"YES:{count_yes}/NO:{count_no}"
    return {"confidence": round(sc_conf, 4), "elapsed": sc_time,
            "decisions": decisions, "vote_detail": vote_detail}

# ==================== L-Conf ====================
def compute_l_conf(messages):
    """策略P3 - Logit-based：通过输出token概率提取置信度"""
    import math, time
    binary_content = messages[-1]["content"] + "\n\n以上医生回答是否存在质量问题？仅输出 YES 或 NO。"
    binary_msgs = messages[:-1] + [{"role": "user", "content": binary_content}]
    payload = {
        "model": MODEL,
        "messages": binary_msgs,
        "max_tokens": 5,
        "temperature": 0.1,
        "logprobs": True,
        "top_logprobs": 5
    }
    t0 = time.time()
    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        elapsed = time.time() - t0
        if resp.status_code != 200:
            return {"confidence": 0.5, "elapsed": elapsed}
        body = resp.json()
        text = body["choices"][0]["message"]["content"].strip().upper()
        logprobs_data = body["choices"][0].get("logprobs", {})
    except Exception as e:
        return {"confidence": 0.5, "elapsed": time.time() - t0}

    raw = 0.5
    if "YES" in text or "是" in text:
        raw = 0.8
    elif "NO" in text or "否" in text:
        raw = 0.2

    calibrated = raw
    if logprobs_data and "content" in logprobs_data:
        for tok in logprobs_data["content"]:
            yes_lp = None
            no_lp = None
            candidates = [tok] + tok.get("top_logprobs", [])
            for t in candidates:
                ts = t["token"].upper().strip()
                if ts in ("YES", "Y", "是", "有"):
                    yes_lp = t["logprob"]
                elif ts in ("NO", "N", "否", "无"):
                    no_lp = t["logprob"]
            if yes_lp is not None or no_lp is not None:
                logits = []
                names = []
                if yes_lp is not None:
                    logits.append(yes_lp); names.append("yes")
                if no_lp is not None:
                    logits.append(no_lp); names.append("no")
                max_l = max(logits)
                exp_l = [math.exp(l - max_l) for l in logits]
                sum_exp = sum(exp_l)
                probs = [e / sum_exp for e in exp_l]
                if "yes" in names:
                    calibrated = round(probs[names.index("yes")], 4)
                break

    return {"confidence": calibrated, "elapsed": elapsed}

# ==================== H-Conf ====================
def compute_h_conf(p, sc, l):
    h = round(0.4 * p + 0.4 * sc + 0.2 * l, 6)
    return h

# ==================== ECE计算 ====================
def compute_ece(results, n_bins=10):
    labeled = [r for r in results if r.get("is_labeled") and r["expert_label"] is not None]
    if len(labeled) < 10:
        return None
    labeled_sorted = sorted(labeled, key=lambda x: x["h_conf"])
    bin_size = len(labeled_sorted) // n_bins
    if bin_size == 0:
        return None
    ece = 0.0
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(labeled_sorted)
        bin_items = labeled_sorted[start:end]
        avg_conf = sum(r["h_conf"] for r in bin_items) / len(bin_items)
        acc = sum(r["expert_label"] for r in bin_items) / len(bin_items)
        ece += (len(bin_items) / len(labeled_sorted)) * abs(avg_conf - acc)
    return round(ece, 4)

# ==================== 三档决策 ====================
def make_decision(conf):
    if conf >= 0.85:
        return "high"
    elif conf >= 0.70:
        return "medium"
    return "low"

# ==================== 加载数据 ====================
def load_data():
    rows = []
    # 源文件是GBK编码
    with open(DATA_FILE, "r", encoding="gbk") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                expert = int(row["expert_label"]) if row.get("expert_label", "").strip() else None
            except:
                expert = None
            rows.append({
                "row_index": int(row["row_index"]),
                "department": row.get("department", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "expert_label": expert,
                "is_labeled": expert is not None,
            })
    if MAX_SAMPLES is not None:
        rows = rows[:MAX_SAMPLES]
    log(f"数据加载: {len(rows)} 条")
    return rows

# ==================== 单条处理 ====================
def process_sample(item):
    row_idx = item["row_index"]
    question = item["question"][:200]
    answer = item["answer"][:500]
    prompt_text = PROMPT_P.format(question=question, answer=answer)  # L-Conf 也用 P-Conf 的 prompt
    messages = [{"role": "user", "content": prompt_text}]

    t_start = time.time()

    # 并发执行 P-Conf 和 SC-Conf
    with ThreadPoolExecutor(max_workers=2) as inner_ex:
        f_p = inner_ex.submit(compute_p_conf, question, answer)
        f_sc = inner_ex.submit(compute_sc_conf, question, answer)
        p_result = f_p.result()
        sc_result = f_sc.result()

    # L-Conf 单独执行
    l_result = compute_l_conf(messages)

    p = p_result["confidence"]
    sc = sc_result["confidence"]
    l = l_result["confidence"]
    h = compute_h_conf(p, sc, l)

    elapsed = time.time() - t_start

    return {
        "row_index": row_idx,
        "department": item["department"],
        "p_conf": p,
        "sc_conf": sc,
        "l_conf": l,
        "h_conf": h,
        "p_text": p_result.get("text", ""),
        "sc_decisions": sc_result.get("decisions", []),
        "sc_vote_detail": sc_result.get("vote_detail", ""),
        "elapsed": round(elapsed, 2),
        "expert_label": item["expert_label"],
        "is_labeled": item["is_labeled"],
        "decision": make_decision(h),
    }

# ==================== 主流程 ====================
def main():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    log("=" * 60)
    log("DataCleanAgent 10万条实验 v4（300条专家标注）")
    log(f"数据: {DATA_FILE}")
    log(f"输出: {OUTPUT_FILE}")
    log("=" * 60)

    data = load_data()

    # 断点续传
    done = set()
    results = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                results = json.load(f)
            done = {r["row_index"] for r in results}
            log(f"断点续传: {len(done)} 条已完成")
        except:
            results = []

    pending = [item for item in data if item["row_index"] not in done]
    total = len(pending)
    log(f"待处理: {total} 条")

    t_start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_sample, item): item for item in pending}
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                results.append(result)
                done.add(item["row_index"])

                if len(results) % 100 == 0:
                    ece = compute_ece(results)
                    ece_str = f" ECE={ece:.4f}" if ece is not None else ""
                    elapsed_total = time.time() - t_start
                    speed = len(results) / elapsed_total
                    eta = (total - len(results)) / speed / 60
                    log(f"[{len(results)}/{total}] p={result['p_conf']:.3f} sc={result['sc_conf']:.3f} l={result['l_conf']:.3f} h={result['h_conf']:.3f}{ece_str} | ETA~{eta:.0f}min")

                if len(results) % SAVE_EVERY == 0:
                    with open(OUTPUT_FILE, 'w') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)

            except Exception as e:
                log(f"处理失败 row={item['row_index']}: {e}")

    # 最终保存
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed_total = time.time() - t_start
    log(f"完成! {len(results)} 条 -> {OUTPUT_FILE}")
    log(f"总耗时: {elapsed_total:.1f}秒 ({len(results)/elapsed_total:.2f}条/秒)")

    # ECE报告
    ece = compute_ece(results)
    labeled_count = sum(1 for r in results if r.get("is_labeled"))
    log(f"\n=== ECE验证报告 ===")
    log(f"有标注数据: {labeled_count} 条")
    if ece is not None:
        log(f"ECE = {ece:.4f} (目标 < 0.10)")
    else:
        log("ECE无法计算（无标注数据）")

if __name__ == "__main__":
    main()
