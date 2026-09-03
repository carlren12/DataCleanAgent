# DataCleanAgent: Uncertainty-Aware Agent for LLM Data Cleaning

Official implementation of **DataCleanAgent**, an uncertainty-aware agent framework for data cleaning with Large Language Models (LLMs).

This repository contains the code required to reproduce the confidence estimation, calibration, and three-tier routing experiments reported in the paper.

---

## Overview

Real-world training data in high-stakes domains (healthcare, education, finance) contains **semantic noise** — factual errors, logical contradictions, semantic conflicts, and incompleteness — that rule-based and statistical cleaners cannot detect. LLMs can identify such noise, but they are poorly calibrated: their expressed confidence often diverges systematically from actual accuracy.

DataCleanAgent addresses this with three components:

1. **Four complementary confidence estimation strategies**
   - `P-Conf` (Prompt-based): elicits a 0–1 confidence via a single prompt (temperature = 0.1).
   - `SC-Conf` (Self-Consistency): N = 3 samplings (temperature = 0.9), confidence = agreement rate ∈ {1/3, 2/3, 1}.
   - `L-Conf` (Logit-based): derived from token logits via softmax over {YES, NO}.
   - `H-Conf` (Hybrid): weighted fusion, `H = 0.4·P + 0.4·SC + 0.2·L`.

2. **Calibration** — Expected Calibration Error (ECE, M = 10 bins) with temperature scaling:
   ```
   c_cal = sigmoid( logit(c) / T )
   ```
   Optimal temperatures: P-Conf `T = 2.55`, L-Conf `T = 4.25`, H-Conf `T = 1.95` (SC-Conf is not calibratable).

3. **Three-tier threshold routing** — `θ_H = 0.85`, `θ_L = 0.70`:
   - `HIGH` (≥ 0.85) → automatic cleaning
   - `MEDIUM` (0.70 – 0.85) → human review
   - `LOW` (< 0.70) → reject automatic processing

---

## Repository Structure

| File | Purpose |
|---|---|
| `confidence_strategies.py` | Implementation of the four confidence estimation strategies (P-Conf, SC-Conf, L-Conf, H-Conf). |
| `temperature_scaling_v2.py` | ECE computation and temperature-scaling calibration. Reproduces Table II of the paper. |
| `run_100k_v4_300anno.py` | Main experiment driver for the 100,000-sample run. Reproduces Table III and the three-tier distribution. |
| `build_all_datasets_v2.py` | Stratified sampling (seed = 42) used to construct the experimental 100K subset. |
| `run_1k_concurrent.py` | Concurrent GPU inference template (ThreadPoolExecutor, ~7.5 samples/s). |
| `config.py` | Central configuration; reads API credentials from environment variables. |
| `data_loader.py` | Data loading utilities. |

---

## Requirements

- Python 3.10+ (developed on Python 3.10)
- An OpenAI-compatible inference endpoint. The experiments use **vLLM** serving `Qwen2.5-7B-Instruct`.
- GPU: 8 × RTX 5090 was used for the reported results.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

**Credentials are never hardcoded.** `config.py` reads all API credentials from environment variables, so it is safe to commit. Set the variables below before running:

```bash
export LOCAL_VLLM_URL="http://<your-gpu-server>:8002/v1"
export LOCAL_VLLM_API_KEY="local-dev"
export CLAW_API_KEY="your-claw-api-key"      # optional, SJTU CLAW API
export ZHIPU_API_KEY="your-zhipu-api-key"    # optional, Zhipu GLM
```

---

## Reproducing the Paper Results

### Step 1 — Temperature scaling calibration (Table II)

```bash
python temperature_scaling_v2.py
```

Expected output (300 expert-annotated samples: 109 noisy + 191 clean, accuracy 63.7%):

| Strategy | Raw ECE | Calibrated ECE | Improvement | Optimal T |
|---|---|---|---|---|
| P-Conf | 0.4800 | 0.3421 | −28.7% | T = 2.55 |
| SC-Conf | 0.5633 | — | — | N/A (MED = 0 at any T) |
| L-Conf | 0.4135 | 0.2775 | −32.9% | T = 4.25 |
| **H-Conf** | **0.4692** | **0.3436** | **−26.7%** | **T = 1.95** |

### Step 2 — Three-tier routing distribution (Table III, 100K samples)

```bash
python run_100k_v4_300anno.py
```

Expected output:

| Strategy | HIGH (≥0.85) | MEDIUM (0.70–0.85) | LOW (<0.70) | Mean |
|---|---|---|---|---|
| P-Conf | 22,291 (22.3%) | 77,628 (77.6%) | 81 (0.1%) | 0.839 |
| SC-Conf | 75,417 (75.4%) | 0 (0.0%) | 24,583 (24.6%) | 0.918 |
| L-Conf | 33,233 (33.2%) | 11,123 (11.1%) | 55,644 (55.6%) | 0.594 |
| **H-Conf** | **40,996 (41.0%)** | **48,583 (48.6%)** | **10,421 (10.4%)** | **0.822** |

### Step 3 — Dataset construction (optional)

```bash
python build_all_datasets_v2.py
```

Stratified sampling with `seed = 42` yields a 100,000-sample subset from MedDial-79W, preserving the six-department distribution (Internal Medicine 27.9%, OB-GYN 23.2%, Surgery 14.6%, Pediatric 12.8%, Andriatria 12.0%, Oncology 9.5%).

---

## Data Availability

The underlying corpus, **MedDial-79W**, contains 792,099 Chinese doctor–patient dialogue records. The full raw corpus is **not publicly released** due to patient-privacy regulations and institutional data-use agreements. A de-identified sample is available from the corresponding author upon reasonable request.

The **MedCleanBench** taxonomy (4 categories × 16 subtypes of Chinese semantic noise) is fully specified in the paper, so the benchmark can be reconstructed for any domain-specific corpus.

---

## Citation

If you use this code, please cite:

```bibtex
@article{ren2026datacleanagent,
  title   = {DataCleanAgent: An Uncertainty-Aware Agent for LLM Data Cleaning},
  author  = {Ren, Hao and Kong, Linghe and Zheng, Guanjie},
  journal = {Applied Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Experiments were conducted on a GPU server equipped with 8 × RTX 5090 at Shanghai Jiao Tong University.
