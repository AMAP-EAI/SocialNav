# [CVPR 2026] SocialNav: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation

<div align="center" style="margin: 1rem 0 1.1rem;">

<p align="center" style="margin: 0; font-size: clamp(1.2rem, 2.6vw, 1.65rem); font-weight: 700; letter-spacing: 0.06em; line-height: 1.35; color: #006BB6; font-family: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;">
  <strong>🎉🎉CVPR 2026🎉🎉</strong>
</p>

</div>

---

This is the official repository for **SocialNav**, a foundational model for socially-aware embodied navigation with a hierarchical *brain–action* architecture. SocialNav unifies high-level social norm understanding with low-level, socially compliant trajectory generation.

> 📢 **Note:** **Pre-trained models** (Hugging Face) and the **CityWalker benchmark evaluation script** are available in this repository.

---

## 🌐 Project Page

For an overview of the project, figures, and teaser video, please visit the project page:

👉 **Project Page:** https://amap-eai.github.io/SocialNav/

---

## 🔍 Overview

**SocialNav** is designed to address socially-aware navigation in real-world environments by:

- Combining a **VLM-based Brain** for high-level semantic and social reasoning
- With a **flow-based Action Expert** for low-level trajectory generation
- Training on the large-scale **SocNav Dataset** (7M samples) and evaluating on the **SocNav Benchmark**

Key components include:

- **SocNav Dataset**
  - **Expert Trajectories Pyramid (ETP)**
  - **Cognitive Activation Dataset (CAD)**

- **SocNav Benchmark**
  - High-fidelity evaluation built on Isaac Sim + 3DGS
  - 9 large-scale social scenes (parks, streets, offices, campus)
  - Metrics for both navigation performance and social compliance

---

## 🤗 Models

SAFE-GRPO checkpoints:

| Backbone | ModelScope | Hugging Face |
|----------|------------|--------------|
| **Qwen2-VL** | [SocialNav-Qwen2-VL-SAFE-GRPO](https://www.modelscope.cn/models/zjugyn/SocialNav-Qwen2-VL-SAFE-GRPO) | [SocialNav-Qwen2-VL-SAFE-GRPO](https://huggingface.co/zjuSekineko/SocialNav-Qwen2-VL-SAFE-GRPO) |
| **Qwen2.5-VL** | [SocialNav-Qwen2.5-VL-SAFE-GRPO](https://www.modelscope.cn/models/zjugyn/SocialNav-Qwen2.5-VL-SAFE-GRPO) | [SocialNav-Qwen2.5-VL-SAFE-GRPO](https://huggingface.co/zjuSekineko/SocialNav-Qwen2.5-VL-SAFE-GRPO) |
| **Qwen3-VL** | *Coming soon* | *Coming soon* |

---

## 📦 Installation

### Requirements

- **OS**: Linux (recommended) or macOS; GPU inference requires **NVIDIA CUDA**
- **Python**: 3.10 / 3.11 (match `requirements.txt`)
- **CUDA**: Version compatible with your PyTorch wheels (e.g. cu12)

### 1. Clone

```bash
git clone https://github.com/AMAP-EAI/SocialNav.git
cd SocialNav
```

### 2. Virtual environment (optional)

```bash
python -m venv .venv
source .venv/bin/activate 
```

### 3. PyTorch

Install from [pytorch.org](https://pytorch.org/get-started/locally/) for your CUDA version, e.g.:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 4. Dependencies

```bash
pip install -r requirements.txt
```

For flow-matching components, if missing from `requirements.txt`:

```bash
pip install torchcfm diffusers
```

### 5. Local Transformers (required)

This repo ships a patched **Qwen3-VL + Flow Matching** tree under `transformers/`. Install in editable mode:

```bash
pip install -e "./transformers[dev]"   # or: pip install -e ./transformers
```

### 6. `PYTHONPATH`

If `modeling_qwen3_vl.py` imports `src.train.sde_with_logprob`, run from the repo root or set:

```bash
export PYTHONPATH="/path/to/SocialNav:${PYTHONPATH}"
```

### 7. Flash Attention (optional)

Install `flash-attn-2` per Qwen3-VL docs if you need lower memory; otherwise PyTorch **SDPA** is fine.

---

## 📊 Evaluation

**Script:** `utils/citywalker.py`  
**Primary metric:** **`mean_angle`** in `metrics_citywalker_qwen3.csv`. Per sample, the script takes the **maximum** over five steps of the angle (degrees) between predicted and GT waypoint vectors; `mean_angle` in the CSV is the **mean** of that value over included samples (by row: categories, `overall`, and `mean`). Implementation: `compute_sample_metrics` and the `mean_angle` lists in `main`.

**Input** (jsonl, one record per line):

| Field | Description |
|-------|-------------|
| `images` | List of local image paths |
| `messages[0].content` | User text |
| `messages[1].gt_waypoints` | `(5, 2)` |
| `messages[1].input_waypoints` | `(6, 2)` |
| `messages[1].step_scale` | `float` |
| `messages[1].arrive` | `[0]` or `[1]` |
| `messages[1].categories` | Aligned with `TEST_CATEGORIES` in the script |

**Run:** Set `MODEL_PATH`, `DATA_PATH`, and `DEVICE` at the top of the script, then:

```bash
cd /path/to/SocialNav
export PYTHONPATH="$(pwd):${PYTHONPATH}"
CUDA_VISIBLE_DEVICES=0 python utils/citywalker.py
```

**Outputs:** `pred_citywalker_qwen3.jsonl`, `metrics_citywalker_qwen3.csv` (default under `MODEL_PATH/infer_result_citywalker_qwen3_fast_step_5/`; see `OUTPUT_DIR` in the script).

---

## 📝 Citation

If you find this project useful in your research, please consider citing (to appear at **CVPR 2026**):

```bibtex
@article{chen2025socialnav,
      title={SocialNav: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation},
      author={Ziyi Chen and Yingnan Guo and Zedong Chu and Minghua Luo and Yanfen Shen and Mingchao Sun and Junjun Hu and Shichao Xie and Kuan Yang and Pei Shi and Zhining Gu and Lu Liu and Honglin Han and Xiaolong Wu and Mu Xu and Yu Zhang},
      journal={arXiv preprint arXiv:2511.21135},
      year={2025}
}
```
