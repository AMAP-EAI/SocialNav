# SocialNav: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation

This is the official repository for **SocialNav**, a foundational model for socially-aware embodied navigation with a hierarchical *brain–action* architecture. SocialNav unifies high-level social norm understanding with low-level, socially compliant trajectory generation.

> 📢 **Note**: The code and data are currently being cleaned up and documented.  
> We will release them **soon**. Please stay tuned!

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

## 📦 Code & Data Release Plan

We are currently preparing the following resources:

- [ ] **Training & inference code** for SocialNav  
- [ ] **SocNav Dataset tools and scripts**  
- [ ] **SocNav Benchmark environments & evaluation code**  
- [ ] **Pre-trained models** and demo scripts

All of the above will be released in this repository once internal checks and documentation are completed.

> ⏳ **Status:**  
> - Code: **Cleaning & refactoring**  
> - Dataset tools: **Packaging for release**  
> - Benchmark: **Preparing scenes & evaluation scripts**

If you are interested, please **watch** or **star** this repository to get notified when we push major updates.

---

## 📝 Citation

If you find this project useful in your research, please consider citing:

```bibtex
@inproceedings{SocialNav2024,
  title     = {SocialNav: A Foundational Model for Socially-Aware Navigation},
  author    = {TODO: Fill in author list},
  booktitle = {TODO: Conference / Journal},
  year      = {2024},
  url       = {https://github.com/AMAP-EAI/SocialNav}
}
