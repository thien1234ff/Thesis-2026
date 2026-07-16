# Gradient Coordination and Data Refinement for 6-Bit Face Recognition Optimization
*(Nghiên cứu Kỹ thuật Điều phối Gradient và Tinh lọc Dữ liệu nhằm Tối ưu hóa Mô hình Nhận dạng Khuôn mặt 6-bit)*

Vietnam National University, Hue University of Science (Thesis 2026).  
Author: **Hoàng Kim Thiên**  
Supervisor: **Dr. Lê Quang Chiến**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.9+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.10+-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 1. Project Overview & Methodology

This project introduces a highly efficient **Quantization-Aware Training (QAT)** framework to compress deep face recognition models down to **6-bit (Q6)** representation for both weights and activations. This achieves a theoretical **5.33x storage reduction** and significantly reduces memory bandwidth requirements, preparing models for resource-constrained Edge AI deployment.

To address embedding distortion and True Acceptance Rate (TAR) sags under strict False Acceptance Rate (FAR) constraints (e.g. $TAR@FAR=10^{-4}$), this research proposes two key innovations:
1. **$\beta$-MSE Gradient Coordination**: Amplifies backpropagation alignment signals on the $L_2$-normalized unit hypersphere.
2. **3-Phase Optimization Schedule**: Coordinates dynamic quantization ranges through progressive Warm-up, Observer Freezing, and Cool-down Fine-tuning.
3. **"Quality over Quantity" Data Refinement**: Utilizes a highly diverse real-world dataset of **~53,500 images** (incorporating pose and age variations) to replace massive synthetic datasets (e.g. 500k images in SOTA QuantFace), resulting in **57.4x faster convergence** (only 3,135 normalized iterations).

---

## 📐 2. System Architecture & Pipeline

The pipeline implements a Teacher-Student distillation framework where the student model is constrained by simulated 6-bit quantizers while guided by a full-precision (FP32) teacher:

```mermaid
graph TD
    Data[Refined Real Dataset: CASIA-1k, Pose, Age] --> |Batch Inputs| Teacher[Teacher Backbone: FP32]
    Data --> |Batch Inputs| Student[Student Backbone: Q6 QAT]
    
    subgraph Student Q6 Quantization Block
        Student --> Obs[1. Dynamic Range Observers]
        Obs --> Q[2. Simulated 6-Bit Quantizer]
        Q --> DQ[3. De-quantization]
    end

    Teacher --> |Extract Embeddings| FT[Teacher Feature f_t]
    DQ --> |Extract Embeddings| FS[Student Feature f_s]
    
    FT --> |L2 Normalization| NT[Normalized f_t]
    FS --> |L2 Normalization| NS[Normalized f_s]
    
    NS & NT --> Loss[Beta-MSE Gradient Alignment Loss]
    Loss --> |Coordinated Backpropagation| Student
    
    subgraph 3-Phase Scheduler
        P1[Phase 1: Warm-up Epochs 0-4 | Dynamic range estimation] --> P2[Phase 2: Freeze Observers Epochs 5-24 | Fixed quantization scales]
        P2 --> P3[Phase 3: Cool-down Epochs 25-29 | Cosine Learning Rate decay]
    end
```

---

## ⚡ 3. Key Core Features

- **Decoupled Modular Architecture (`src/`)**: Independent components for models (`models.py`), quantizers (`quantizer.py`), loss functions (`losses.py`), and evaluation tools (`evaluation.py`).
- **Mathematical Gradient Coordination**: Uses $\beta$-Weighted Mean Squared Error ($\beta$-MSE) loss:
  $$L_{GC} = \beta \cdot \frac{1}{N} \sum_{i=1}^{N} \|\tilde{s}_i - \tilde{t}_i\|^2_2$$
  where $\tilde{s}_i$ and $\tilde{t}_i$ represent $L_2$-normalized embeddings. The $\beta$ factor (optimal at 200–300) coordinates updates to overcome quantization noise.
- **Robustness in Fallback Mode (Web UI)**: The interactive Streamlit Web UI automatically detects missing local weights and falls back to a lightweight SSD face crop + OpenCV HSV Color Histogram correlation metric, preventing system crash on recruiter machines.

---

## 📊 4. Replicated Experimental Results

Here are the key baseline performance, ablation, and convergence speedup tables reproduced exactly from the thesis report:

### Table A: Baseline Performance (FP32 vs Q6 PTQ) - Table 4.1
*Measures verification accuracy and TAR@FAR=10^-4 on five benchmarks. Note the severe sags in PTQ (e.g. iResNet-50 CPLFW falls to 0.100%).*

| Architecture | Metric | LFW | CFP-FP | AgeDB-30 | CALFW | CPLFW |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **iResNet-18** | FP32 Accuracy | 99.617% | 93.671% | 96.687% | 95.533% | 89.183% |
| | Q6 PTQ Accuracy | 99.500% | 92.671% | 96.633% | 95.283% | 87.733% |
| | FP32 TAR@1e-4 | 99.133% | 77.686% | 84.967% | 85.000% | 50.733% |
| | Q6 PTQ TAR@1e-4 | 99.033% | 71.429% | 84.000% | 83.300% | 56.933% |
| **iResNet-50** | FP32 Accuracy | 99.800% | 95.957% | 97.983% | 96.083% | 92.217% |
| | Q6 PTQ Accuracy | 99.683% | 91.557% | 96.083% | 95.133% | 87.017% |
| | FP32 TAR@1e-4 | 99.600% | 88.914% | 92.900% | 90.500% | 53.533% |
| | Q6 PTQ TAR@1e-4 | 98.500% | 67.343% | 76.400% | 82.833% | 0.100% |

### Table B: Robustness and Recovery Rate Comparison - Table 4.4 & 4.6
*Comparing standard FP32 vs post-training Q6 (PTQ) vs Q6 without 3-phase (wo-3p) vs proposed Q6 with 3-phase (w-3p).*

| Architecture | Model | LFW | CFP-FP | AgeDB-30 | CALFW | CPLFW |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **iResNet-18** | FP32 | 99.133% | 77.686% | 84.967% | 85.000% | 50.733% |
| *(TAR@1e-4)* | Q6 PTQ | 99.033% | 71.429% | 84.000% | 83.300% | **56.933%** |
| | Ours (wo-3p) | 99.067% | 73.429% | 76.033% | 83.900% | 49.567% |
| | **Ours (w-3p)** | **99.233%** | **76.543%** | **87.900%** | **87.267%** | 53.100% |
| **iResNet-50** | FP32 | **99.600%** | **88.914%** | **92.900%** | **90.500%** | 53.533% |
| *(TAR@1e-4)* | Q6 PTQ | 98.500% | 67.343% | 76.400% | 82.833% | 0.100% |
| | **Ours (wo-3p)** | 99.433% | 81.514% | 88.300% | **90.433%** | **56.233%** |
| | Ours (w-3p) | 99.333% | **85.657%** | **92.700%** | 89.333% | 37.000% |

### Table C: Convergence Speed & Training Complexity (Ours vs SOTA) - Table 4.9
*Normalization compares training iteration counts required to reach convergence at batch size 512.*

| Metric / Setting | QuantFace (SOTA Baseline [2]) | Ours (Proposed Framework) |
| :--- | :---: | :---: |
| Dataset Size | 500,000 synthetic images | **53,458 refined real images** |
| Epochs | 60 | 30 |
| Total Iterations (BS=512) | 180,000 | **3,135** |
| **⚡ Acceleration Factor** | **1.0x (Baseline)** | **57.4x FASTER CONVERGENCE** |

---

## 🧠 5. Deep Scientific/Technical Insights

1. **Quantization as a Regularizer**:
   Our Q6 model with 3-phase training achieves **87.900% TAR@FAR=10^-4** on the AgeDB-30 benchmark, exceeding the teacher FP32 model (84.967%), representing a **~102% recovery rate**. Grad-CAM analysis demonstrates that 6-bit quantization narrows focus away from background noise and clothing context, acting as a strong regularizer that guides attention toward core facial details (eyes, nose, mouth).
2. **Gradient Coordination Dynamics**:
   Standard Mean Squared Error ($MSE$) gradients on normalized embeddings ($L_2$) are extremely small. In low-bit quantization, this prevents the student from correcting rounding errors. Scaling the loss by $\beta \in [200, 300]$ coordinate updates, pulling parameters back to the teacher's manifold.
3. **The Importance of Observer Freezing**:
   If dynamic observers remain active during later epochs, quantization scales fluctuation adds noise. Freezing them in Phase 2 stabilizes scales, allowing Phase 3 cosine decay to fine-tune weights around discrete thresholds.

---

## 🛠️ 6. Installation & Reproduction

### Prerequisites
- Python 3.8+
- PyTorch (with CUDA support recommended)

```bash
# Clone the repository
git clone https://github.com/thien1234ff/Thesis-2026.git
cd Thesis-2026

# Install dependencies
pip install -r requirements.txt
```

### Reproducing Experiments
Download datasets and weights as guided in [data/README.md](data/README.md) first. If datasets are missing, the scripts will gracefully output verified thesis results.

```bash
# 1. Run baseline comparison report (Table 4.1)
python experiments/eval_baselines.py

# 2. Run Beta parameter ablation report (Table 4.2 / 4.3)
python experiments/ablation_beta.py

# 3. Run robustness report (Table 4.4 / 4.6)
python experiments/eval_robustness.py

# 4. Run convergence comparison (Table 4.9)
python experiments/convergence_analysis.py

# 5. Run dry-run QAT training verification (generates mock data automatically)
python experiments/train_q6_w3p.py --dry-run
```

### Launching Streamlit Web App
Launch the interactive web UI dashboard to run local face verification (supporting image uploads and fallback visual matching):

```bash
streamlit run app.py
```
---

## 📝 Citation
```bibtex
@thesis{hkthien_thesis2026,
  author    = {Hoàng Kim Thiên},
  title     = {Nghiên cứu Kỹ thuật Điều phối Gradient và Tinh lọc Dữ liệu nhằm Tối ưu hóa Mô hình Nhận dạng Khuôn mặt 6-bit},
  school    = {Hue University of Science},
  year      = {2026},
  type      = {Bachelor's Thesis}
}
```
