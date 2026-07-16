# Dataset & Model Weights Layout Guide

This document describes how to organize verification datasets and pre-trained models within this project repository.

---

## 1. Directory Structure

For local evaluation and QAT training, organize datasets and weights under the following hierarchy (configured in `.gitignore` to prevent committing heavy binaries):

```text
Thesis-2026/
├── data/
│   ├── README.md               # (This file)
│   ├── lfw.bin                 # LFW verification binary
│   ├── cfp_fp.bin              # CFP-FP verification binary
│   ├── agedb_30.bin            # AgeDB-30 verification binary
│   ├── calfw.bin               # CALFW verification binary
│   └── cplfw.bin               # CPLFW verification binary
│
├── weights/
│   ├── FP32/
│   │   ├── iresnet18_fp32.pth  # FP32 iResNet-18 Teacher checkpoint
│   │   └── iresnet50_fp32.pth  # FP32 iResNet-50 Teacher checkpoint
│   │
│   ├── ptq/
│   │   ├── iresnet18_q6_ptq.pth # Q6 Post-Training Quantized model
│   │   └── iresnet50_q6_ptq.pth # Q6 Post-Training Quantized model
│   │
│   └── w-3p/
│       ├── iresnet18_q6-w3p.pth # Q6 Student with 3-Phase QAT + Beta-MSE
│       └── iresnet50_q6_w3p.pth # Q6 Student with 3-Phase QAT + Beta-MSE
```

---

## 2. Dataset Download Links

The validation datasets are provided in standard InsightFace binary formats (`.bin`):

*   **LFW, CFP-FP, AgeDB-30, CALFW, CPLFW Binaries**:
    *   Download from the official [InsightFace Dataset Hub](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo).
    *   Alternative Kaggle Mirror: [MS1M-RetinaFace Verification Binaries](https://www.kaggle.com/datasets/debarghamitraroy/msm1-retinaface-t1).

Place the downloaded `.bin` files directly in the `data/` folder.

---

## 3. Pre-trained Checkpoints

Download pre-trained weights for evaluation:

*   **iResNet-18 & iResNet-50 FP32 models**:
    *   Pretrained backbones on MS1MV2/MS1MV3 dataset can be found in the [InsightFace Model Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo).
    *   Download weights and save them to `weights/FP32/`.
