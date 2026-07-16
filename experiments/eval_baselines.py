import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import iresnet18, iresnet50, mobilefacenet
from src.quantizer import quantize_model
from src.evaluation import load_bin, extract_embeddings, evaluate_10fold, tar_at_far

# --------------------------------------------------
# Verified Results from Thesis Report (Table 4.1)
# --------------------------------------------------
THESIS_TABLE = {
    "iresnet18": {
        "FP32": {
            "Accuracy": {"LFW": 99.617, "CFP-FP": 93.671, "AgeDB-30": 96.687, "CALFW": 95.533, "CPLFW": 89.183},
            "TAR@1e-4": {"LFW": 99.133, "CFP-FP": 77.686, "AgeDB-30": 84.967, "CALFW": 85.000, "CPLFW": 50.733}
        },
        "Q6_PTQ": {
            "Accuracy": {"LFW": 99.500, "CFP-FP": 92.671, "AgeDB-30": 96.633, "CALFW": 95.283, "CPLFW": 87.733},
            "TAR@1e-4": {"LFW": 99.033, "CFP-FP": 71.429, "AgeDB-30": 84.000, "CALFW": 83.300, "CPLFW": 56.933}
        }
    },
    "iresnet50": {
        "FP32": {
            "Accuracy": {"LFW": 99.800, "CFP-FP": 95.957, "AgeDB-30": 97.983, "CALFW": 96.083, "CPLFW": 92.217},
            "TAR@1e-4": {"LFW": 99.600, "CFP-FP": 88.914, "AgeDB-30": 92.900, "CALFW": 90.500, "CPLFW": 53.533}
        },
        "Q6_PTQ": {
            "Accuracy": {"LFW": 99.683, "CFP-FP": 91.557, "AgeDB-30": 96.083, "CALFW": 95.133, "CPLFW": 87.017},
            "TAR@1e-4": {"LFW": 98.500, "CFP-FP": 67.343, "AgeDB-30": 76.400, "CALFW": 82.833, "CPLFW": 0.100}
        }
    },
    "mobilefacenet": {
        "FP32": {
            "Accuracy": {"LFW": 99.433, "CFP-FP": 91.529, "AgeDB-30": 95.567, "CALFW": 95.150, "CPLFW": 87.800},
            "TAR@1e-4": {"LFW": 98.167, "CFP-FP": 65.857, "AgeDB-30": 74.500, "CALFW": 84.700, "CPLFW": 9.600}
        },
        "Q6_PTQ": {
            "Accuracy": {"LFW": 98.150, "CFP-FP": 83.400, "AgeDB-30": 89.267, "CALFW": 91.067, "CPLFW": 77.017},
            "TAR@1e-4": {"LFW": 91.600, "CFP-FP": 25.286, "AgeDB-30": 24.600, "CALFW": 49.600, "CPLFW": 0.067}
        }
    }
}

def print_table_from_data(arch):
    """Prints the baseline comparison table from thesis data."""
    data = THESIS_TABLE[arch]
    print(f"\n==================================================================================")
    print(f"   BASELINE PERFORMANCE FOR {arch.upper()} (FROM THESIS REPORT)")
    print(f"==================================================================================")
    print(f"Dataset      |  Accuracy (%)                   |  TAR @ FAR=10^-4 (%)")
    print(f"             |  FP32        |  Q6 PTQ          |  FP32        |  Q6 PTQ")
    print(f"-------------+--------------+------------------+--------------+-------------------")
    for ds in ["LFW", "CFP-FP", "AgeDB-30", "CALFW", "CPLFW"]:
        acc_fp32 = data["FP32"]["Accuracy"][ds]
        acc_q6 = data["Q6_PTQ"]["Accuracy"][ds]
        tar_fp32 = data["FP32"]["TAR@1e-4"][ds]
        tar_q6 = data["Q6_PTQ"]["TAR@1e-4"][ds]
        print(f"{ds:12s} |  {acc_fp32:.3f}%     |  {acc_q6:.3f}%          |  {tar_fp32:.3f}%     |  {tar_q6:.3f}%")
    print(f"==================================================================================")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths configuration
    data_dir = "data"
    weights_dir = "weights"
    
    benchmarks = ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"]
    missing_data = False
    
    for bm in benchmarks:
        if not os.path.exists(os.path.join(data_dir, f"{bm}.bin")):
            missing_data = True
            break
            
    if missing_data:
        print("[INFO] InsightFace validation binaries not found in 'data/' directory.")
        print("[INFO] Displaying verified baseline experimental results from the Thesis Report (Table 4.1).")
        print("To run a live evaluation, please refer to 'data/README.md' to download the verification bin files.")
        
        for arch in ["iresnet18", "iresnet50", "mobilefacenet"]:
            print_table_from_data(arch)
        return

    # If data is present, run actual evaluation on PyTorch
    print("[INFO] Validation datasets detected! Running live evaluation...")
    archs = {
        "iresnet18": iresnet18,
        "iresnet50": iresnet50,
        "mobilefacenet": mobilefacenet
    }
    
    for name, creator in archs.items():
        print(f"\nEvaluating Backbone: {name}")
        
        # Paths for model checkpoints
        fp32_path = os.path.join(weights_dir, "FP32", f"{name}_fp32.pth")
        ptq_path = os.path.join(weights_dir, "ptq", f"{name}_q6_ptq.pth")
        
        # 1. FP32 Model Eval
        if os.path.exists(fp32_path):
            print(f"--- FP32 Model ({fp32_path}) ---")
            model = creator(num_features=512 if "mobile" not in name else 128)
            model.load_state_dict(torch.load(fp32_path, map_location=device), strict=False)
            model.to(device)
            model.eval()
            
            for bm in benchmarks:
                images, issame = load_bin(os.path.join(data_dir, f"{bm}.bin"))
                emb = extract_embeddings(model, images, device=device)
                acc_mean, acc_std = evaluate_10fold(emb, issame)
                tar = tar_at_far(emb, issame, target_far=1e-4)
                print(f"  {bm.upper():8s} | Acc: {acc_mean*100:.3f}% | TAR@1e-4: {tar*100:.3f}%")
        else:
            print(f"⚠️ FP32 weight file {fp32_path} not found.")

        # 2. Q6 PTQ Model Eval
        if os.path.exists(ptq_path):
            print(f"--- Q6 PTQ Model ({ptq_path}) ---")
            model_fp32 = creator(num_features=512 if "mobile" not in name else 128)
            model = quantize_model(model_fp32, weight_bit=6, act_bit=6)
            model.load_state_dict(torch.load(ptq_path, map_location=device), strict=False)
            model.to(device)
            model.eval()
            
            for bm in benchmarks:
                images, issame = load_bin(os.path.join(data_dir, f"{bm}.bin"))
                emb = extract_embeddings(model, images, device=device)
                acc_mean, acc_std = evaluate_10fold(emb, issame)
                tar = tar_at_far(emb, issame, target_far=1e-4)
                print(f"  {bm.upper():8s} | Acc: {acc_mean*100:.3f}% | TAR@1e-4: {tar*100:.3f}%")
        else:
            print(f"⚠️ Q6 PTQ weight file {ptq_path} not found.")

if __name__ == "__main__":
    main()
