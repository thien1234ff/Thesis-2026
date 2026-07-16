import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Robustness data from Table 4.4 and Table 4.6 (TAR @ FAR=10^-4)
ROBUSTNESS_DATA = {
    "iresnet18": {
        "FP32": {"LFW": 99.133, "CFP-FP": 77.686, "AgeDB-30": 84.967, "CALFW": 85.000, "CPLFW": 50.733},
        "Q6_PTQ": {"LFW": 99.033, "CFP-FP": 71.429, "AgeDB-30": 84.000, "CALFW": 83.300, "CPLFW": 56.933},
        "Q6_wo_3p": {"LFW": 99.067, "CFP-FP": 73.429, "AgeDB-30": 76.033, "CALFW": 83.900, "CPLFW": 49.567},
        "Q6_w_3p": {"LFW": 99.233, "CFP-FP": 76.543, "AgeDB-30": 87.900, "CALFW": 87.267, "CPLFW": 53.100}
    },
    "iresnet50": {
        "FP32": {"LFW": 99.600, "CFP-FP": 88.914, "AgeDB-30": 92.900, "CALFW": 90.500, "CPLFW": 53.533},
        "Q6_PTQ": {"LFW": 98.500, "CFP-FP": 67.343, "AgeDB-30": 76.400, "CALFW": 82.833, "CPLFW": 0.100},
        "Q6_wo_3p": {"LFW": 99.433, "CFP-FP": 81.514, "AgeDB-30": 88.300, "CALFW": 90.433, "CPLFW": 56.233},
        "Q6_w_3p": {"LFW": 99.333, "CFP-FP": 85.657, "AgeDB-30": 92.700, "CALFW": 89.333, "CPLFW": 37.000}
    },
    "mobilefacenet": {
        "FP32": {"LFW": 98.167, "CFP-FP": 65.857, "AgeDB-30": 74.500, "CALFW": 84.700, "CPLFW": 9.600},
        "Q6_PTQ": {"LFW": 91.600, "CFP-FP": 25.286, "AgeDB-30": 24.600, "CALFW": 49.600, "CPLFW": 0.067},
        "Q6_wo_3p": {"LFW": 94.767, "CFP-FP": 31.800, "AgeDB-30": 31.800, "CALFW": 76.000, "CPLFW": 15.133},
        "Q6_w_3p": {"LFW": 95.900, "CFP-FP": 62.914, "AgeDB-30": 44.400, "CALFW": 73.200, "CPLFW": 4.033}
    }
}

def print_robustness_summary(arch):
    """Prints comparisons showing the robustness of 3-Phase alignment (w-3p vs wo-3p vs baseline)."""
    data = ROBUSTNESS_DATA[arch]
    
    print(f"\n==========================================================================================")
    print(f"  ROBUSTNESS EVALUATION (TAR @ FAR=10^-4) FOR {arch.upper()}")
    print(f"==========================================================================================")
    print("Dataset      |  FP32        |  Q6 PTQ      |  Ours (wo-3p) |  Ours (w-3p)")
    print("-------------+--------------+--------------+---------------+------------------------------")
    for ds in ["LFW", "CFP-FP", "AgeDB-30", "CALFW", "CPLFW"]:
        val_fp32 = data["FP32"][ds]
        val_ptq = data["Q6_PTQ"][ds]
        val_wo3p = data["Q6_wo_3p"][ds]
        val_w3p = data["Q6_w_3p"][ds]
        
        # Highlight cases where our w-3p model matches or outperforms FP32
        note = " * [Exceeds FP32]" if val_w3p >= val_fp32 else ""
        
        print(f"{ds:12s} |  {val_fp32:.3f}%     |  {val_ptq:.3f}%     |  {val_wo3p:.3f}%     |  {val_w3p:.3f}%{note}")
    print(f"==========================================================================================")

def generate_robustness_chart(arch, output_path="robustness_chart.png"):
    """Saves a comparison chart comparing FP32, Q6 PTQ, and proposed Q6 w-3p model."""
    data = ROBUSTNESS_DATA[arch]
    benchmarks = ["CFP-FP", "AgeDB-30", "CALFW", "CPLFW"]
    models = ["FP32", "Q6_PTQ", "Q6_w_3p"]
    
    x = np.arange(len(benchmarks))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, model in enumerate(models):
        tars = [data[model][bm] for bm in benchmarks]
        # Make ours look premium
        color = '#1f77b4' if model == "FP32" else ('#d62728' if model == "Q6_PTQ" else '#2ca02c')
        ax.bar(x + (i - 1) * width, tars, width, label=model.replace('_', ' '), color=color)
        
    ax.set_ylabel("TAR @ FAR=1e-4 (%)")
    ax.set_title(f"Robustness Benchmarking (TAR @ FAR=10^-4) on {arch.upper()}")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[INFO] Robustness comparison saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    import numpy as np
    
    for arch in ["iresnet18", "iresnet50", "mobilefacenet"]:
        print_robustness_summary(arch)
        try:
            generate_robustness_chart(arch, f"robustness_{arch}.png")
        except Exception as e:
            print(f"Could not generate plot for {arch}: {e}")
