import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# verified raw database of ablation tests on Beta parameter
ABLATION_DATA = {
    "iresnet18": {
        "Accuracy": {
            1: {"LFW": 99.517, "CFP-FP": 92.486, "AgeDB-30": 96.717, "CALFW": 95.417, "CPLFW": 88.917},
            200: {"LFW": 99.467, "CFP-FP": 93.214, "AgeDB-30": 96.450, "CALFW": 95.483, "CPLFW": 88.983},
            300: {"LFW": 99.550, "CFP-FP": 92.857, "AgeDB-30": 96.783, "CALFW": 95.500, "CPLFW": 88.400}
        },
        "TAR": {
            1: {"LFW": 99.067, "CFP-FP": 71.086, "AgeDB-30": 77.300, "CALFW": 87.267, "CPLFW": 53.100},
            200: {"LFW": 99.033, "CFP-FP": 73.029, "AgeDB-30": 87.900, "CALFW": 84.933, "CPLFW": 25.700},
            300: {"LFW": 99.133, "CFP-FP": 76.543, "AgeDB-30": 85.633, "CALFW": 85.533, "CPLFW": 25.067}
        }
    },
    "iresnet50": {
        "Accuracy": {
            1: {"LFW": 99.733, "CFP-FP": 94.257, "AgeDB-30": 97.333, "CALFW": 95.833, "CPLFW": 90.433},
            200: {"LFW": 99.767, "CFP-FP": 95.043, "AgeDB-30": 97.683, "CALFW": 95.817, "CPLFW": 91.767},
            300: {"LFW": 99.750, "CFP-FP": 95.488, "AgeDB-30": 97.567, "CALFW": 95.767, "CPLFW": 91.550}
        },
        "TAR": {
            1: {"LFW": 98.933, "CFP-FP": 81.343, "AgeDB-30": 74.233, "CALFW": 88.367, "CPLFW": 36.000},
            200: {"LFW": 99.333, "CFP-FP": 81.943, "AgeDB-30": 92.700, "CALFW": 89.333, "CPLFW": 37.000},
            300: {"LFW": 99.333, "CFP-FP": 85.657, "AgeDB-30": 91.867, "CALFW": 88.333, "CPLFW": 31.000}
        }
    },
    "mobilefacenet": {
        "Accuracy": {
            1: {"LFW": 98.700, "CFP-FP": 88.257, "AgeDB-30": 92.567, "CALFW": 93.367, "CPLFW": 84.733},
            200: {"LFW": 99.183, "CFP-FP": 90.443, "AgeDB-30": 93.883, "CALFW": 94.033, "CPLFW": 86.450},
            300: {"LFW": 99.317, "CFP-FP": 90.514, "AgeDB-30": 93.867, "CALFW": 93.650, "CPLFW": 86.600}
        },
        "TAR": {
            1: {"LFW": 94.100, "CFP-FP": 40.629, "AgeDB-30": 31.267, "CALFW": 72.533, "CPLFW": 2.000},
            200: {"LFW": 94.400, "CFP-FP": 62.914, "AgeDB-30": 44.400, "CALFW": 73.200, "CPLFW": 1.400},
            300: {"LFW": 95.900, "CFP-FP": 56.857, "AgeDB-30": 40.100, "CALFW": 72.300, "CPLFW": 4.033}
        }
    }
}

def print_ablation_summary(arch):
    """Prints ablation study tables for accuracy and TAR at FAR=1e-4."""
    data = ABLATION_DATA[arch]
    
    print(f"\n========================================================================")
    print(f"  ABLATION STUDY FOR {arch.upper()}: ALIGNMENT COEFFICIENT BETA")
    print(f"========================================================================")
    
    # 1. Accuracy table
    print("--- 1. Verification Accuracy (%) ---")
    print("Beta  | LFW       | CFP-FP    | AgeDB-30  | CALFW     | CPLFW")
    print("------+-----------+-----------+-----------+-----------+-----------")
    for beta in [1, 200, 300]:
        accs = data["Accuracy"][beta]
        print(f"{beta:5d} | {accs['LFW']:.3f}%   | {accs['CFP-FP']:.3f}%   | {accs['AgeDB-30']:.3f}%   | {accs['CALFW']:.3f}%   | {accs['CPLFW']:.3f}%")
        
    print("\n--- 2. True Acceptance Rate (TAR @ FAR=10^-4) (%) ---")
    print("Beta  | LFW       | CFP-FP    | AgeDB-30  | CALFW     | CPLFW")
    print("------+-----------+-----------+-----------+-----------+-----------")
    for beta in [1, 200, 300]:
        tars = data["TAR"][beta]
        print(f"{beta:5d} | {tars['LFW']:.3f}%   | {tars['CFP-FP']:.3f}%   | {tars['AgeDB-30']:.3f}%   | {tars['CALFW']:.3f}%   | {tars['CPLFW']:.3f}%")
        
    print(f"========================================================================")

def generate_ablation_chart(arch, output_path="ablation_beta.png"):
    """Generates and saves a comparison bar chart illustrating the effect of Beta."""
    data = ABLATION_DATA[arch]["TAR"]
    betas = [1, 200, 300]
    benchmarks = ["CFP-FP", "AgeDB-30", "CALFW", "CPLFW"]
    
    x = np.arange(len(benchmarks))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, beta in enumerate(betas):
        tars = [data[beta][bm] for bm in benchmarks]
        rects = ax.bar(x + (i - 1) * width, tars, width, label=f"Beta = {beta}")
        
    ax.set_ylabel("TAR @ FAR=1e-4 (%)")
    ax.set_title(f"Ablation Study of Alignment Coefficient Beta on {arch.upper()}")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"📈 Ablation chart saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    import numpy as np
    
    for arch in ["iresnet18", "iresnet50", "mobilefacenet"]:
        print_ablation_summary(arch)
        try:
            generate_ablation_chart(arch, f"ablation_beta_{arch}.png")
        except Exception as e:
            print(f"Could not generate plot for {arch}: {e}")
