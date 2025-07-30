import matplotlib.pyplot as plt
import numpy as np

# stage indices for bars
stages = np.arange(1, 6)
bar_w = 0.3

# Data (same as before)
metrics = {
    "iCaRL": {
        "jac": [0.0001, 0.0013, 0.0052, 0.0056, 0.0257],
        "hes": [0.1278, 1.1047, 4.5420, 2.5230, 4.9603],
    },
    "GDumb": {
        "jac": [0.0000, 0.0004, 0.0007, 0.0008, 0.0009],
        "hes": [0.0064, 0.3499, 0.5779, 0.8206, 0.8231],
    },
    "ER-ACE": {
        "jac": [0.0000, 0.0003, 0.0010, 0.0017, 0.0019],
        "hes": [0.0266, 0.5405, 1.2031, 2.8512, 1.5224],
    },
    "ER-AML": {
        "jac": [0.0000, 0.0006, 0.0013, 0.0012, 0.0022],
        "hes": [0.0266, 0.7296, 0.7004, 0.6247, 1.0232],
    },
}

for title, vals in metrics.items():
    plt.figure(figsize=(4.125, 3))
    
    # left bars: Jacobian
    ax = plt.gca()
    ax.bar(stages - bar_w/2, vals["jac"], width=bar_w, label="Lipschitz constant $L_t$", color='tab:blue', alpha=0.9)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Lipschitz constant $L_t$")
    
    # right axis bars: Hessian 
    ax2 = ax.twinx()
    ax2.bar(stages + bar_w/2, vals["hes"], width=bar_w, label="Hessian $λ^{max}_t$", color='tab:red', alpha=0.5)
    ax2.set_ylabel("Hessian $λ^{max}_t$")
    
    # plt.title(title)
    
    # combine legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="upper left")
    
    ax.set_xticks(stages)
    ax.set_xticklabels(['1', '2', '3', '4', '5'])
    
    
    plt.tight_layout()
    save_path = f"./results/hessian_mnist_{title.lower()}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved at {save_path}")
