import xgi
import hypercontagion as hc
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import multiprocessing as mp
from model_generation import *
import dynamic_on_model as dom

def run_SIR_for_es(es):
    idx = int(es * 10 - 1)
    num_node = 1000
    num_max_hyperedge = 2000
    approx_num_C = 13857
    gamma = 0.05
    num_edges = idx * 1000 + 3000  # unique per es

    # Create an isolated figure for this process
    fig, ax = plt.subplots(figsize=(4, 3))

    dom.run_multiple_SIR_with_errorbands(
        es,
        approx_num_C,
        num_max_hyperedge,
        num_node,
        gamma,
        num_graphs=100,
        colors=["#00B388", "#DA291C", "#418FDF"],
        ax=ax
    )
    ax.set_title(f"es = {es}", fontsize=12)

    output_path = f"experiment_result/dynamics_on_model/single_plot_es_{es:.1f}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return (idx, output_path)

if __name__ == "__main__":
    os.makedirs("experiment_result/dynamics_on_model", exist_ok=True)

    es_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # Run in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(run_SIR_for_es, es_list)

    # Sort results by subplot index
    results.sort()

    # Combine all plots into a single figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    axes = axes.flatten()

    for idx, path in results:
        img = plt.imread(path)
        axes[idx].imshow(img)
        axes[idx].axis("off")
        axes[idx].set_title(f"es = {es_list[idx]:.1f}", fontsize=12)

    fig.suptitle(
        "SIR on Synthetic Data with Error Bands\nN=1000, Ẽ=2000, varied E",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    summary_path = (
        "experiment_result/dynamics_on_model/SIR_on_Synthetic_Data_Error_Bands.png"
    )
    fig.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.show()
