import xgi
import hypercontagion as hc
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import multiprocessing as mp
from model_generation import *
import dynamic_on_model as dom
import datetime
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import tempfile

def run_SIR_for_es(es):
    idx = int(es/2*10-1)  # Convert es to an index for subplot
    num_node = 1000
    num_max_hyperedge = 2000
    approx_num_C = 8000
    gamma = 0.05
    num_edges = 4000 + (int(es * 10 - 1) - 1) * 1000
    #approx_num_C = (num_edges - num_max_hyperedge + es * num_max_hyperedge)/es

    fig, ax = plt.subplots(figsize=(4, 3))
    dom.run_multiple_SIR_with_errorbands(
        es,
        approx_num_C,
        num_max_hyperedge,
        num_node,
        gamma,
        num_graphs=10,
        colors=["#00B388", "#DA291C", "#418FDF"],
        ax=ax
    )
    ax.set_title(f"es = {es}", fontsize=12)

    # Save to a temporary file
    tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp_file.name, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return (idx, tmp_file.name)



if __name__ == "__main__":
    
    start = time.time()
    
    os.makedirs("experiment_result/dynamics_on_model", exist_ok=True)

    es_list = [0.2, 0.4, 0.6, 0.8]

    # Run in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(run_SIR_for_es, es_list)

    # Sort results by subplot index
    results.sort()

    # Combine all plots into a single figurett
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    axes = axes.flatten()

    for idx, path in results:
        img = plt.imread(path)
        axes[idx].imshow(img)
        axes[idx].axis("off")




    fig.suptitle(
        f"SIR on Synthetic Data with Error Bands\nN=1000, Ẽ=2000, Approx_Num_C = 12000, varied E",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    summary_path = f"experiment_result/dynamics_on_model/SIR_on_Synthetic_Data_Error_Bands_{run_id}.png"

    fig.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")