import xgi
import hypercontagion as hc
import matplotlib.pyplot as plt
import time
import numpy as np
import random
from model_generation import *
import dynamic_on_model as  dom
    
if __name__ == "__main__":
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    axes = axes.flatten()
    num_max_hyperedge = 2000
    num_node = 1000
    num_edges = 3000
    approx_num_C = 13857  # Set high to allow target_num_edges to work
    gamma = 0.05
    '''sum = 0
    
    for es in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]: 
        sum  += (num_edges - num_max_hyperedge + es * num_max_hyperedge) / es 
        
    approx_num_C = int(sum / 8) '''
    
    for es in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:      
        idx = int(es * 10 - 1)  
        num_edges = idx * 1000
        dom.run_multiple_SIR_with_errorbands(
            es,  
            approx_num_C, 
            num_max_hyperedge, 
            num_node,
            gamma, 
            num_graphs=10,
            colors = ["#00B388","#DA291C", "#418FDF"],
            ax = axes[idx])
        axes[idx].set_title(f"es = {es}", fontsize=12)
    
    fig.suptitle(
    f"SIR on Synthetic Data with Error Bands with E={num_edges}, N={num_node}, Ẽ={num_max_hyperedge}",
    fontsize=16)



    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Save the figure
    fig.savefig(
    f"experiment_result/dynamics_on_model/SIR on Synthetic Data with Error Bands with E={num_edges}, N={num_node}, Ẽ={num_max_hyperedge}.png",
    dpi=300,
    bbox_inches='tight')