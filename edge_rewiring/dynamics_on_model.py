import xgi
import hypercontagion as hc
import matplotlib.pyplot as plt
import time
import numpy as np
import random
from model_generation import *

output_dir = r'experiment_result\dynamics_on_model\fig'

def dynamics_on_model(es, approx_num_C, num_max_hyperedge, num_node, min_size=2, max_size=None, adjust_es=False, compare_interval_smaller_case=2, compare_interval_bigger_case=2):
    # Generate the hypergraph
    H = model_generation_es(es, approx_num_C, num_max_hyperedge,num_node, min_size, max_size, adjust_es, compare_interval_smaller_case, compare_interval_bigger_case)
    initial_size = 100
    gamma = 0.05
    tau = {i: 0.1 for i in xgi.unique_edge_sizes(H)}
    start = time.time()
    t1, S1, I1, R1 = hc.discrete_SIR(H, tau, gamma, tmin=0, tmax=100, dt=1, rho=0.1)
    print(time.time() - start)
    
    plt.figure()
    plt.plot(t1, S1 / num_node, "g--", label="S (discrete)")
    plt.plot(t1, I1 / num_node, "r--", label="I (discrete)")
    plt.plot(t1, R1 / num_node, "b--", label="R (discrete)")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Fraction of population")
    plt.savefig(os.path.join(output_dir, f"{es}_SIR.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    es_list = np.linspace(0.15, 0.95, num=5)
    for es in es_list:
        dynamics_on_model(es, 9000, 300, 1000, 2, 11, True, 2, 2)