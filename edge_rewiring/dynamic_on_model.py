import xgi
import hypercontagion as hc
import matplotlib.pyplot as plt
import time
import numpy as np
import random
from model_generation import *

def run_multiple_SIR_with_errorbands(
    es=0.05, 
    num_graphs=100, 
    approx_num_C=4253, 
    num_max_hyperedge=292, 
    num_node=516,
    gamma=0.05,
    rho=0.1,
    tmin=0,
    tmax=100,
    dt=1
):
    S_all, I_all, R_all = [], [], []
    t_vals = None  # to store time vector from first run

    for i in range(num_graphs):
        print(f"Simulation {i+1}/{num_graphs}")
        H = model_generation_es(
            es=es, 
            approx_num_C=approx_num_C, 
            num_max_hyperedge=num_max_hyperedge,
            num_node=num_node, 
            min_size=2, 
            max_size=None, 
            adjust_es=True
        )

        es = edit_simpliciality(H)
        print(f"es = {es}")

        mean_degree = sum(dict(H.degree()).values()) / H.num_nodes
        print(mean_degree)
        # mean_edge_size = sum(len(e) for e in H.edges) / H.num_edges
        # print(f"Mean edge size: {mean_edge_size:.4f}")

        tau = {k: 0.1 for k in xgi.unique_edge_sizes(H)}
        t, S, I, R = hc.discrete_SIR(H, tau, gamma=gamma, rho=rho, tmin=tmin, tmax=tmax, dt=dt)

        if t_vals is None:
            t_vals = t  # save the time vector
        
        print(f"Run {i+1}: Length of I = {len(I)}, S = {len(S)}, R = {len(R)}")

        min_len = min(len(S), len(I), len(R))
        S_all.append(S[:min_len] / num_node)
        I_all.append(I[:min_len] / num_node)
        R_all.append(R[:min_len] / num_node)

        if t_vals is None:
            t_vals = t[:min_len]

    print(f"Run {i+1}: Length of I = {len(I)}, S = {len(S)}, R = {len(R)}")

        # After the for loop:
    min_len = min(len(arr) for arr in S_all)  # Find the minimum time series length

    # Trim all arrays to min_len
    S_all = np.array([s[:min_len] for s in S_all])
    I_all = np.array([i[:min_len] for i in I_all])
    R_all = np.array([r[:min_len] for r in R_all])
    t_vals = t_vals[:min_len]

    # Compute mean and std
    S_mean, S_std = np.mean(S_all, axis=0), np.std(S_all, axis=0)
    I_mean, I_std = np.mean(I_all, axis=0), np.std(I_all, axis=0)
    R_mean, R_std = np.mean(R_all, axis=0), np.std(R_all, axis=0)


    # Plotting
    plt.figure(figsize=(10, 6))

    # Susceptible
    plt.plot(t_vals, S_mean, color="green", label="S (mean)")
    plt.fill_between(t_vals, S_mean - S_std, S_mean + S_std, color="green", alpha=0.3)

    # Infected
    plt.plot(t_vals, I_mean, color="red", label="I (mean)")
    plt.fill_between(t_vals, I_mean - I_std, I_mean + I_std, color="red", alpha=0.3)

    # Recovered
    plt.plot(t_vals, R_mean, color="blue", label="R (mean)")
    plt.fill_between(t_vals, R_mean - R_std, R_mean + R_std, color="blue", alpha=0.3)

    plt.xlabel("Time")
    plt.ylabel("Fraction of Population")
    plt.title(f"SIR Dynamics with Error Bands (es={es})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    run_multiple_SIR_with_errorbands()

# output_dir = r'experiment_result\dynamics_on_model\fig'

# def dynamics_on_model(es, approx_num_C, num_max_hyperedge, num_node, min_size=2, max_size=None, adjust_es=False, compare_interval_smaller_case=2, compare_interval_bigger_case=2):
#     # Generate the hypergraph
#     H = model_generation_es(es, approx_num_C, num_max_hyperedge,num_node, min_size, max_size, adjust_es, compare_interval_smaller_case, compare_interval_bigger_case)
#     initial_size = 100
#     gamma = 0.05
#     tau = {i: 0.1 for i in xgi.unique_edge_sizes(H)}
#     start = time.time()
#     t1, S1, I1, R1 = hc.discrete_SIR(H, tau, gamma, tmin=0, tmax=100, dt=1, rho=0.1)
#     print(time.time() - start)
    
#     plt.figure()
#     plt.plot(t1, S1 / num_node, "g--", label="S (discrete)")
#     plt.plot(t1, I1 / num_node, "r--", label="I (discrete)")
#     plt.plot(t1, R1 / num_node, "b--", label="R (discrete)")
#     plt.legend()
#     plt.xlabel("Time")
#     plt.ylabel("Fraction of population")
#     plt.savefig(os.path.join(output_dir, f"{es}_SIR.png"), dpi=300, bbox_inches='tight')
#     plt.show()
    
# if __name__ == "__main__":
#     es_list = np.linspace(0.15, 0.95, num=5)
#     for es in es_list:
#         dynamics_on_model(es, 9000, 300, 1000, 2, 11, True, 2, 2)