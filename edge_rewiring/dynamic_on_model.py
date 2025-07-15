import xgi
import hypercontagion as hc
import matplotlib.pyplot as plt
import time
import numpy as np
import random
from model_generation import *

datasets = [
    "contact-primary-school",
    "contact-high-school",
    "hospital-lyon",
    "email-enron",
    "email-eu",
    "ndc-substances",
    "diseasome",
    "disgenenet",
    "congress-bills",
    "tags-ask-ubuntu",
]

def run_multiple_SIR_with_errorbands(
    es,  
    approx_num_C, 
    num_max_hyperedge, 
    num_node,
    gamma,
    num_graphs=5,
    rho=0.1,
    tmin=0,
    tmax=100,
    dt=1,
    ax=None, 
):
    S_all, I_all, R_all = [], [], []
    t_vals = None  # to store time vector from first run

    for i in range(num_graphs):
        print(f"Simulation {i+1}/{num_graphs}")
        H = model_generation_es(
            # es=es, 
            # approx_num_C=approx_num_C, 
            # num_max_hyperedge=num_max_hyperedge,
            # num_node=num_node, 
            # min_size=2, 
            # max_size=None, 
            # adjust_es=True
            es=0.80432,
            approx_num_C=11200,  # Set high to allow target_num_edges to work
            num_max_hyperedge=8010,
            num_node=242,
            min_size=2,
            max_size=None,
            adjust_es=True,
            compare_interval_smaller_case=40,
            compare_interval_bigger_case=40,
        )
        H.cleanup(connected=True)
        es = edit_simpliciality(H)
        print(f"es = {es}")

        mean_degree = sum(dict(H.degree()).values()) / H.num_nodes
        print(mean_degree)

        tau = {k: 0.1 for k in xgi.unique_edge_sizes(H)}
        t, S, I, R = hc.discrete_SIR(H, tau, gamma=gamma, rho=rho, tmin=tmin, tmax=tmax, dt=dt)

        if t_vals is None:
            t_vals = t  # save the time vector
        
        min_len = min(len(S), len(I), len(R))
        S_all.append(S[:min_len] / num_node)
        I_all.append(I[:min_len] / num_node)
        R_all.append(R[:min_len] / num_node)

        if t_vals is None:
            t_vals = t[:min_len]

    #print(f"Run {i+1}: Length of I = {len(I)}, S = {len(S)}, R = {len(R)}")

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
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Use ax instead of axes for plotting
    ax.plot(t_vals, S_mean, color="green", label="S (mean)")
    ax.fill_between(t_vals, S_mean - S_std, S_mean + S_std, color="green", alpha=0.3)
    ax.plot(t_vals, I_mean, color="red", label="I (mean)")
    ax.fill_between(t_vals, I_mean - I_std, I_mean + I_std, color="red", alpha=0.3)
    ax.plot(t_vals, R_mean, color="blue", label="R (mean)")
    ax.fill_between(t_vals, R_mean - R_std, R_mean + R_std, color="blue", alpha=0.3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Fraction of Population")
    ax.set_title(f"SIR on Generated Code Error Bands")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig, ax

def SIR_original_graph(
    dataset,
    gamma,
    ax=None
):
    
    H = xgi.load_xgi_data(dataset)
    H.cleanup(connected=True)
    num_node = H.num_nodes
    tau = {i: 0.1 for i in xgi.unique_edge_sizes(H)}
    start = time.time()
    t1, S1, I1, R1 = hc.discrete_SIR(H, tau, gamma, tmin=0, tmax=100, dt=1, rho=0.1)
    print(time.time() - start)

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.plot(t1, S1 / num_node, "g--", label="S (discrete)")
    ax.plot(t1, I1 / num_node, "r--", label="I (discrete)")
    ax.plot(t1, R1 / num_node, "b--", label="R (discrete)")
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Fraction of population")
    ax.set_title("SIR on Original Graph")
    ax.grid(True)
    fig.tight_layout()
    return fig, ax

# Example usage:
if __name__ == "__main__":
    dataset = datasets[int(sys.argv[1])]
    es = float(sys.argv[2])
    approx_num_C = int(sys.argv[3])
    num_max_hyperedge = int(sys.argv[4])
    num_node = int(sys.argv[5])
    dataset = "contact-primary-school"
    gamma = 0.05

    print(f"Running SIR on dataset: {dataset}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

    run_multiple_SIR_with_errorbands(es, approx_num_C, num_max_hyperedge, num_node, gamma, ax=axes[0])
    SIR_original_graph(dataset, gamma, ax=axes[1])

    H = xgi.load_xgi_data(dataset)
    es = round(edit_simpliciality(H), 2)
    fig.suptitle(dataset + "  -  es = " + str(es))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
