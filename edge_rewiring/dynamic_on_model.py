import xgi
import hypercontagion as hc
import matplotlib.pyplot as plt
import time
import numpy as np
import random
from model_generation import *
from functools import partial
import multiprocessing as mp
import datetime
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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

def single_SIR_simulation(es, approx_num_C, num_max_hyperedge, num_node, gamma, rho, tmin, tmax, dt, i):

    print(f"Generating graph {i+1} with es = {es}, approx_num_C = {approx_num_C}, num_max_hyperedge = {num_max_hyperedge}, num_node = {num_node}")
    start = time.time()
    H = model_generation_es(
        es=es,
        approx_num_C=approx_num_C,
        num_max_hyperedge=num_max_hyperedge,
        num_node=num_node,
        min_size=2,
        max_size=None,
        adjust_es=True
    )
    H.cleanup()
    es_new = new_edit_simpliciality(H)
    error_es = abs(es_new - es)

    mean_degree = sum(dict(H.degree()).values()) / H.num_nodes
    tau = {k: 0.1/k for k in xgi.unique_edge_sizes(H)}

    edges = 4000 + (int(es * 10 - 1) - 1) * 1000
    num_max_hyperedge_new = len(H.edges.maximal().filterby("size", 2, "geq"))
    approx_num_C_new = (H.num_edges - num_max_hyperedge_new + es_new * num_max_hyperedge)/es_new

    print(f"es = {es_new}, node = {H.num_nodes}, edges = {H.num_edges}, num_max_hyperedge = {num_max_hyperedge_new}, approx_c = {approx_num_C_new} for {i+1}th graph \n"
          f"error : es = {es_new - es}, node = {H.num_nodes - num_node}, edges = {H.num_edges - edges}, num_max_hyperedge = {num_max_hyperedge_new - num_max_hyperedge}, approx_c = {approx_num_C_new - approx_num_C} \n")

    t, S, I, R = hc.discrete_SIR(H, tau, gamma=gamma, rho=rho, tmin=tmin, tmax=tmax, dt=dt)
    end = time.time()
    print(f"Graph {i+1} simulation completed in {end - start:.2f} seconds")

    min_len = min(len(S), len(I), len(R))
    return {
        'S': S[:min_len] / num_node,
        'I': I[:min_len] / num_node,
        'R': R[:min_len] / num_node,
        't': t[:min_len],
        'error_es': error_es
    }




def run_multiple_SIR_with_errorbands(
    es,  
    approx_num_C, 
    num_max_hyperedge, 
    num_node,
    gamma,
    colors,
    num_graphs=10,
    rho=0.1,
    tmin=0,
    tmax=100,
    dt=1,
    ax=None, 
):
    print(f"Running SIR with error bands for {num_graphs} graphs in parallel...")

    # Prepare multiprocessing pool
    pool = mp.Pool(processes=mp.cpu_count())

    # Partial function for passing fixed arguments
    simulation_func = partial(
        single_SIR_simulation,
        es,
        approx_num_C,
        num_max_hyperedge,
        num_node,
        gamma,
        rho,
        tmin,
        tmax,
        dt
    )

    # Run simulations in parallel
    results = pool.map(simulation_func, range(num_graphs))
    pool.close()
    pool.join()

    # Aggregate results
    S_all, I_all, R_all = [], [], []
    error_es = 0
    t_vals = None

    min_len = min(len(res['t']) for res in results)
    for res in results:
        S_all.append(res['S'][:min_len])
        I_all.append(res['I'][:min_len])
        R_all.append(res['R'][:min_len])
        error_es += res['error_es']
        if t_vals is None:
            t_vals = res['t'][:min_len]

    print(f"Average error in es: {error_es / num_graphs} while es = {es}")

    S_all = np.array(S_all)
    I_all = np.array(I_all)
    R_all = np.array(R_all)

    S_mean, S_std = np.mean(S_all, axis=0), np.std(S_all, axis=0)
    I_mean, I_std = np.mean(I_all, axis=0), np.std(I_all, axis=0)
    R_mean, R_std = np.mean(R_all, axis=0), np.std(R_all, axis=0)

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.plot(t_vals, S_mean, color=colors[0], label="S (mean)")
    ax.fill_between(t_vals, S_mean - S_std, S_mean + S_std, color=colors[0], alpha=0.3)
    ax.plot(t_vals, I_mean, color=colors[1], label="I (mean)")
    ax.fill_between(t_vals, I_mean - I_std, I_mean + I_std, color=colors[1], alpha=0.3)
    ax.plot(t_vals, R_mean, color=colors[2], label="R (mean)")
    ax.fill_between(t_vals, R_mean - R_std, R_mean + R_std, color=colors[2], alpha=0.3)

    ax.set_xlabel("Time")
    ax.set_ylabel("Fraction of Population")
    ax.set_title(f"SIR on Generated Graphs")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig, ax

def SIR_original_graph(
    dataset,
    gamma,
    colors,
    ax=None
):
    
    H = xgi.load_xgi_data(dataset)
    num_node = H.num_nodes
    tau = {i: 0.1/i for i in xgi.unique_edge_sizes(H)}
    start = time.time()
    t1, S1, I1, R1 = hc.discrete_SIR(H, tau, gamma, tmin=0, tmax=100, dt=1, rho=0.1)
    print(time.time() - start)

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.plot(t1, S1 / num_node, "g--", color = colors[0], label="S (discrete)")
    ax.plot(t1, I1 / num_node, "r--", color = colors[1], label="I (discrete)")
    ax.plot(t1, R1 / num_node, "b--", color = colors[2], label="R (discrete)")
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
    H_og = (xgi.load_xgi_data(dataset, max_order=11))
    es = new_edit_simpliciality(H_og)
    num_edges = H_og.num_edges
    num_max_hyperedge = len(H_og.edges.maximal().filterby("size", 2, "geq"))
    approx_num_C = (num_edges - num_max_hyperedge + es * num_max_hyperedge) / es
    num_node = H_og.num_nodes
    gamma = 0.05
    colors = ["#00B388","#DA291C", "#418FDF"]

    print(f"Running SIR on dataset: {dataset}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

    run_multiple_SIR_with_errorbands(es, approx_num_C, num_max_hyperedge, num_node, gamma, colors, num_graphs=7, ax=axes[0])
    SIR_original_graph(dataset, gamma, colors, ax=axes[1])

    H = xgi.load_xgi_data(dataset)
    es = round(edit_simpliciality(H), 2)
    fig.suptitle(dataset + "  -  es = " + str(es))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    summary_path = f"experiment_result/dynamics_on_model/SIR_on_{dataset}_Error_Bands_{run_id}.png"

    fig.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.show()