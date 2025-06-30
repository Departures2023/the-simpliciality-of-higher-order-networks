import xgi
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
import random
import pandas as pd
import seaborn as sns
import time
import statistics
import math
import os
from statsmodels.stats.proportion import proportion_confint
from model_generation import model_generation_es
from edge_rewiring.power_law import has_powerlaw
from multiprocessing import Pool, cpu_count
from sod import simplicial_fraction, edit_simpliciality, face_edit_simpliciality

random.seed(42)
np.random.seed(42)

# Directories for saving stats files
dir = {
    "general_data": "experiment_result/model_generation_es/general_data.txt",
    "cumulative_data": "experiment_result/model_generation_es/cumulative_data.txt",
}

# Globals for detailed stats collection (for model_generation_es_exper)
graph_lst = []
local_cluster_coefficients_average_lst = []
connected_components_lst = []
simplicial_fraction_lst = []
edit_simpliciality_lst = []
face_edit_simpliciality_lst = []
density_lst = []
degree_count_average_lst = []
degree_assortativity_lst = []
num_node_lst = []
num_edge_lst = []
evaluation_time_lst = []
graph_generation_time_lst = []

cumulative_stats = {
    "local_cluster_coefficient_avg": 0,
    "local_cluster_coefficient_std": 0,
    "connected_components_avg": 0,
    "connected_components_std": 0,
    "simplicial_fraction_avg": 0,
    "simplicial_fraction_std": 0,
    "edit_simpliciality_avg": 0,
    "edit_simpliciality_std": 0,
    "cumulative_edit_simpliciality_diff": 0,
    "face_edit_simpliciality_avg": 0,
    "face_edit_simpliciality_std": 0,
    "density_avg": 0,
    "density_std": 0,
    "degree_count_avg": 0,
    "degree_count_median": 0,
    "degree_assortativity_avg": 0,
    "degree_assortativity_std": 0,
    "num_node_avg": 0,
    "num_edge_avg": 0,
    "num_node_std": 0,
    "num_edge_std": 0,
    "num_node_median": 0,
    "num_edge_median": 0,
    "num_node_max": 0,
    "num_edge_max": 0,
    "num_node_min": 0,
    "num_edge_min": 0,
    "evaluation_time_avg": 0,
    "evaluation_time_std": 0,
    "graph_generation_time_avg": 0,
    "graph_generation_time_std": 0,
}

def save_general_data(trial, es, stats, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    lines = [
        f"trial: {trial}",
        f"expected_es: {es}",
        f"local_cluster_coefficient_average: {stats['local_cluster_coefficient_average']:.5f}",
        f"connected_components: {stats['connected_components']}",
        f"simplicial_fraction: {stats['simplicial_fraction']:.5f}",
        f"edit_simpliciality: {stats['edit_simpliciality']:.5f}",
        f"face_edit_simpliciality: {stats['face_edit_simpliciality']:.5f}",
        f"density: {stats['density']:.5f}",
        f"degree_count_average: {stats['degree_count_average']:.5f}",
        f"degree_assortativity: {stats['degree_assortativity']:.5f}",
        f"num_node: {stats['num_node']}",
        f"num_edge: {stats['num_edge']}",
        f"evaluation_time: {stats['evaluation_time']:.3f}",
        f"graph_generation_time: {stats['graph_generation_time']:.3f}",
        "=" * 50
    ]
    with open(filename, 'a', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

def save_cumulative_data(trial, es, stats, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    lines = [
        f"trial: {trial}",
        f"expected_es: {es}",
        f"local_cluster_coefficient_average: {stats['local_cluster_coefficient_average']:.5f}",
        f"connected_components: {stats['connected_components']}",
        f"simplicial_fraction: {stats['simplicial_fraction']:.5f}",
        f"edit_simpliciality: {stats['edit_simpliciality']:.5f}",
        f"face_edit_simpliciality: {stats['face_edit_simpliciality']:.5f}",
        f"density: {stats['density']:.5f}",
        f"degree_count_average: {stats['degree_count_average']:.5f}",
        f"degree_assortativity: {stats['degree_assortativity']:.5f}",
        f"num_node: {stats['num_node']}",
        f"num_edge: {stats['num_edge']}",
        f"evaluation_time: {stats['evaluation_time']:.3f}",
        f"graph_generation_time: {stats['graph_generation_time']:.3f}",
        "=" * 50
    ]
    with open(filename, 'a', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

def model_generation_es_exper(trial, es, approx_num_C, num_max_hyperedge, num_node, min_size, max_size, adjust_es):
    global graph_lst, local_cluster_coefficients_average_lst, connected_components_lst, simplicial_fraction_lst, edit_simpliciality_lst, \
        face_edit_simpliciality_lst, density_lst, degree_count_average_lst, degree_assortativity_lst, num_node_lst, num_edge_lst, \
        evaluation_time_lst, graph_generation_time_lst, cumulative_stats
    
    # Clear all global lists at the beginning of each experiment
    graph_lst.clear()
    local_cluster_coefficients_average_lst.clear()
    connected_components_lst.clear()
    simplicial_fraction_lst.clear()
    edit_simpliciality_lst.clear()
    face_edit_simpliciality_lst.clear()
    density_lst.clear()
    degree_count_average_lst.clear()
    degree_assortativity_lst.clear()
    num_node_lst.clear()
    num_edge_lst.clear()
    evaluation_time_lst.clear()
    graph_generation_time_lst.clear()
    
    # Clear the files at the beginning of each experiment (Uncomment to clear the files)
    # os.makedirs(os.path.dirname(dir["general_data"]), exist_ok=True)
    # os.makedirs(os.path.dirname(dir["cumulative_data"]), exist_ok=True)
    # with open(dir["general_data"], 'w') as f:
    #     f.write("")  # Clear the file
    # with open(dir["cumulative_data"], 'w') as f:
    #     f.write("")  # Clear the file
    for i in range(trial):
        # Initialize statistics
        stats = {
            "local_cluster_coefficient_average": 0,
            "connected_components": 0,
            "simplicial_fraction": 0,
            "edit_simpliciality": 0,
            "face_edit_simpliciality": 0,
            "density": 0,
            "degree_count_average": 0,
            "degree_assortativity": 0,
            "num_node": 0,
            "num_edge": 0,
            "evaluation_time": 0.0,
            "graph_generation_time": 0.0,
        }
        # graph generation time
        start_time = time.time()
        H_es = model_generation_es(es, approx_num_C, num_max_hyperedge, num_node, min_size, max_size, adjust_es)
        graph_lst.append(H_es)
        end_time = time.time()
        stats["graph_generation_time"] = end_time - start_time
        graph_generation_time_lst.append(stats["graph_generation_time"])
        
        # graph evaluation time start
        start_time = time.time()
        # local cluster coefficient
        try:
            local_cluster_coefficients = xgi.local_clustering_coefficient(H_es)
            local_cluster_coefficient_average = sum(local_cluster_coefficients.values()) / len(local_cluster_coefficients)
            stats["local_cluster_coefficient_average"] = local_cluster_coefficient_average
        except (KeyError, ValueError, IndexError) as e:
            # Handle cases where clustering coefficient calculation fails
            stats["local_cluster_coefficient_average"] = float('nan')
            print(f"Warning: Local clustering coefficient calculation failed for trial {i+1}: {e}")
        local_cluster_coefficients_average_lst.append(stats["local_cluster_coefficient_average"])
        # connected components
        stats["connected_components"] = xgi.number_connected_components(H_es)
        connected_components_lst.append(stats["connected_components"])
        # simplicial fraction
        stats["simplicial_fraction"] = simplicial_fraction(H_es)
        simplicial_fraction_lst.append(stats["simplicial_fraction"])
        # edit simpliciality
        stats["edit_simpliciality"] = edit_simpliciality(H_es)
        edit_simpliciality_lst.append(stats["edit_simpliciality"])
        # face edit simpliciality
        stats["face_edit_simpliciality"] = face_edit_simpliciality(H_es)
        face_edit_simpliciality_lst.append(stats["face_edit_simpliciality"])
        # density
        try:
            stats["density"] = xgi.density(H_es, ignore_singletons=True)
        except OverflowError:
            # Handle the case where density calculation overflows due to large numbers
            stats["density"] = float('nan')  # or 0, or some other default value
            print(f"Warning: Density calculation overflowed for trial {i+1}, setting to NaN")
        density_lst.append(stats["density"])
        # num node
        stats["num_node"] = H_es.num_nodes
        num_node_lst.append(stats["num_node"])
        # num edge
        stats["num_edge"] = H_es.num_edges
        num_edge_lst.append(stats["num_edge"])
        # degree count average
        degree_count = xgi.degree_counts(H_es)
        for degree in range(len(degree_count)):
            stats["degree_count_average"] += degree * degree_count[degree]
        stats["degree_count_average"] /= stats["num_node"]
        degree_count_average_lst.append(stats["degree_count_average"])
        # degree assortativity
        if len(num_edge_lst) > 1:
            # degree assortativity
            stats["degree_assortativity"] = xgi.degree_assortativity(H_es)
            degree_assortativity_lst.append(stats["degree_assortativity"])
        else:
            stats["degree_assortativity"] = -1
        
        
        # evaluation time
        end_time = time.time()
        stats["evaluation_time"] = end_time - start_time
        evaluation_time_lst.append(stats["evaluation_time"])
        save_general_data(i+1, es, stats, dir["general_data"])
    # calculate the average of the stats
    # Filter out NaN values before calculating local clustering coefficient statistics
    valid_cluster_lst = [c for c in local_cluster_coefficients_average_lst if not (isinstance(c, float) and math.isnan(c))]
    cumulative_stats["local_cluster_coefficient_avg"] = sum(valid_cluster_lst) / len(valid_cluster_lst) if len(valid_cluster_lst) > 0 else 0
    cumulative_stats["connected_components_avg"] = sum(connected_components_lst) / len(connected_components_lst) if len(connected_components_lst) > 1 else 0
    cumulative_stats["simplicial_fraction_avg"] = sum(simplicial_fraction_lst) / len(simplicial_fraction_lst) if len(simplicial_fraction_lst) > 1 else 0
    cumulative_stats["edit_simpliciality_avg"] = sum(edit_simpliciality_lst) / len(edit_simpliciality_lst) if len(edit_simpliciality_lst) > 1 else 0
    cumulative_stats["face_edit_simpliciality_avg"] = sum(face_edit_simpliciality_lst) / len(face_edit_simpliciality_lst) if len(face_edit_simpliciality_lst) > 1 else 0
    # Filter out NaN values before calculating density statistics
    valid_density_lst = [d for d in density_lst if not (isinstance(d, float) and math.isnan(d))]
    cumulative_stats["density_avg"] = sum(valid_density_lst) / len(valid_density_lst) if len(valid_density_lst) > 0 else 0
    cumulative_stats["degree_count_avg"] = sum(degree_count_average_lst) / len(degree_count_average_lst) if len(degree_count_average_lst) > 1 else 0
    cumulative_stats["degree_assortativity_avg"] = sum(degree_assortativity_lst) / len(degree_assortativity_lst) if len(degree_assortativity_lst) > 1 else 0
    cumulative_stats["num_node_avg"] = sum(num_node_lst) / len(num_node_lst) if len(num_node_lst) > 1 else 0
    cumulative_stats["num_edge_avg"] = sum(num_edge_lst) / len(num_edge_lst) if len(num_edge_lst) > 1 else 0
    cumulative_stats["evaluation_time_avg"] = sum(evaluation_time_lst) / len(evaluation_time_lst) if len(evaluation_time_lst) > 1 else 0
    cumulative_stats["graph_generation_time_avg"] = sum(graph_generation_time_lst) / len(graph_generation_time_lst) if len(graph_generation_time_lst) > 1 else 0
    
    # calculate standard deviations
    cumulative_stats["local_cluster_coefficient_std"] = statistics.stdev(valid_cluster_lst) if len(valid_cluster_lst) > 1 else 0
    cumulative_stats["connected_components_std"] = statistics.stdev(connected_components_lst) if len(connected_components_lst) > 1 else 0
    cumulative_stats["simplicial_fraction_std"] = statistics.stdev(simplicial_fraction_lst) if len(simplicial_fraction_lst) > 1 else 0
    cumulative_stats["edit_simpliciality_std"] = statistics.stdev(edit_simpliciality_lst) if len(edit_simpliciality_lst) > 1 else 0
    cumulative_stats["face_edit_simpliciality_std"] = statistics.stdev(face_edit_simpliciality_lst) if len(face_edit_simpliciality_lst) > 1 else 0
    cumulative_stats["density_std"] = statistics.stdev(valid_density_lst) if len(valid_density_lst) > 1 else 0
    cumulative_stats["degree_assortativity_std"] = statistics.stdev(degree_assortativity_lst) if len(degree_assortativity_lst) > 1 else 0
    cumulative_stats["num_node_std"] = statistics.stdev(num_node_lst) if len(num_node_lst) > 1 else 0
    cumulative_stats["num_edge_std"] = statistics.stdev(num_edge_lst) if len(num_edge_lst) > 1 else 0
    cumulative_stats["evaluation_time_std"] = statistics.stdev(evaluation_time_lst) if len(evaluation_time_lst) > 1 else 0
    cumulative_stats["graph_generation_time_std"] = statistics.stdev(graph_generation_time_lst) if len(graph_generation_time_lst) > 1 else 0
    
    # calculate medians
    cumulative_stats["degree_count_median"] = statistics.median(degree_count_average_lst) if len(degree_count_average_lst) > 1 else 0
    cumulative_stats["num_node_median"] = statistics.median(num_node_lst) if len(num_node_lst) > 1 else 0
    cumulative_stats["num_edge_median"] = statistics.median(num_edge_lst) if len(num_edge_lst) > 1 else 0
    
    # calculate min/max values
    cumulative_stats["num_node_max"] = max(num_node_lst) if len(num_node_lst) > 1 else 0
    cumulative_stats["num_edge_max"] = max(num_edge_lst) if len(num_edge_lst) > 1 else 0
    cumulative_stats["num_node_min"] = min(num_node_lst) if len(num_node_lst) > 1 else 0
    cumulative_stats["num_edge_min"] = min(num_edge_lst) if len(num_edge_lst) > 1 else 0
    
    # calculate cumulative edit simpliciality difference (this is the difference from target es)
    for es_actual in edit_simpliciality_lst:
        cumulative_stats["cumulative_edit_simpliciality_diff"] += es - es_actual
    
    save_cumulative_data(trial, es, cumulative_stats, dir["cumulative_data"])

def run_powerlaw_trials(es, trials=400, approx_num_C=1200, num_max_hyperedge=300, num_node=1200, min_size=2, max_size=20):
    success_count = 0
    all_stats = []
    start_time = time.time()
    for i in range(trials):
        H = model_generation_es(
            es=es,
            approx_num_C=approx_num_C,
            num_max_hyperedge=num_max_hyperedge,
            num_node=num_node,
            min_size=min_size,
            max_size=max_size,
        )
        result = has_powerlaw(H, p_value=0.10)
        all_stats.append({
            "es": es,
            "trial": i,
            "follows_powerlaw": result[0] if isinstance(result, tuple) else result,
            "run_time": time.time() - start_time,
        })
        if result[0] if isinstance(result, tuple) else result:
            success_count += 1
    end_time = time.time()
    duration = end_time - start_time
    lower, upper = proportion_confint(success_count, trials, method='wilson')
    print(f"es={es} | Success: {success_count}/{trials} | CI=({lower:.2f}, {upper:.2f}) | Time: {duration:.2f}s")
    return {
        "es": es,
        "success_count": success_count,
        "trials": trials,
        "success_rate": success_count / trials,
        "CI_lower": lower,
        "CI_upper": upper,
        "duration": duration,
        "all_stats": all_stats,
    }

def time_graph_generation(es, num_node, approx_num_C=9000, num_max_hyperedge=300, min_size=2, max_size=11):
    start = time.time()
    H = model_generation_es(es, approx_num_C, num_max_hyperedge, num_node, min_size, max_size, False)
    end = time.time()
    print(f"Graph generation time for n={num_node}, es={es}: {end - start:.3f} seconds")
    return end - start

def worker(es):
    print(f"Starting experiments for es={es}")
    # Run detailed stats experiment (few trials for speed)
    model_generation_es_exper(5, es, 9000, 300, 1000, 2, 11, False)
    # Run power-law success trial (longer)
    return run_powerlaw_trials(es)

if __name__ == "__main__":
    es_values = [0.2, 0.6, 0.7, 0.8, 0.9]

    # Option 1: Serial run (simple)
    for es in es_values:
        worker(es)

    # Option 2: Parallel run (faster)
    with Pool(processes=cpu_count() - 1) as pool:
        results = pool.map(worker, es_values)

    # After experiments, aggregate or save results as needed
    df_powerlaw = pd.DataFrame(results)
    df_powerlaw.to_csv("powerlaw_success_rates.csv", index=False)

    # Optionally plot success rates
    sns.barplot(x="es", y="success_rate", data=df_powerlaw)
    plt.title("Power-law Success Rate by Edit Simpliciality")
    plt.ylabel("Success Rate")
    plt.xlabel("Edit Simpliciality (es)")
    plt.show()
