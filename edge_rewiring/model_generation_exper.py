import xgi
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sod.simpliciality import edit_simpliciality, face_edit_simpliciality, simplicial_fraction
import matplotlib.pyplot as plt
from edge_rewiring import model_generation
from multiprocessing import Process, Manager
def es_new (es, approx_num_C, num_max_hyperedge, num_node, min_size, max_size, lst):
    H_es = model_generation.model_generation_es(es, approx_num_C, num_max_hyperedge, num_node, min_size, max_size)
    es = edit_simpliciality(H_es, min_size=2)
    lst.append(es)
    

if __name__ == "__main__":
    with Manager() as manager:
        lst = manager.list()   
        processes = []
        for i in range(1, 200, 10): 
            p = Process(target=es_new, args=(0.4, 30, i, 200, 2, 10, lst))
            processes.append(p)
            p.start() 
            
        for p in processes:
            p.join()  
            
        print(lst)
        
        plt.plot(range(1, 200, 10), lst, marker='o')
        plt.show()
        

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_generation')))
from model_generation import *
from sod import *
from sod.simpliciality import simplicial_fraction, edit_simpliciality, face_edit_simpliciality
import threading
from colorama import Fore
from edge_rewiring import *
from colorama import init
from termcolor import colored
from multiprocessing import Process, Manager, Queue
import time
import statistics
import numpy as np
import math

dir = {
    "general_data": "experiment_result/model_generation_es/general_data.txt",
    "cumulative_data": "experiment_result/model_generation_es/cumulative_data.txt",
}

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
    # gets all of the stats 
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
    # opens and writes to the file
    with open(filename, 'a', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

def save_cumulative_data(trial, es, stats, filename):
    # gets all of the cumulative stats 
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    lines = [
        f"trial: {trial}",
        f"expected_es: {es}",
        f"local_cluster_coefficient_avg: {stats['local_cluster_coefficient_avg']:.5f}",
        f"local_cluster_coefficient_std: {stats['local_cluster_coefficient_std']:.5f}",
        f"connected_components_avg: {stats['connected_components_avg']:.5f}",
        f"connected_components_std: {stats['connected_components_std']:.5f}",
        f"simplicial_fraction_avg: {stats['simplicial_fraction_avg']:.5f}",
        f"simplicial_fraction_std: {stats['simplicial_fraction_std']:.5f}",
        f"edit_simpliciality_avg: {stats['edit_simpliciality_avg']:.5f}",
        f"edit_simpliciality_std: {stats['edit_simpliciality_std']:.5f}",
        f"cumulative_edit_simpliciality_diff: {stats['cumulative_edit_simpliciality_diff']:.5f}",
        f"face_edit_simpliciality_avg: {stats['face_edit_simpliciality_avg']:.5f}",
        f"face_edit_simpliciality_std: {stats['face_edit_simpliciality_std']:.5f}",
        f"density_avg: {stats['density_avg']:.5f}",
        f"density_std: {stats['density_std']:.5f}",
        f"degree_count_avg: {stats['degree_count_avg']:.5f}",
        f"degree_count_median: {stats['degree_count_median']:.5f}",
        f"degree_assortativity_avg: {stats['degree_assortativity_avg']:.5f}",
        f"degree_assortativity_std: {stats['degree_assortativity_std']:.5f}",
        f"num_node_avg: {stats['num_node_avg']:.5f}",
        f"num_edge_avg: {stats['num_edge_avg']:.5f}",
        f"num_node_std: {stats['num_node_std']:.5f}",
        f"num_edge_std: {stats['num_edge_std']:.5f}",
        f"num_node_median: {stats['num_node_median']:.5f}",
        f"num_edge_median: {stats['num_edge_median']:.5f}",
        f"num_node_max: {stats['num_node_max']}",
        f"num_edge_max: {stats['num_edge_max']}",
        f"num_node_min: {stats['num_node_min']}",
        f"num_edge_min: {stats['num_edge_min']}",
        f"evaluation_time_avg: {stats['evaluation_time_avg']:.3f}",
        f"evaluation_time_std: {stats['evaluation_time_std']:.3f}",
        f"graph_generation_time_avg: {stats['graph_generation_time_avg']:.3f}",
        f"graph_generation_time_std: {stats['graph_generation_time_std']:.3f}",
        "=" * 50
    ]
    #opens and writes to the file
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
    

if __name__ == "__main__":
    es_lst = np.linspace(0.1, 0.95, num=50)
    for es in es_lst:
        model_generation_es_exper(5, es, 9000, 300, 1000, 2, 11, False)
    
