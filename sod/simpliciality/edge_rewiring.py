import os
import random
import threading
import time
import numpy as np
import xgi

from ..simpliciality import edit_simpliciality, face_edit_simpliciality, simplicial_fraction
from ..trie import Trie
from .utilities import missing_subfaces, powerset


def rewire_Alg1(H, min_size=2, max_size=None):
    """
    Returns a list of maximal hyperedges that are not simplices.
    """
    es_init = edit_simpliciality(H, min_size=min_size)
    fes_init = face_edit_simpliciality(H, min_size=min_size)
    # Filter edges bigger than min_size
    edges = H.edges.filterby("size", min_size, "geq").members()
    # Filter maximal edges bigger than min_size
    max_edges = (H.edges.maximal().filterby("size", 4, "geq").members())
    tmp_max_edges = max_edges.copy()
    print("Maximal edges:", max_edges)
    # Build a trie for finding subfaces
    t = Trie()
    t.build_trie(edges)

    # edge_index record the index of the first maximal edge that has missing subfaces
    edge_index = 0
    # set_missing will contain the missing subfaces of the first maximal edge
    set_missing = set()
    curr = set()
    
    # RANDOMLY iterate through the maximal edges to find the first one with missing subfaces
    for i in range(len(max_edges), 0, -1):
        curr = tmp_max_edges[random.randint(0, i-1)]
        set_missing.update(missing_subfaces(t, curr, min_size))
        tmp_max_edges.remove(curr)
        if len(set_missing) != 0:
            break
    
    #############################################################################################################
    # SEQUENTIALLY Iterate through the maximal edges to find the first one with missing subfaces
    # for e in max_edges:
    #     set_missing.update(missing_subfaces(t, e, min_size))
    #     #print(set_missing)
    #     if len(set_missing) != 0:
    #         break
    #     edge_index += 1
    #############################################################################################################
    
    
    # Edge_remove = P(maximal edge) - missing subfaces - maximal edges
    edges_remove = set()
    # Print statement for debugging
    # print(len(max_edges))
    # print(max_edges)
    # print(set_missing)
    edges_remove.update(
        frozenset(x) for x in powerset(curr, min_size, max_size)
        if frozenset(x) not in set_missing and frozenset(x) not in map(frozenset, max_edges)
    )
    # Print statement for debugging
    # print(edges_remove)
    # print(sorted(edges_remove.union(set_missing), key = len))
    # print(len(edges_remove.union(set_missing)))
    # print(max_edges[edge_index])
    
    # max_to_rewire is the maximum number of edges we can rewire (remove and add)
    max_to_rewire = min(len(edges_remove), len(set_missing))
    print("max to rewire:", max_to_rewire)
    # Print statement for debugging
    #print(edges_remove)
    #print(set_missing)
    count = 0
    success_add = []
    success_delete = []
    delta_es = []
    delta_fes = []
    for i in range(max_to_rewire):
        tmp_remove = set(edges_remove.pop())
        tmp_add = set(set_missing.pop())
        # The size of added edge and removed edge must be different
        if (len(tmp_add) != len(tmp_remove)):
            # Traverse through the edges of the hypergraph to find the edgeID of the edge to remove
            for id, edge in H.edges.members(dtype=dict).items():
                # Print statement for debugging
                # print(edge, tmp_remove)
                if (edge == tmp_remove):
                    H.remove_edge(id)
                    H.add_edge(tmp_add, id="rewired_edge" + str(count))
                    count += 1
                    success_add.append(tmp_add)
                    success_delete.append(tmp_remove)
                    es_tmp = edit_simpliciality(H, min_size=min_size)
                    fes_tmp = face_edit_simpliciality(H, min_size=min_size)
                    delta_es.append(es_init - es_tmp)
                    delta_fes.append(fes_init - fes_tmp)
                    es_init = es_tmp
                    fes_init = fes_tmp
    print("Actual rewired number:", count)
    print("Edge added:" + str(success_add))
    print("Edge removed:" + str(success_delete))
    print("Delta ES:", str(delta_es))
    print("Delta FES:", str(delta_fes))
    return H




# def important_nodes(H, min_size=2, edges=None, nodes=None):
#     """
#     Returns a list of maximal hyperedges that are not simplices.
#     """
#     max_degree = max([edges.degree(e) for e in edges])
#     min_degree = min([edges.degree(e) for e in edges])
#     greatest_node = [e for e in edges if (edges.degree(e) == max_degree)]
#     least_node = [e for e in edges if (edges.degree(e) == min_degree)]
#     neighbor_edges = H.nodes.memberships(least_node).filterby("size", min_size, "geq").members()
    
#     for e in edges:
#         if 
#             non_simplex_maximal_edges.append(e)
#     return non_simplex_edges

def save_expr_data(dataset, round, stats, filename):
    """
    save the experiment data to a file.
    
    """
    
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Prepare the lines to write to the file
    lines = [
        f"dataset: {dataset}",
        f"round: {round}",
        f"num_maximal_hyperedge: {stats['num_maximal_hyperedge']}",
        f"max_to_rewire: {stats['max_to_rewire']}",
        f"success_update: {stats['success_update']}",
        f"num_same_size: {stats['num_same_size']}",
        f"total_time: {stats['total_time']:.3f}",
        f"edges_searching_time: {stats['edges_searching_time']:.3f}",
        f"rewiring_time: {stats['rewiring_time']:.3f}",
        f"num_missing_subface: {stats['num_missing_subface']}",
        f"delta_SF: {stats['delta_SF']:.6f}",
        f"delta_ES: {stats['delta_ES']:.6f}",
        f"delta_FES: {stats['delta_FES']:.6f}",
        "-" * 50
    ]
    
    # Open the file in append mode and write the lines, Change 'a' to 'w' to rewrite the file
    with open(filename, 'a', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

def rewire_Alg1_expr(H, min_size=2, max_size=None):
    """
    Returns a list of maximal hyperedges that are not simplices.
    """
    
    # Alg start time
    start_time = time.time()
    
    # Initialize the simplicial fraction, edit simpliciality, and face edit simpliciality
    sf_init = simplicial_fraction(H, min_size=min_size)
    es_init = edit_simpliciality(H, min_size=min_size)
    fes_init = face_edit_simpliciality(H, min_size=min_size)
    
    # Initialize statistics
    stats = {
        "num_maximal_hyperedge": 0,
        "max_to_rewire": 0,
        "success_update": 0,
        "num_same_size": 0,
        "total_time": 0.0,
        "edges_searching_time": 0.0,
        "rewiring_time": 0.0,
        "num_missing_subface": 0,
        "delta_SF": 0.0,
        "delta_ES": 0.0,
        "delta_FES": 0.0
    }
    
    # Edges searching process start
    edges_searching_start = time.time()
    
    # Filter edges bigger than min_size
    edges = H.edges.filterby("size", min_size, "geq").members()
    
    # Filter maximal edges bigger than min_size
    max_edges = (H.edges.maximal().filterby("size", 4, "geq").members())
    stats["num_maximal_hyperedge"] = len(max_edges)
    tmp_max_edges = max_edges.copy()

    # Build a trie for finding subfaces
    t = Trie()
    t.build_trie(edges)

    # set_missing will contain the missing subfaces of the first maximal edge
    set_missing = set()
    curr = set()
    
    # RANDOMLY iterate through the maximal edges to find the first one with missing subfaces
    for i in range(len(max_edges), 0, -1):
        curr = tmp_max_edges[random.randrange(0, i)]
        set_missing.update(missing_subfaces(t, curr, min_size))
        tmp_max_edges.remove(curr)
        if len(set_missing) != 0:
            stats["num_missing_subface"] = len(set_missing)
            break
    
    # Edge_remove = P(maximal edge) - missing subfaces - maximal edges
    edges_remove = set()
    edges_remove.update(
        frozenset(x) for x in powerset(curr, min_size, max_size)
        if frozenset(x) not in set_missing and frozenset(x) not in map(frozenset, max_edges)
    )
    
    # Store the time taken for searching edges
    edges_searching_end = time.time()
    stats["edges_searching_time"] = (edges_searching_end - edges_searching_start)
    
    # max_to_rewire is the maximum number of edges we can rewire (remove and add)
    stats["max_to_rewire"] = min(len(edges_remove), len(set_missing))
    
    # Rewiring process start
    rewiring_start = time.time()
    
    for i in range(stats["max_to_rewire"]):
        # Randomly select an edge to remove and an edge to add
        tmp_remove = list(edges_remove)[random.randrange(0, len(edges_remove))]
        tmp_add = list(set_missing)[random.randrange(0, len(set_missing))]
        edges_remove.remove(tmp_remove)
        set_missing.remove(tmp_add)

        # The size of added edge and removed edge must be different
        if (len(tmp_add) != len(tmp_remove)):
            # Traverse through the edges of the hypergraph to find the edgeID of the edge to remove
            for id, edge in H.edges.members(dtype=dict).items():
                if (edge == tmp_remove):
                    # Remove the edge and add the new edge
                    H.remove_edge(id)
                    H.add_edge(tmp_add, id="rewired_edge")
                    
                    # Record the time taken for rewiring
                    rewiring_end = time.time()
                    stats["rewiring_time"] += rewiring_end - rewiring_start
                    
                    # Update statistics
                    stats["success_update"] = 1
                    sf_tmp = simplicial_fraction(H, min_size=min_size)
                    es_tmp = edit_simpliciality(H, min_size=min_size)
                    fes_tmp = face_edit_simpliciality(H, min_size=min_size)
                    stats["delta_SF"] = sf_init - sf_tmp
                    stats["delta_ES"] = es_init - es_tmp
                    stats["delta_FES"] = fes_init - fes_tmp
                    break
            break
        else:
            # If the sizes are the same, we do not rewire (count the number of such cases)
            stats["num_same_size"] += 1
    
    # Alg end time
    end_time = time.time()
    stats["total_time"] = end_time - start_time
    return H, stats

# Use a single dataset to run the algorithm multiple times
def loop_Alg1_expr(index, iter, min_size, max_size):
    for i in range(iter):
        H, stats = rewire_Alg1_expr(graphs[index], min_size, max_size)
        H.cleanup(singletons=True)
        graphs[index] = H
        # Save the experiment data
        save_expr_data(datasets[index], i, stats, dir[datasets[i]])
        
        
if __name__ == "__main__":
    global graphs, dir, datasets
    graphs = []
    max_size = 11
    min_size = 2
    
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
    
    dir = {
        "contact-primary-school": "experiment_result/contact-primary-school.txt",
        "contact-high-school": "experiment_result/contact-high-school.txt",
        "hospital-lyon": "experiment_result/hospital-lyon.txt",
        "email-enron": "experiment_result/email-enron.txt",
        "email-eu": "experiment_result/email-eu.txt",
        "ndc-substances": "experiment_result/ndc-substances.txt",
        "diseasome": "experiment_result/diseasome.txt",
        "disgenenet": "experiment_result/disgenenet.txt",
        "congress-bills": "experiment_result/congress-bills.txt",
        "tags-ask-ubuntu": "experiment_result/tags-ask-ubuntu.txt",
    }
    
    # Load the datasets and clean them
    for i in range (10):
        graphs.append(xgi.load_xgi_data(datasets[i], max_order=max_size))
        graphs[i].cleanup(singletons=True)
    
    # Create threads to run the algorithm in parallel
    threads = []
    for i in range(10):
        thread = threading.Thread(target=loop_Alg1_expr, args=(i, 100, min_size, max_size,))
        threads.append(thread)
        thread.start()
        
    for thread in threads:
        thread.join()

    print("All threads finished")

