import copy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import time
import numpy as np
import xgi
from sod.simpliciality import edit_simpliciality, face_edit_simpliciality, simplicial_fraction
from sod.trie import Trie
from sod.simpliciality.utilities import missing_subfaces, powerset


def rewire_Alg1(H, min_size=2, max_size=None):
    """
    Returns a list of maximal hyperedges that are not simplices.
    """
    #es_init = edit_simpliciality(H, min_size=min_size)
    #fes_init = face_edit_simpliciality(H, min_size=min_size)
    # Filter edges bigger than min_size
    edges = H.edges.filterby("size", min_size, "geq").members()
    # Filter maximal edges bigger than min_size
    max_edges = (H.edges.maximal().filterby("size", 4, "geq").members())

    # Build a trie for finding subfaces
    t = Trie()
    t.build_trie(edges)

    weight = []
    
    
    set_missing_list = []
    edge_remove_list = []
    for e in max_edges:
        # set_missing will contain the missing subfaces of the current maximal edge
        set_missing = set()
        # edges_remove will contain the edges to remove from the current maximal edge
        edges_remove = set()
        # Find the "missing subfaces" and "edges to remove" of the current maximal edge
        set_missing.update(missing_subfaces(t, e, min_size))
        edges_remove.update(frozenset(x) for x in powerset(e, min_size, max_size)
            if frozenset(x) not in set_missing and frozenset(x) not in map(frozenset, max_edges))
        
        if (len(set_missing) != 0) and (len(edges_remove) != 0):
            set_missing_list.append(set_missing)
            edge_remove_list.append(edges_remove)
            weight.append(1 / (len(edges_remove) * len(set_missing)))
    
    print("Weight:", weight)
    print("Set missing list:", set_missing_list)
    print("Edge remove list:", edge_remove_list)
    
    # Randomly select a maximal edge with the given weight
    curr_idx = random.choices(range(len(set_missing_list)), weights=weight, k=1)[0]
    print("Current index:", curr_idx)
    # Get the set missing and edges remove of the selected maximal edge
    set_missing = set_missing_list[curr_idx]
    edges_remove = edge_remove_list[curr_idx]
    
    while len(set_missing) > 0:
        # Randomly select an edge to add
        tmp_add = list(set_missing)[random.randrange(0, len(set_missing))]
        # Remove the edge with the same size as the edge to add
        tmp_set_missing = copy.deepcopy(set_missing)
        for e in tmp_set_missing:
            if len(e) == len(tmp_add):
                set_missing.remove(e)
        # Find the edge in edges_remove that has different size than the edge to add
        tmp_remove_list = []
        for e in edges_remove:
            if len(e) != len(tmp_add):
                tmp_remove_list.append(e)
        if len(tmp_remove_list) > 0:
            # Randomly select an edge to remove
            tmp_remove = tmp_remove_list[random.randrange(0, len(tmp_remove_list))]
            
            # Traverse through the edges of the hypergraph to find the edgeID of the edge to remove
            for id, edge in H.edges.members(dtype=dict).items():
                if (edge == tmp_remove):
                    H.remove_edge(id)
                    H.add_edge(tmp_add, id="rewired_edge")
                    return H
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
    #gets all of the stats 
    os.makedirs(os.path.dirname(filename), exist_ok=True)
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
        "-" * 50
    ]
    #opens and writes to the file
    with open(filename, 'a', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

""""
Rewire_Alg1_expr
Inputs: 
    H - hypergraph 
    min_size - minimum size of edges, set to two to avoid singletons
    max_size - maximum size of edges, set to None

Output:
    Returns H, the rewired hypergraph
    Stores statistics in stats, including number of maximal hyperedges, maximum number of edges to rewire,
    success updates, number of same size edges, total time taken, edges searching time, and rewiring time,

Rewires edges in the given hypergraph by removing one and adding another of different size.
"""
def rewire_Alg1_expr(H, min_size=2, max_size=None):
    # Alg start time
    start_time = time.time()
    
    # Initialize statistics
    stats = {
        "num_maximal_hyperedge": 0,
        "max_to_rewire": 0,
        "success_update": 0,
        "num_same_size": 0,
        "total_time": 0.0,
        "edges_searching_time": 0.0,
        "rewiring_time": 0.0,
        "num_missing_subface": 0
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
                    break
            break
        else:
            # If the sizes are the same, we do not rewire (count the number of such cases)
            stats["num_same_size"] += 1
    
    # Alg end time
    end_time = time.time()
    stats["total_time"] = end_time - start_time
    return H, stats