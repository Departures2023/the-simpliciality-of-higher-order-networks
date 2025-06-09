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
    #initialize return statistics
    global num_maximal_hyperedges
    global num_same_size
    global success_update
    global total_time
    global edges_searching_time
    global rewiring_time
    global num_missing_subface
    global delta_SF
    global delta_ES
    global delta_FES

    num_maximal_hyperedges = 0
    num_same_size = 0
    success_update = 0
    total_time = 0.0
    edges_searching_time = 0.0
    rewiring_time = 0.0
    num_missing_subface = 0
    delta_SF = 0.0
    delta_ES = 0.0
    delta_FES = 0.0

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
        f"delta_SF: {stats['delta_SF']:.6f}",
        f"delta_ES: {stats['delta_ES']:.6f}",
        f"delta_FES: {stats['delta_FES']:.6f}",
        "-" * 50
    ]
    
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
            num_missing_subface = len(set_missing)
            stats["num_missing_subface"] = num_missing_subface
            break
    
    # Edge_remove = P(maximal edge) - missing subfaces - maximal edges
    edges_remove = set()
    edges_remove.update(
        frozenset(x) for x in powerset(curr, min_size, max_size)
        if frozenset(x) not in set_missing and frozenset(x) not in map(frozenset, max_edges)
    )
    
    # Store the time taken for searching edges
    edges_searching_end = time.time()
    edges_searching_time = (edges_searching_end - edges_searching_start)
    stats["edges_searching_time"] = edges_searching_time
    
    # max_to_rewire is the maximum number of edges we can rewire (remove and add)
    max_to_rewire = min(len(edges_remove), len(set_missing))
    stats["max_to_rewire"] = max_to_rewire
    
    # Rewiring process start
    rewiring_start = time.time()
    
    for i in range(stats["max_to_rewire"]):
        # Randomly select an edge to remove and an edge to add
        tmp_remove = list(edges_remove)[random.randrange(0, len(edges_remove))]
        tmp_add = list(set_missing)[random.randrange(0, len(set_missing))]
        edges_remove.remove(tmp_remove)
        set_missing.remove(tmp_add)

        # The size of added edge and removed edge must be different
        global rewiring_time
        rewiring_time = 0.0
        global num_same_size
        num_same_size = 0
        if (len(tmp_add) != len(tmp_remove)):
            # Traverse through the edges of the hypergraph to find the edgeID of the edge to remove
            for id, edge in H.edges.members(dtype=dict).items():
                if (edge == tmp_remove):
                    # Remove the edge and add the new edge
                    H.remove_edge(id)
                    H.add_edge(tmp_add, id="rewired_edge")
                    
                    # Record the time taken for rewiring
                    rewiring_end = time.time()
                    rewiring_time += rewiring_end - rewiring_start
                    stats["rewiring_time"] = rewiring_time

                    # Update statistics

                    success_update = 1
                    stats["success_update"] = success_update
                    sf_tmp = simplicial_fraction(H, min_size=min_size)
                    es_tmp = edit_simpliciality(H, min_size=min_size)
                    fes_tmp = face_edit_simpliciality(H, min_size=min_size)
                    delta_SF = sf_init - sf_tmp
                    stats["delta_SF"] = delta_SF
                    delta_ES = es_init - es_tmp
                    stats["delta_ES"] = delta_ES
                    delta_FES = fes_init - fes_tmp
                    stats["delta_FES"] = delta_FES
                    break
            break
        else:
            # If the sizes are the same, we do not rewire (count the number of such cases)
            num_same_size += 1
            stats["num_same_size"] = num_same_size
    
    # Alg end time
    end_time = time.time()
    total_time = end_time - start_time
    stats["total_time"] = total_time
    list_all = [max_edges, max_to_rewire, success_update, num_same_size, total_time, edges_searching_time, rewiring_time, num_missing_subface, delta_SF, delta_ES, delta_FES] 
    return H, stats, list_all