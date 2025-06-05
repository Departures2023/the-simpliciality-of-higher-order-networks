import random
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
    
    # RANDOMLY iterate through the maximal edges to find the first one with missing subfaces
    for i in range(len(max_edges), 0, -1):
        curr = tmp_max_edges[random.randint(0, i-1)]
        set_missing.update(missing_subfaces(t, curr, min_size))
        tmp_max_edges.remove(curr)
        if len(set_missing) != 0:
            edge_index = max_edges.index(curr)
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
        frozenset(x) for x in powerset(max_edges[edge_index], min_size, max_size)
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