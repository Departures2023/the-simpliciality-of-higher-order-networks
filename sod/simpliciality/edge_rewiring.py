import numpy as np
import xgi
from ..trie import Trie
from .utilities import missing_subfaces, powerset


def rewire_Alg1(H, min_size=2, max_size=None):
    """
    Returns a list of maximal hyperedges that are not simplices.
    """
    # Filter edges bigger than min_size
    edges = H.edges.filterby("size", min_size, "geq").members()
    # Filter maximal edges bigger than min_size
    max_edges = (H.edges.maximal().filterby("size", min_size, "geq").members())
    # Build a trie for finding subfaces
    t = Trie()
    t.build_trie(edges)

    # edge_index record the index of the first maximal edge that has missing subfaces
    edge_index = 0
    # set_missing will contain the missing subfaces of the first maximal edge
    set_missing = set()
    # Iterate through the maximal edges to find the first one with missing subfaces
    for e in max_edges:
        set_missing.update(missing_subfaces(t, e, min_size))
        #print(set_missing)
        if len(set_missing) != 0:
            break
        edge_index += 1
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
    print("Actual rewired number:", count)
    print(success_add)
    print(success_delete)
    return H

# def non_simplex_maximal_edges(H, min_size=2, exclude_min_size=True):
#     """
#     Returns a list of maximal hyperedges that are not simplices.
#     """
#     t = Trie()
#     t.build_trie(H.edges.members())
#     edges = (
#         H.edges.maximal().filterby("size", min_size + exclude_min_size, "geq").members()
#     )
#     #non_simplex_edges = []
#     two_nodes_edges = []
#     for e in edges:
#         set_missing = missing_subfaces(H, e, min_size)
#         #num_missing = len(set_missing)
#         two_nodes_edges.append(x for x in set_missing if x == 2)
#         if two_nodes_edges>1:  # only considering simplices that are missing more than one subface
#             non_simplex_edges = e
#             break    
#     return non_simplex_edges, two_nodes_edges

###############################################################################
# def non_simplex_maximal_edges(H, min_size=2, exclude_min_size=True):
#     """
#     Returns a list of maximal hyperedges that are not simplices.
#     """

#     edges = H.edges.filterby("size", min_size, "geq").members()
#     max_edges = (
#         H.edges.maximal().filterby("size", min_size + exclude_min_size, "geq").members()
#     )

#     t = Trie()
#     t.build_trie(edges)

#     edge_index = 0
#     set_missing = set()
#     for e in max_edges:
#         set_missing.update(missing_subfaces(t, e, min_size=min_size))
#         if len(set_missing) != 0:
#             break
#         edge_index += 1
#     edges_remove = set()
#     edges_remove.update(x for x in powerset(max_edges[edge_index], min_size) if x not in set_missing)
#     H.add_edge(list(set_missing)[0], id="rewired_edge")
    
#     remove_id = 0
#     for id, edge in H.edges.members(dtype=dict).items():
#         if (edge == edges_remove):
#             remove_id = id
#     H.remove_edge(remove_id)
#     return H
####################################################################################################################


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