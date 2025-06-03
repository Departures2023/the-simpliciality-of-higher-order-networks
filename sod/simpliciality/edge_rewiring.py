import numpy as np
import xgi
from ..trie import Trie
from .utilities import missing_subfaces, powerset


def non_simplex_maximal_edges(H, min_size=2, exclude_min_size=True):
    """
    Returns a list of maximal hyperedges that are not simplices.
    """

    edges = H.edges.filterby("size", min_size, "geq").members()
    max_edges = (
        H.edges.maximal().filterby("size", min_size + exclude_min_size, "geq").members()
    )

    t = Trie()
    t.build_trie(edges)

    edge_index = 0
    set_missing = set()
    for e in max_edges:
        set_missing.update(missing_subfaces(t, e, min_size=min_size))
        if len(set_missing) != 0:
            break
        edge_index += 1
    edges_remove = set()
    edges_remove.update(x for x in powerset(max_edges[edge_index], min_size) if x not in set_missing) #and also not in max_edges
    max_to_remove = min(len(edges_remove), len(set_missing))
    print(max_to_remove)
    print(list(edges_remove)[0])
    print(len(edges_remove))
    count = 0
    success_add = []
    success_delete = []
    for i in range(max_to_remove):
        #H.add_edge(list(set_missing)[i], id="rewired_edge")
        for id, edge in H.edges.members(dtype=dict).items():
            #print(edge)
            #print(list(edges_remove)[i])
            if (edge == set(list(edges_remove)[i])):
                H.remove_edge(id)
                print(id)
                H.add_edge(list(set_missing)[i], id="rewired_edge" + str(count))
                count += 1
                success_add.append(list[edge])
                success_delete.append(list(set_missing)[i])
    # for remove_edge in edges_remove:
    #     for id, edge in H.edges.members(dtype=dict).items():
    #         if set(edge) == set(remove_edge):
    #             H.remove_edge(id)
    #             print(id)
    #             H.add_edge(list(set_missing)[count], id="rewired_edge")
    #             count += 1
    #             success.append(list(edge))
    print(count)
    print(len(success_add))
    print(len(success_delete))
    #print(edges_remove)
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