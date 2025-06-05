import numpy as np
import xgi
from ..trie import Trie
from .utilities import missing_subfaces, powerset

def h1():
    return xgi.Hypergraph([{1, 2, 3}, {2, 3, 4, 5}, {5, 6, 7}, {5, 6}])

def non_simplex_maximal_edges(H, min_size=2, exclude_min_size=True):
    #filters all edges so that size is greater than or equal to min size 
    edges = H.edges.filterby("size", min_size, "geq").members()
    #excludes edges smaller or equal to than min size + 1
    max_edges = (
        H.edges.maximal().filterby("size", min_size + exclude_min_size, "geq").members()
    )
    #creates trie from edges
    t = Trie()
    t.build_trie(edges)

    print(t)
    return H

dataset = "email-enron"
max_order = 5
min_size = 2
H = xgi.load_xgi_data(dataset, max_order=max_order)
H.cleanup(singletons=True)
non_simplex_maximal_edges(H, 1, True)