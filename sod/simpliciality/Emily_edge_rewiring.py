#TO RUN: python -m sod.simpliciality.Emily_edge_rewiring  

import numpy as np
import xgi
from ..trie import Trie
from .utilities import missing_subfaces, powerset
from itertools import chain, combinations


def h1():
    return xgi.Hypergraph([{1, 2, 3}, {2, 3, 4, 5}, {5, 6, 7}, {5, 6}])

def Alg_1(H, min_size=2, exclude_min_size=True):
    #gets all edges from hypergraph
    edges = H.edges.members()
    print("made edges")
    pedges = powerset(edges, min_size, None)
    print("made powerset")
    #sorted_edges = sorted(edges, key=len, reverse=True)           don't need
    edge_index = 0
    print("getting to while loop")
    print("edge:", edges[edge_index])
    #for e in pedges:
    #    print(list(e))

    while(len(edges[edge_index]) < 1):
        print("edge_index:", edge_index)
        edge_index += 1
        print("edge_index:", edge_index)
    e = edges[edge_index]
    pe = set(powerset(e, min_size, None))

    if((pe - e) != 0):
        e1 = edge(random.randint(0, len(edge)), )
        print(pe)
        print("entered")
    print("returning")
    return H

dataset = "email-enron"
max_order = 5
min_size = 2
H = xgi.load_xgi_data(dataset, max_order=max_order)
H.cleanup(singletons=True)
Alg_1(H, 1, True)