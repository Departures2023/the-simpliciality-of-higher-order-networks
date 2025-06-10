import math
import sys
import os

import scipy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import threading
import time
import numpy as np
import xgi
from fractions import Fraction

from sod.simpliciality import edit_simpliciality, face_edit_simpliciality, simplicial_fraction
from sod.trie import Trie
from sod.simpliciality.utilities import missing_subfaces, powerset

# Function to calculate the number of possible combinations of nodes -> possible edges
def possible_combinations(num_node, min_size=2, max_size=None):
    if min_size < 2:
        min_size = 2
    if max_size > num_node:
        max_size = num_node

    sum = 0
    for i in range(min_size, max_size + 1):
        sum += math.comb(num_node, i)
    return sum

def combination_to_size(C):
    num_node = 2
    while C > possible_combinations(num_node=num_node, min_size=2, max_size=num_node):
        num_node += 1
    C_next = possible_combinations(num_node=num_node+1, min_size=2, max_size=num_node)
    C_current = possible_combinations(num_node=num_node, min_size=2, max_size=num_node)
    # If the difference berween next combination and expected is smaller than the current and expected, we can add one more node
    # This can compensate the negative effect of calculated combinations being smaller than the expected combinations
    if (C_next - C)  < (C - C_current):
        num_node += 1
    return num_node

#num_maxhyperedge???
def edge_rewire_model(edit_simpliciality, C_max, num_max_hyperedge, num_node, min_size=2, max_size=None):
    # Checking if input parameters are valid
    if C_max > possible_combinations(num_node, min_size, max_size):
        raise ValueError("C_max is too large for the number of nodes and the specified min/max size.")
    edit_simpliciality_fraction = Fraction(edit_simpliciality).limit_denominator(max_denominator=C_max)
    # |H| of the graph
    edge_total = edit_simpliciality_fraction.numerator
    # |C| of the graph
    C_total = edit_simpliciality_fraction.denominator
    # Generate empty hypergraph
    H = xgi.Hypergraph()
    # Fill the hypergraph with nodes
    nodes = [i for i in range(num_node)]
    H.add_nodes_from(nodes)
    # Calculate the average number of induced hyperedges
    C_avg = C_total / num_max_hyperedge
    
    # Q1: HOW TO CHOOSE THE VALUE OF STANDARD DEVIATION?
    std = 0.5 * C_avg
    # Q2: CHECK IF I USE THE RIGHT EQUATION FOR UPPER AND LOWER BOUND?
    adjusted_lower = (min_size - C_avg) / std
    adjusted_higher = (max_size - C_avg) / std
    
    distribution = scipy.stats.truncnorm(
        adjusted_lower, 
        adjusted_higher, 
        loc=C_avg, 
        scale=std,
        size=num_max_hyperedge
    )
    num_node_avg = [combination_to_size(i) for i in distribution]
    
    


if __name__ == "__main__":
    print("Starting edge rewiring experiments...")
    print(Fraction(78).limit_denominator())
    print(Fraction(0.78).limit_denominator())
    
    combination_to_size