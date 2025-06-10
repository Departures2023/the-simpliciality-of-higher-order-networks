from itertools import combinations
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


# Function to generate a list of n random numbers, each at least min_value, such that their sum is target_sum.
# All input should be integers
def generate_edge_distribution(min_edge_num, C_distribution, target_sum):
    """
    Generate a list of n random numbers, each at least min_value, such that their sum is target_sum.
    """
    # length of the actual edge distribution equals the number of maximal hyperedges
    length = len(C_distribution)
    
    if target_sum > C_distribution.sum() or target_sum < length * min_edge_num:
        raise ValueError("Impossible to generate numbers: Value Error.")

    # list of n min_value numbers
    edge_distribution = [min_edge_num] * length
    remaining = target_sum - length * min_edge_num
    
    for i in range(int(remaining)):
        # exclude indices where the number is already equal to the corresponding C_distribution value
        idx_exclude = [i for i in range(length) if edge_distribution[i] == C_distribution[i]]
        idx = random.choice(list(set([x for x in range(0, length)]) - set(idx_exclude)))
        edge_distribution[idx] += 1

    # No need for fractopnal part in this case, as we are generating integers
    # # If remaining is not integer, distribute the fractional part
    # frac = remaining - int(remaining)
    # if frac > 0:
    #     idx = random.randint(0, length - 1)
    #     edge_distribution[idx] += frac

    return edge_distribution

def all_possible_edges(arr_node):
    lst = []
    for i in range(2, len(arr_node) + 1):
        lst.append(list(combinations(arr_node, i)))
    return sum(lst, [])


#num_maxhyperedge???
def edge_rewire_model(edit_simpliciality, approx_num_C, num_max_hyperedge, num_node, min_size=2, max_size=None):
    # # Checking if input parameters are valid
    # if C_max > possible_combinations(num_node, min_size, max_size):
    #     raise ValueError("C_max is too large for the number of nodes and the specified min/max size.")
    # edit_simpliciality_fraction = Fraction(edit_simpliciality).limit_denominator(max_denominator=C_max)

    # |C| of the graph
    C_total = int(approx_num_C)
    
    # |H| of the graph
    edge_total = int(approx_num_C * edit_simpliciality)
    
    # Generate empty hypergraph
    H = xgi.Hypergraph()
    # Fill the hypergraph with nodes
    nodes = [i for i in range(num_node)]
    H.add_nodes_from(nodes)
    # Calculate the average number of induced hyperedges
    C_avg = C_total / num_max_hyperedge
    
    print("edge_total:", edge_total)
    print("C_total:", C_total)
    print("num_max_hyperedge:", num_max_hyperedge)
    print("C_avg:", C_avg)
    
    # Q1: HOW TO CHOOSE THE VALUE OF STANDARD DEVIATION?
    std = 0.5 * C_avg
    # Q2: CHECK IF I USE THE RIGHT EQUATION FOR UPPER AND LOWER BOUND?
    adjusted_lower = (min_size - C_avg) / std
    adjusted_higher = (max_size - C_avg) / std
    
    # Q3: NEED IMPROVEMENT - BETTER DISTRIBUTION METHOD?
    # Generate the distribution of C values (Union of powerset(maximal hyperedges))
    C_distribution = scipy.stats.truncnorm.rvs(
        adjusted_lower, 
        adjusted_higher, 
        loc=C_avg, 
        scale=std,
        size=num_max_hyperedge
    )
    print("C_distribution before rounding:", C_distribution)
    C_distribution = np.round(C_distribution).astype(int)
    # Ensure that the C_distribution values are within the specified min and max size
    
    # Generate the distribution of numbers of edges actually connected
    edge_distribution  = generate_edge_distribution(
        min_edge_num=min_size, 
        C_distribution=C_distribution, 
        target_sum=edge_total
    )
    
    # Convert the distribution of C values to the number of nodes in maximal hyperedges
    maximal_edge_size_list = [combination_to_size(i) for i in C_distribution]
    
    for i in range(num_max_hyperedge):
        #TODO: ADD MAXIMAL HYPEREDGE BEFORE HAND
        # Randomly select nodes for the maximal hyperedge
        selected_nodes = random.sample(nodes, maximal_edge_size_list[i])
        print("selected_nodes:", selected_nodes)
        # Add the maximal hyperedge to the hypergraph
        H.add_edge(selected_nodes)
        
        # Add edges to the hyperedge according to the edge distribution
        possible_edges = all_possible_edges(selected_nodes)
        possible_edge_idx = list(range(len(possible_edges)))
        print("possible_edges:", possible_edges)
        print("possible_edge_idx:", possible_edge_idx)
        print("edge_distribution:", edge_distribution)
        print("edge_distribution[i]:", edge_distribution[i])
        selected_edge_idx = random.sample(possible_edge_idx, edge_distribution[i])
        for idx in selected_edge_idx:
            print("selected_edge_idx:", idx)
            print("Adding edge:", possible_edges)
            H.add_edge(possible_edges[idx])
            
    return H
    
    


if __name__ == "__main__":
    print("Starting edge rewiring experiments...")
    print(Fraction(78).limit_denominator())
    print(Fraction(0.78).limit_denominator())
    
    combination_to_size