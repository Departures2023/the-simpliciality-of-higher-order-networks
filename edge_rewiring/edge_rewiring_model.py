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
    # Avoid unexpected min_size and max_size values
    if min_size < 2:
        min_size = 2
    if max_size > num_node:
        max_size = num_node

    sum = 0
    # Calculate the sum of combinations for sizes from min_size to max_size
    for i in range(min_size, max_size + 1):
        sum += math.comb(num_node, i)
    return sum

# Function to convert the number of combinations (edges) to the number of nodes
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
def generate_C_distribution(min_size, max_size, C_avg, num_max_hyperedge, target_sum):
    """
    Generate a list of n random numbers, each at least min_value, such that their sum is target_sum.
    """
    
    # Q1: HOW TO CHOOSE THE VALUE OF STANDARD DEVIATION?
    std = 0.5 * C_avg
    # Q2: CHECK IF I USE THE RIGHT EQUATION FOR UPPER AND LOWER BOUND?
    adjusted_lower = (min_size - C_avg) / std
    adjusted_higher = (max_size - C_avg) / std
    
    # length of the actual edge distribution equals the number of maximal hyperedges
    C_distribution = scipy.stats.truncnorm.rvs(
        adjusted_lower, 
        adjusted_higher, 
        loc=C_avg, 
        scale=std,
        size=num_max_hyperedge
    )
    # Round the distribution to integers
    C_distribution = np.round(C_distribution).astype(int)

    # Check if the sum of the generated distribution is bigger to the target sum
    excess = C_distribution.sum() - target_sum
    if excess > 0:
        for i in range(int(excess)):
            # exclude indices where the number is already equal to the corresponding C_distribution value
            idx_exclude = [i for i in range(num_max_hyperedge) if min_size == C_distribution[i]]
            idx = random.choice(list(set([x for x in range(0, num_max_hyperedge)]) - set(idx_exclude)))
            C_distribution[idx] -= 1
        return C_distribution
    elif excess < 0:
        excess = abs(excess)
        for i in range(int(excess)):
            # exclude indices where the number is already equal to the corresponding C_distribution value
            idx_exclude = [i for i in range(num_max_hyperedge) if max_size == C_distribution[i]]
            idx = random.choice(list(set([x for x in range(0, num_max_hyperedge)]) - set(idx_exclude)))
            C_distribution[idx] += 1
        return C_distribution
    else:
        # If the sum is equal to the target sum, return the distribution
        return C_distribution
    



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
    
    # Distribute the remaining value across the edge_distribution
    if remaining > 0:
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

# Function to generate all possible edges from a list of nodes
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
    
    
    # Q3: NEED IMPROVEMENT - BETTER DISTRIBUTION METHOD?
    # Generate the distribution of C values (Union of powerset(maximal hyperedges))
    C_distribution = generate_C_distribution(
        min_size=min_size, 
        max_size=max_size, 
        C_avg=C_avg, 
        num_max_hyperedge=num_max_hyperedge, 
        target_sum=C_total
    )
    print("C_distribution:", C_distribution)
    
    # Generate the distribution of numbers of edges actually connected
    edge_distribution  = generate_edge_distribution(
        min_edge_num=min_size, 
        C_distribution=C_distribution, 
        target_sum=edge_total
    )
    print("edge_distribution:", edge_distribution)
    # Convert the distribution of C values to the number of nodes in maximal hyperedges
    maximal_edge_size_list = [combination_to_size(i) for i in C_distribution]
    print("maximal_edge_size_list:", maximal_edge_size_list)
    final_possible_edge_list = []
    for i in range(num_max_hyperedge):
        # Randomly select nodes for the maximal hyperedge
        selected_nodes = random.sample(nodes, maximal_edge_size_list[i])
        # Generate the powerset of the selected nodes (possible edges to add for adjustment)
        tmp_list = powerset(selected_nodes, 2)
        list_of_sets = [set(item) for item in list(tmp_list)]
        final_possible_edge_list.append(list_of_sets)
        
        
        print("selected_nodes:", selected_nodes)
        print("final_possible_edge_list:", final_possible_edge_list)
        
        # Add the maximal hyperedge to the hypergraph
        H.add_edge(selected_nodes)
        
        # Add edges to the hyperedge according to the edge distribution
        possible_edges = all_possible_edges(selected_nodes)
        possible_edge_idx = list(range(len(possible_edges)))
        # Print statements for debugging
        # print("possible_edges:", possible_edges)
        # print("possible_edge_idx:", possible_edge_idx)
        # print("edge_distribution:", edge_distribution)
        # print("edge_distribution[i]:", edge_distribution[i])
        selected_edge_idx = random.sample(possible_edge_idx, edge_distribution[i])
        for idx in selected_edge_idx:
            print("selected_edge_idx:", idx)
            print("Adding edge:", possible_edges)
            H.add_edge(possible_edges[idx])
    # Final adjustment of the hypergraph
    edges = H.edges.filterby("size", min_size, "geq").members()
    H = final_edge_adjustment(
        H, 
        edges, 
        final_possible_edge_list, 
        expected_es=edit_simpliciality
    )
    return H
    
# Function to slightly adjust the hypergraph to match the expected edit simpliciality
def final_edge_adjustment(H, edges, final_possible_edge_list, expected_es):
    # Calculate the current edit simpliciality
    curr_es = edit_simpliciality(H, min_size=2)
    # Split to cases to add or remove edges respectively
    if curr_es < expected_es:
        # Add edges to the hypergraph
        for i in range(len(final_possible_edge_list)):
            tmp_idx = random.randint(0, len(final_possible_edge_list) - 1)
            tmp_add = final_possible_edge_list.pop(tmp_idx)
            # tmp_add is a list of sets representing possible edges
            for edge_set in tmp_add:
                if edge_set not in [set(e) for e in edges]:
                    H.add_edge(list(edge_set))
                    curr_es = edit_simpliciality(H, min_size=2)
                    if curr_es >= expected_es:
                        return H
    elif curr_es > expected_es:
        # Remove edges from the hypergraph untul the edit simpliciality is equal to the expected value
        while curr_es > expected_es:
            tmp_remove = edges[random.randint(0, len(edges) - 1)]
            for id, edge in H.edges.members(dtype=dict).items():
                    if (edge == tmp_remove):
                        H.remove_edge(id)
                        curr_es = edit_simpliciality(H, min_size=2)
        return H
    else:
        return H
                    


# Adjust Main function if needed
# if __name__ == "__main__":
#     print("Starting edge rewiring experiments...")
#     print(Fraction(78).limit_denominator())
#     print(Fraction(0.78).limit_denominator())
    
#     combination_to_size