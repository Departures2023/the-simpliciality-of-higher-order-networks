import copy
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
import numpy as np

from sod.simpliciality import edit_simpliciality, face_edit_simpliciality, simplicial_fraction, new_edit_simpliciality
from sod.trie import Trie
from sod.simpliciality.utilities import missing_subfaces, powerset

# Function to calculate the number of possible combinations of nodes -> possible edges
def possible_combinations(num_node, min_size=2, max_size=None):
    """
    Calculate the number of possible combinations of nodes -> possible edges

    Args:
        num_node (int): number of nodes
        min_size (int): min size of an edge. Defaults to 2.
        max_size (int): max size of an edge. Defaults to None.

    Returns:
        int: possible combinations of nodes -> possible edges
    """
    # Avoid unexpected min_size and max_size values
    if min_size < 2:
        min_size = 2
    if max_size is None:
        max_size = num_node
    if max_size > num_node:
        max_size = num_node

    sum = 0
    # Calculate the sum of combinations for sizes from min_size to max_size
    for i in range(min_size, max_size + 1):
        sum += math.comb(num_node, i)
    return sum

# Function to convert the number of combinations (edges) to the number of nodes
def combination_to_size(C):
    """
    Convert the number of combinations (edges) to the number of nodes

    Args:
        C (int): Number of combinations (edges) within a maximal hyperedge

    Returns:
        int: Minimum number of nodes needed in the maximal hyperedge
    """
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
def generate_C_distribution(min_size, max_size, C_avg, std, num_max_hyperedge, target_sum):
    """
    Generate a list of n random numbers, each at least min_value, such that their sum is target_sum.

    Args:
        min_size (int): min size of an edge
        max_size (int): max size of an edge
        C_avg (float): average number of combinations (induced edges) within a maximal hyperedge
        num_max_hyperedge (int): number of maximal hyperedges
        target_sum (int): total number of combinations (induced edges) within all maximal hyperedges

    Returns:
        array: distribution of C values
    """
    
    # Use lognormal distribution which is better for positive values
    
    # For lognormal distribution with desired mean C_avg:
    # mean_lognormal = exp(mu + std^2/2)
    # So: mu = ln(mean_lognormal) - std^2/2
    mu = np.log(C_avg) - (std**2 / 2)
    
    # Generate lognormal distribution
    C_distribution = np.random.lognormal(mean=mu, sigma=std, size=num_max_hyperedge)
    
    # Round to integers and clip to bounds
    C_distribution = np.round(C_distribution).astype(int)
    C_distribution = np.clip(C_distribution, min_size, max_size)
    C_distribution.sort()
    
    #print("C_distribution:", C_distribution)

    # Adjust the sum to match target_sum using lognormal PDF for weighting
    excess = C_distribution.sum() - target_sum
    print("excess:", excess)
    
    if excess > 0:
        # Reduce values that are above minimum
        for _ in range(int(abs(excess))):
            reducible_indices = [i for i in range(num_max_hyperedge) if C_distribution[i] > min_size]
            if len(reducible_indices) == 0:
                break
                
            # Calculate PDF values for reducible elements
            reducible_values = C_distribution[reducible_indices]
            pdf_values = scipy.stats.lognorm.pdf(reducible_values, s=std, scale=np.exp(mu))
            
            # lower PDF = higher chance of being selected for reduction
            if np.sum(pdf_values) > 0:
                inverse_weights = 1.0 / (pdf_values + 1e-10)  # Add small epsilon to avoid division by zero
                weights = inverse_weights / np.sum(inverse_weights)
                selected_idx = np.random.choice(reducible_indices, p=weights)
            else:
                # Fallback to random if all PDFs are zero
                selected_idx = random.choice(reducible_indices)
            
            C_distribution[selected_idx] -= 1
            
    elif excess < 0:
        # Increase values that are below maximum
        for _ in range(int(abs(excess))):
            increasable_indices = [i for i in range(num_max_hyperedge) if C_distribution[i] < max_size]
            if len(increasable_indices) == 0:
                break
            # Calculate PDF values for increasable elements  
            increasable_values = C_distribution[increasable_indices] + 1  # +1 because we're considering the increased value
            pdf_values = scipy.stats.lognorm.pdf(increasable_values, s=std, scale=np.exp(mu))
            
            # higher PDF = higher chance of being selected for increase
            if np.sum(pdf_values) > 0:
                weights = pdf_values / np.sum(pdf_values)
                selected_idx = np.random.choice(increasable_indices, p=weights)
            else:
                # Fallback to random if all PDFs are zero
                selected_idx = random.choice(increasable_indices)
            
            C_distribution[selected_idx] += 1
    # print("C_distribution:", C_distribution)
    return C_distribution




# Function to generate a list of n random numbers, each at least min_value, such that their sum is target_sum.
# All input should be integers
def generate_edge_distribution(min_edge_num, C_distribution, target_sum):
    """
    Generate a list of n random numbers, each at least min_value, such that their sum is target_sum.
    """
    # length of the actual edge distribution equals the number of maximal hyperedges
    length = len(C_distribution)
    
    # CHOICE2: RETURN adjusted list
    if target_sum > C_distribution.sum():
        print(f"❌ Warning: target_sum ({target_sum}) is larger than C_distribution.sum() ({C_distribution.sum()})")
        print(f"Adjusting target_sum to {C_distribution.sum()}")
        target_sum = C_distribution.sum()

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

    return edge_distribution

# Function to generate all possible edges from a list of nodes
def all_possible_edges(arr_node):
    lst = []
    for i in range(2, len(arr_node) + 1):
        lst.append(list(combinations(arr_node, i)))
    return sum(lst, [])




# Function to generate a hypergraph with a given edit simpliciality, number of maximal hyperedges, and number of nodes
def model_generation_es(es, approx_num_C, num_max_hyperedge, num_node, min_size=2, max_size=None, adjust_es=False, compare_interval_smaller_case=2, compare_interval_bigger_case=2, C_distribution=None):
    # Checking if input parameters are valid
    if max_size is None:
        max_size = num_node
        
    if approx_num_C < num_max_hyperedge:
        print(f"❌ Warning: approx_num_C ({approx_num_C}) is smaller than num_max_hyperedge ({num_max_hyperedge})")
        print(f"Adjusting approx_num_C to {num_max_hyperedge}")
        approx_num_C = num_max_hyperedge
        
    # # Calculate the maximum possible combinations for the given parameters
    # max_possible_C = approximate_C_upperbound(num_node, min_size, max_size, num_max_hyperedge)
    
    # if approx_num_C > max_possible_C:
    #     print(f"❌ Warning: approx_num_C ({approx_num_C}) is larger than maximum possible combinations ({max_possible_C})")
    #     print(f"Adjusting approx_num_C to {max_possible_C}")
    #     approx_num_C = max_possible_C

    # |C| of the graph
    C_total = int(approx_num_C)
    
    # |H| of the graph
    edge_total = int(approx_num_C * es)
    
    # TODO: Check the performance of model generation with and without adjust_es
    # New implementation of edit simpliciality (es = (|E| - num_max_hyperedge)/(|C| - num_max_hyperedge))
    if adjust_es:
        edge_total = int((C_total - num_max_hyperedge) * es) + num_max_hyperedge
        if not (C_total >= edge_total and edge_total >= num_max_hyperedge):
            print(f"❌ Warning: C_total ({C_total}) is smaller than edge_total ({edge_total}) or edge_total ({edge_total}) is smaller than num_max_hyperedge ({num_max_hyperedge})")
            sys.exit()
    
    
    # Generate empty hypergraph
    H = xgi.Hypergraph()
    # Fill the hypergraph with nodes
    nodes = [i for i in range(num_node)]
    
    #TODO: Wait for implementation
    weight_node = [1] * len(nodes)
    
    H.add_nodes_from(nodes)
    # Calculate the average number of induced hyperedges
    C_avg = C_total / num_max_hyperedge
    if es < 0.15:
        std = 5 #3
    elif es < 0.5:
        std = 5 #2
    elif es < 0.85:
        std = 7.5 #1
    else:
        std = 5 #0.5
    # Print statements for debugging
    # print("edge_total:", edge_total)
    # print("C_total:", C_total)
    # print("num_max_hyperedge:", num_max_hyperedge)
    # print("C_avg:", C_avg)
    
    
    # Q3: NEED IMPROVEMENT - BETTER DISTRIBUTION METHOD?
    # Generate the distribution of C values (Union of powerset(maximal hyperedges))
    if C_distribution is None:
        C_distribution = generate_C_distribution(
            min_size=possible_combinations(min_size), 
            max_size=C_total, 
            C_avg=C_avg, 
            std=std,
            num_max_hyperedge=num_max_hyperedge, 
            target_sum=C_total
        )
    else:
        C_distribution = C_distribution
    # Print statements for debugging
    print("C_distribution:", C_distribution)
    
    # Generate the distribution of numbers of edges actually connected
    edge_distribution  = generate_edge_distribution(
        min_edge_num=possible_combinations(min_size), 
        C_distribution=C_distribution, 
        target_sum=edge_total
    )
    
    # Print statements for debugging
    # print("edge_distribution:", edge_distribution)
    
    # Convert the distribution of C values to the number of nodes in maximal hyperedges
    maximal_edge_size_list = [combination_to_size(i) for i in C_distribution]
    maximal_edge_size_list.sort(reverse=True)
    #print("maximal_edge_size_list:", maximal_edge_size_list)
    # Avoid adding repeating edges - use set for consistent comparison
    edge_to_exclude = set()
    # Print statements for debugging
    # print("maximal_edge_size_list:", maximal_edge_size_list)
    
    maximal_edge_set = set()
    final_possible_edge_list = []
    used_nodes = set()
    unused_nodes = set(nodes)
    
    for i in range(num_max_hyperedge):
        # Initialize the range and weights for the random selection of size of maximal hyperedge
        if i == 0:
            selected_nodes = random.sample(nodes, maximal_edge_size_list[i])
            selected_nodes_set = frozenset(selected_nodes)  # Convert to frozenset for consistent comparison
        else:
            # Handle the case when unused_nodes becomes empty
            if len(unused_nodes) == 0:
                # All nodes are used, just sample from used_nodes
                selected_nodes = random.sample(list(used_nodes), maximal_edge_size_list[i])
                selected_nodes_set = frozenset(selected_nodes)
            else:
                size_lst = range(1, maximal_edge_size_list[i])
                weights = []
                for size in size_lst:
                    weights.append(1/size)
                # Determine the number of nodes to chosen from used and not used
                used_selected_size = random.choices(size_lst, weights=weights, k=1)[0]
                unused_selected_size = maximal_edge_size_list[i] - used_selected_size
                
                # Safety checks to prevent sampling more than available
                unused_selected_size = min(unused_selected_size, len(unused_nodes))
                used_selected_size = min(used_selected_size, len(used_nodes))
                
                # Ensure we have enough nodes total
                total_needed = maximal_edge_size_list[i]
                total_available = unused_selected_size + used_selected_size

                if total_available < total_needed:
                    # Adjust by taking more from the larger pool
                    deficit = total_needed - total_available
                    if len(unused_nodes) - unused_selected_size >= deficit:
                        unused_selected_size += deficit
                    elif len(used_nodes) - used_selected_size >= deficit:
                        used_selected_size += deficit
                    else:
                        # Not enough nodes available, sample what we can
                        unused_selected_size = len(unused_nodes)
                        used_selected_size = min(len(used_nodes), total_needed - unused_selected_size)
                
                # Randomly select nodes for the maximal hyperedge
                unused_selected_nodes = random.sample(list(unused_nodes), unused_selected_size)
                used_selected_nodes = random.sample(list(used_nodes), used_selected_size)
                # print("len(unused_selected_nodes)", len(unused_selected_nodes))
                # print("len(used_selected_nodes)", len(used_selected_nodes))
                # print("xgi.number_connected_components(H)", xgi.number_connected_components(H))
                selected_nodes = list(unused_selected_nodes) + list(used_selected_nodes)
                selected_nodes_set = frozenset(selected_nodes)  # Convert to frozenset for consistent comparison
        
        # Avoid adding repeating nodes and make sure the selected nodes are not a subset of any existing maximal hyperedge
        # Fixed logic: use OR instead of AND, and fix subset comparison
        while (selected_nodes_set in edge_to_exclude or any(selected_nodes_set.issubset(existing_edge) for existing_edge in maximal_edge_set)):
            if i == 0:
                selected_nodes = random.sample(nodes, maximal_edge_size_list[i])
                selected_nodes_set = frozenset(selected_nodes)  # Convert to frozenset for consistent comparison
            else:
                # Handle the case when unused_nodes becomes empty
                if len(unused_nodes) == 0:
                    # All nodes are used, just sample from used_nodes
                    selected_nodes = random.sample(list(used_nodes), maximal_edge_size_list[i])
                    selected_nodes_set = frozenset(selected_nodes)
                else:
                    size_lst = range(1, maximal_edge_size_list[i])
                    weights = []
                    for size in size_lst:
                        weights.append(1/size)
                    # Determine the number of nodes to chosen from used and not used
                    used_selected_size = random.choices(size_lst, weights=weights, k=1)[0]
                    unused_selected_size = maximal_edge_size_list[i] - used_selected_size
                    
                    # Safety checks to prevent sampling more than available
                    unused_selected_size = min(unused_selected_size, len(unused_nodes))
                    used_selected_size = min(used_selected_size, len(used_nodes))
                    
                    # Ensure we have enough nodes total
                    total_needed = maximal_edge_size_list[i]
                    total_available = unused_selected_size + used_selected_size
                    
                    if total_available < total_needed:
                        # Adjust by taking more from the larger pool
                        deficit = total_needed - total_available
                        if len(unused_nodes) - unused_selected_size >= deficit:
                            unused_selected_size += deficit
                        elif len(used_nodes) - used_selected_size >= deficit:
                            used_selected_size += deficit
                        else:
                            # Not enough nodes available, sample what we can
                            unused_selected_size = len(unused_nodes)
                            used_selected_size = min(len(used_nodes), total_needed - unused_selected_size)
                    
                    # Randomly select nodes for the maximal hyperedge
                    unused_selected_nodes = random.sample(list(unused_nodes), unused_selected_size)
                    used_selected_nodes = random.sample(list(used_nodes), used_selected_size)
                    selected_nodes = list(unused_selected_nodes) + list(used_selected_nodes)
                    selected_nodes_set = frozenset(selected_nodes)  # Convert to frozenset for consistent comparison
            
        # Add the maximal hyperedge to the hypergraph
        H.add_edge(selected_nodes)
        maximal_edge_set.add(selected_nodes_set)
        edge_to_exclude.add(selected_nodes_set)
        used_nodes.update(selected_nodes)
        unused_nodes.difference_update(selected_nodes)
        # print("len(unused_nodes)", len(unused_nodes))
        # print("len(used_nodes)", len(used_nodes))
        # Generate the powerset of the selected nodes (possible edges to add for adjustment)
        tmp_list = powerset(selected_nodes, 2, len(selected_nodes) - 1)
        possible_edges = [frozenset(item) for item in list(tmp_list)]  # Use frozenset for consistent comparison
        possible_edges_copy = copy.deepcopy(possible_edges)
        
        # Print statements for debugging
        # print("selected_nodes:", selected_nodes)
        # print("final_possible_edge_list:", final_possible_edge_list)
        
        possible_edge_idx = []
        # Only add the non-repeating edges
        for j in range(len(possible_edges)):
            if (possible_edges[j] not in edge_to_exclude):
                possible_edge_idx.append(j)

        # Print statements for debugging
        # print("possible_edges:", possible_edges)
        # print("possible_edge_idx:", possible_edge_idx)
        # print("edge_distribution:", edge_distribution)
        # print("edge_distribution[i]:", edge_distribution[i])
        
        # Avoid the case that edge_distribution[i] is bigger than len(possible_edge_idx) after repeated nodes are deleted
        num_edges_to_add = min(edge_distribution[i], len(possible_edge_idx))
        if num_edges_to_add > 0:
            selected_edge_idx = random.sample(possible_edge_idx, num_edges_to_add)
            for idx in selected_edge_idx:
                # Print statements for debugging
                # print("selected_edge_idx:", idx)
                # print("Adding edge:", possible_edges[idx])
                H.add_edge(list(possible_edges[idx]))  # Convert frozenset back to list for adding to hypergraph
                edge_to_exclude.add(possible_edges[idx])
                possible_edges_copy.remove(possible_edges[idx])

        final_possible_edge_list.append(possible_edges_copy)
        
    # Final adjustment of the hypergraph
    

    # print("new edges:", len(edges))
    # print("xgi.number_connected_components(H)", xgi.number_connected_components(H))
    H = final_edge_adjustment_es(
        H, 
        min_size,
        maximal_edge_set, 
        final_possible_edge_list, 
        edge_to_exclude=edge_to_exclude,
        expected_es=es,
        adjust_es=adjust_es,
        compare_interval_smaller_case=compare_interval_smaller_case,
        compare_interval_bigger_case=compare_interval_bigger_case
    )
    print("H.num_nodes", H.num_nodes)
    print("H.num_edges", H.num_edges)
    print("H.num_edges without singletons", len(H.edges.filterby("size", 2, "geq").members()))
    print("len(H.edges.maximal())", len(H.edges.maximal().filterby("size", 2, "geq").members()))
    
    if xgi.number_connected_components(H) > 1:
        print(f"❌ Warning: The generated hypergraph has {xgi.number_connected_components(H)} connected components. Input parameters are not good, please check the input parameters")
    return H
    
# Function to slightly adjust the hypergraph to match the expected edit simpliciality
def final_edge_adjustment_es(H, min_size, maximal_edge_set, final_possible_edge_list, edge_to_exclude, expected_es, adjust_es=False, compare_interval_smaller_case = 2, compare_interval_bigger_case = 2):
    #print("final_possible_edge_list:", len(final_possible_edge_list))
    # Calculate the current edit simpliciality
    if adjust_es:
        curr_es = new_edit_simpliciality(H, min_size=2)
    else:
        curr_es = edit_simpliciality(H, min_size=2)
    # Use count to increase efficiency
    count = 0
    # Split to cases to add or remove edges respectively
    if curr_es < expected_es:
        # Add edges to the hypergraph
        for i in range(len(final_possible_edge_list)):
            tmp_idx = random.randint(0, len(final_possible_edge_list) - 1)
            tmp_add = final_possible_edge_list.pop(tmp_idx)
            # tmp_add is a list of frozensets representing possible edges
            for edge_set in tmp_add:
                # Check if edge is already excluded to avoid duplicates
                if edge_set not in edge_to_exclude:
                    H.add_edge(list(edge_set))
                    edge_to_exclude.add(edge_set)  # Track the added edge
            count += 1
            #print("final_possible_edge_list:", len(final_possible_edge_list))
            # Check if the edit simpliciality is close to the expected value only every 2 iterations
            if count >= compare_interval_smaller_case:
                count = 0
                if adjust_es:
                    curr_es = new_edit_simpliciality(H, min_size=2)
                else:
                    curr_es = edit_simpliciality(H, min_size=2)
                print("curr_es:", curr_es)
                # if curr_es >= expected_es:
                if (abs(curr_es - expected_es) < 0.002):
                    return H
                if (curr_es >= expected_es):
                    break
                
    edges = H.edges.filterby("size", min_size, "geq").members()
    edges = [edge for edge in edges if frozenset(edge) not in maximal_edge_set]
    if curr_es > expected_es:
        # Remove edges from the hypergraph untul the edit simpliciality is equal to the expected value
        edge_id_map = {}
        for edge_id, edge_members in H.edges.members(dtype=dict).items():
            edge_id_map[frozenset(edge_members)] = edge_id
        while ((curr_es > expected_es) or (abs(curr_es - expected_es) > 0.002)) and len(edges) > 0:
            tmp_remove_idx = random.randint(0, len(edges) - 1)
            tmp_remove = edges[tmp_remove_idx]
            H.remove_edge(edge_id_map[frozenset(tmp_remove)])
            # Remove the edge from the edges list to avoid trying to remove it again
            edges.pop(tmp_remove_idx)
            count += 1
            if count >= compare_interval_bigger_case:
                count = 0
                if adjust_es:
                    curr_es = new_edit_simpliciality(H, min_size=2)
                else:
                    curr_es = edit_simpliciality(H, min_size=2)
                print("curr_es:", curr_es)
        return H
    print(f"❌ Warning: Input parameters are not good, please check the input parameters")
    return H
                    


# Function to generate a hypergraph with a given simplicial fraction, number of edges, and number of nodes
def model_generation_sf(sf, approx_num_E, num_node, min_size=2, max_size=None):
    """
    Generate a hypergraph with a given simplicial fraction.
    
    Parameters:
    -----------
    sf : float
        Target simplicial fraction
    approx_num_E : int
        Approximate number of edges
    num_node : int
        Number of nodes
    min_size : int, default=2
        Minimum edge size
    max_size : int, default=None
        Maximum edge size
        
    Returns:
    --------
    xgi.Hypergraph
        Generated hypergraph
    """
    if max_size is None:
        max_size = num_node
    
    # Total number of edges in the graph
    edge_total = int(approx_num_E)
    
    # Total number of edges that are simplices in the graph
    S_total = int(approx_num_E * sf)
    
    # Generate empty hypergraph
    H = xgi.Hypergraph()
    # Fill the hypergraph with nodes
    nodes = [i for i in range(num_node)]
    H.add_nodes_from(nodes)
    
    # Track maximal simplices and non-simplices using sets for comparison
    maximal_edge_is_simplex = set()
    maximal_edge_not_simplex = set()
    edge_to_exclude = set()  # Track all edges to avoid duplicates
    final_possible_edge_list = []
    
    print(f"Target simplicial fraction: {sf}")
    print(f"Target total edges: {edge_total}")
    print(f"Target simplicial edges: {S_total}")
    
    # Phase 1: Add simplices to the hypergraph until the total number of simplices reaches S_total
    simplicial_edges_added = 0
    while S_total > 0 and simplicial_edges_added < S_total:
        # current number of edges in the hypergraph
        current_edges = H.edges.filterby("size", min_size, "geq").members()
        current_edge_sets = {frozenset(edge) for edge in current_edges}
        
        # Find the maximal size of the simplex that can be added
        remaining_simplices = S_total - simplicial_edges_added
        simplex_size_max = min(combination_to_size(remaining_simplices), max_size)
        
        # Randomly select the size of the simplex to be added
        curr_size = random.randint(min_size, simplex_size_max)
        
        # Randomly select nodes for the simplex
        selected_nodes = random.sample(nodes, curr_size)
        selected_nodes_set = frozenset(selected_nodes)
        
        # Avoid adding repeating simplices
        max_attempts = 100  # Prevent infinite loops
        attempts = 0
        while (selected_nodes_set in maximal_edge_is_simplex) and attempts < max_attempts:
            selected_nodes = random.sample(nodes, curr_size)
            selected_nodes_set = frozenset(selected_nodes)
            attempts += 1
            
        if attempts >= max_attempts:
            print(f"Warning: Could not find unique simplex after {max_attempts} attempts")
            break
            
        maximal_edge_is_simplex.add(selected_nodes_set)
        
        # Add all edges of the simplex that haven't been added to the hypergraph
        all_edges = list(powerset(selected_nodes, 2))
        new_simplicial_edges = []
        
        for edge in all_edges:
            edge_set = frozenset(edge)
            if edge_set not in current_edge_sets and edge_set not in edge_to_exclude:
                new_simplicial_edges.append(edge)
                edge_to_exclude.add(edge_set)
                
        # Add the new edges and update counters
        if new_simplicial_edges:
            H.add_edges_from(new_simplicial_edges)
            simplicial_edges_added += len(new_simplicial_edges)
            S_total -= len(new_simplicial_edges)
            
        # Break if we can't add more simplices
        if len(new_simplicial_edges) == 0:
            break
    
    print(f"Added {simplicial_edges_added} simplicial edges")
    
    # Phase 2: Add non-simplicial edges to reach the target total number of edges
    current_edge_count = len(H.edges.filterby("size", min_size, "geq"))
    remaining_edges_needed = edge_total - current_edge_count
    
    print(f"Current edges: {current_edge_count}, Need {remaining_edges_needed} more edges")
    
    while remaining_edges_needed > 0:
        # current number of edges in the hypergraph
        current_edges = H.edges.filterby("size", min_size, "geq").members()
        current_edge_sets = {frozenset(edge) for edge in current_edges}
        
        # Find the maximal size of the edge that can be added
        edge_size_max = min(combination_to_size(remaining_edges_needed), max_size)
        
        # Randomly select the size of the edge to be added
        curr_size = random.randint(min_size, edge_size_max)
        
        # Randomly select nodes for the maximal hyperedge
        selected_nodes = random.sample(nodes, curr_size)
        selected_nodes_set = frozenset(selected_nodes)
        
        # Avoid adding edges that would form complete simplices
        max_attempts = 100
        attempts = 0
        while (selected_nodes_set in maximal_edge_is_simplex) and attempts < max_attempts:
            selected_nodes = random.sample(nodes, curr_size)
            selected_nodes_set = frozenset(selected_nodes)
            attempts += 1
            
        if attempts >= max_attempts:
            print(f"Warning: Could not find non-simplicial edge after {max_attempts} attempts")
            break
            
        maximal_edge_not_simplex.add(selected_nodes_set)
        
        # Add the maximal hyperedge first
        if selected_nodes_set not in edge_to_exclude:
            H.add_edge(selected_nodes)
            edge_to_exclude.add(selected_nodes_set)
            remaining_edges_needed -= 1
        
        # Find potential sub-edges that can be added (but don't form complete simplices)
        all_sub_edges = list(powerset(selected_nodes, 2, len(selected_nodes) - 1))
        possible_edges = []
        
        for edge in all_sub_edges:
            edge_set = frozenset(edge)
            if edge_set not in current_edge_sets and edge_set not in edge_to_exclude:
                possible_edges.append(edge_set)
        
        if len(possible_edges) > 0 and remaining_edges_needed > 0:
            # Randomly choose some edges to add (but not all, to avoid forming simplices)
            max_edges_to_add = min(remaining_edges_needed, len(possible_edges) - 1) if len(possible_edges) > 1 else 0
            
            if max_edges_to_add > 0:
                num_edges_to_add = random.randint(0, max_edges_to_add)
                selected_edges = random.sample(possible_edges, num_edges_to_add)
                
                for edge_set in selected_edges:
                    H.add_edge(list(edge_set))
                    edge_to_exclude.add(edge_set)
                    remaining_edges_needed -= 1
                    
                # Store remaining edges for potential final adjustment
                remaining_edges = [edge for edge in possible_edges if edge not in selected_edges]
                if remaining_edges:
                    final_possible_edge_list.extend(remaining_edges)
        
        # Break if we can't add more edges
        if remaining_edges_needed > 0 and len(possible_edges) == 0:
            break
    
    current_edge_count = len(H.edges.filterby("size", min_size, "geq"))
    print(f"Final edge count: {current_edge_count}")
    
    # Calculate current simplicial fraction
    curr_sf = simplicial_fraction(H, min_size=2)
    print(f"Current simplicial fraction: {curr_sf}")
    
    if abs(curr_sf - sf) > 0.05:
        # Final adjustment of the hypergraph
        H = final_edge_adjustment_sf(
            H, 
            maximal_edge_is_simplex, 
            maximal_edge_not_simplex,
            edge_to_exclude,
            expected_sf=sf
        )
    return H

# Function to adjust the hypergraph to match the expected simplicial fraction
def final_edge_adjustment_sf(H, maximal_edge_is_simplex, maximal_edge_not_simplex, edge_to_exclude, expected_sf):
    # Calculate the current simplicial fraction
    curr_sf = simplicial_fraction(H, min_size=2)
    print("!!!curr_sf before adjustment!!!:", curr_sf)
    # Split to cases to add or remove edges respectively
    if curr_sf > expected_sf:
        # Adjustment 1: remove edges that are simplices from the hypergraph
        maximal_simplices = list(maximal_edge_is_simplex)  # Convert to list for indexing
        while (curr_sf > expected_sf) and len(maximal_simplices) > 0:
            # Randomly select a maximal simplex, which edges will be removed one by one
            tmp_idx = random.randint(0, len(maximal_simplices) - 1)
            # Remove the selected maximal simplex from the list
            maximal_selected = maximal_simplices.pop(tmp_idx)
            maximal_edge_is_simplex.discard(maximal_selected)  # Also remove from set
            
            # Find all edges of the selected maximal simplex
            tmp_edges = powerset(maximal_selected, 2, len(maximal_selected) - 1)
            edges = [frozenset(item) for item in list(tmp_edges)]
            
            # Remove edges one by one until the simplicial fraction is equal to the expected value
            edges_to_remove = edges.copy()
            random.shuffle(edges_to_remove)  # Randomize removal order
            
            # Check if there are edges to remove
            if len(edges_to_remove) == 0:
                continue  # Skip this simplex if it has no removable edges
            
            edge_id_map = {}
            for edge_id, edge_members in H.edges.members(dtype=dict).items():
                edge_id_map[frozenset(edge_members)] = edge_id
            # First remove 1 edge to avoid overshooting
            H.remove_edge(edge_id_map[edges_to_remove[0]])
            edge_to_exclude.discard(edges_to_remove[0])
            edges_to_remove.pop(0)
            curr_sf = simplicial_fraction(H, min_size=2)
            
            if curr_sf <= expected_sf:
                for edge_remove in edges_to_remove:
                    # Find and remove the edge from hypergraph
                    H.remove_edge(edge_id_map[edge_remove])
                    edge_to_exclude.discard(edge_remove)
                    curr_sf = simplicial_fraction(H, min_size=2)
                    if curr_sf <= expected_sf:
                        return H

        # Note: we don't do this adjusment in this case because it can easily form new simplex
        ############################################################################        
        # # Adjustment 2: add edges that are not simplices to the hypergraph:
        # while (curr_sf > expected_sf) and len(maximal_edge_is_simplex) > 0:
        #     tmp_idx = random.randint(0, len(final_possible_edge_list) - 1)
        #     tmp_add = final_possible_edge_list.pop(tmp_idx)
        #     # tmp_add is a list of sets representing possible edges
        #     H.add_edge(tmp_add)
        #     curr_sf = simplicial_fraction(H, min_size=2)
        ############################################################################
    print("curr_sf:", curr_sf)
    # Split to cases to add or remove edges respectively
    if curr_sf < expected_sf:
        # Add non-simplicial edges to increase the total number of edges (decreasing simplicial fraction)
        maximal_non_simplices = list(maximal_edge_not_simplex)  # Convert to list for indexing
        while (curr_sf < expected_sf) and len(maximal_non_simplices) > 0:
            if (expected_sf - curr_sf) > 0.3:
                # Randomly select a maximal hyperedge, which edges can be removed to decrease simplicial fraction
                tmp_idx = random.randint(0, len(maximal_non_simplices) - 1)
                # Remove the selected maximal hyperedge from the list
                maximal_selected = maximal_non_simplices.pop(tmp_idx)
                maximal_edge_not_simplex.discard(maximal_selected)  # Also remove from set
                maximal_edge_is_simplex.add(maximal_selected)
                
                # Find all edges in the selected maximal hyperedge
                tmp_edges = powerset(maximal_selected, 2, len(maximal_selected) - 1)
                all_edges = [frozenset(item) for item in list(tmp_edges)]
                
                # Add back all remaining edges to complete the simplex
                current_edges = H.edges.filterby("size", 2, "geq").members()
                current_edge_sets = {frozenset(edge) for edge in current_edges}
                
                edges_to_add_back = []
                for edge_to_add in all_edges:
                    if edge_to_add not in current_edge_sets:
                        edges_to_add_back.append(list(edge_to_add))
                        edge_to_exclude.add(edge_to_add)
                
                if edges_to_add_back:
                    H.add_edges_from(edges_to_add_back)
                    maximal_edge_is_simplex.add(maximal_selected)  # Add back to simplex set
                
                if curr_sf >= expected_sf:
                    return H
            else:
                tmp_idx = random.randint(0, len(maximal_non_simplices) - 1)
                maximal_selected = maximal_non_simplices.pop(tmp_idx)  # Remove from list to avoid reselecting
                maximal_edge_not_simplex.discard(maximal_selected)  # Also remove from set
                
                # Find all edges in the selected maximal hyperedge
                tmp_edges = powerset(maximal_selected, 2, len(maximal_selected) - 1)
                all_edges = [frozenset(item) for item in list(tmp_edges)]
                
                # Find edges that are not part of complete simplices AND actually exist in the hypergraph
                current_edges = H.edges.filterby("size", 2, "geq").members()
                current_edge_sets = {frozenset(edge) for edge in current_edges}
                remove_options = [e for e in all_edges if e not in current_edge_sets]
                
                # If there are non-simplicial edges that can be removed, remove some
                if int(len(remove_options)/5) > 1:
                    edge_id_map = {}
                    for edge_id, edge_members in H.edges.members(dtype=dict).items():
                        edge_id_map[frozenset(edge_members)] = edge_id
                    num_to_remove = int(len(remove_options)/3)  # Remove 1-(len(non_simplicial_edges)/5) edges
                    edges_to_remove = random.sample(remove_options, num_to_remove)
                    
                    for edge_remove in edges_to_remove:
                        if edge_remove in edge_id_map:
                            H.remove_edge(edge_id_map[edge_remove])
                            print("removed edge:", edge_id_map[edge_remove])
                            edge_to_exclude.discard(edge_remove)  # Remove from exclusion set
                            curr_sf = simplicial_fraction(H, min_size=2)
                            if curr_sf >= expected_sf:
                                return H
        return H
    else:
        return H
    
# NOTE THAT THE UPPER BOUND THIS FUNCTION RETURN IS POSSIBLE TO BE A LITTLE SMALLER THAN THE ACTUAL UPPER BOUND (IT DIDN'T CONSIDER OVERLAPPING MAXIMAL HYPEREDGES)
# Function to approximate the upper bound of |C| of a hypergraph with a given edit simpliciality, number of maximal hyperedges, and number of nodes
def approximate_C_upperbound(num_node, min_size, max_size, num_max_hyperedge):
    
    # Case 1: size of maximal hyperedge is not limited by max_size
    if max_size is None or max_size > num_node:
        # Upper bound is the case to form a maximal hyperedge with largest size possible, while other maximal hyperedges are of size min_size
        C_big_edge = possible_combinations((num_node - min_size*(num_max_hyperedge - 1)), min_size)
        C_upperbound = C_big_edge + (num_max_hyperedge - 1)
        return C_upperbound
    else:
        # Case 2: size of maximal hyperedge is limited by max_size
        # Upper bound is the case to form bunch of maximal hyperedges with largest size possible, while other maximal hyperedges are of size min_size
        # except the last one, which is of size num_node % max_size (takes the rest of the possible nodes)
        C_upperbound = 0
        # Calculate the sum of combinations for sizes from min_size to max_size (maximal hyperedges of size max_size)
        while ((num_node - max_size) >= min_size*num_max_hyperedge) and (num_max_hyperedge > 0):
            C_big_edge = possible_combinations(max_size, min_size)
            C_upperbound+= C_big_edge
            num_node -= max_size
            num_max_hyperedge -= 1
        # If there are still maximal hyperedges to form, form them with the rest of the possible nodes
        if num_max_hyperedge > 1:
            # -1 from num_max_hyperedge because we need 1 edge to form a maximal hyperedge with the rest of the possible nodes
            C_upperbound += num_max_hyperedge - 1
            # Considering overlap of maximal hyperedges, 2 edge -> at least 3 nodes, 3 edges -> at least 4 nodes, etc.
            num_node -= (num_max_hyperedge - 1) + 1
        # Add the case to form a maximal hyperedge with the rest of the possible nodes
        C_upperbound += possible_combinations(min(num_node, max_size), min_size)
        return C_upperbound

# Test function to verify no duplicate edges are generated
def test_no_duplicate_edges():
    """Test both model_generation_es and model_generation_sf for duplicate edges"""
    print("Testing model_generation_es...")
    H_es = model_generation_es(es=0, approx_num_C=10, num_max_hyperedge=10, num_node=150, min_size=2, max_size=5, adjust_es=True)
    edges_es = H_es.edges.members()
    
    # Convert edges to frozensets for comparison
    edge_sets_es = [frozenset(edge) for edge in edges_es]
    unique_edges_es = set(edge_sets_es)
    
    print(f"ES - Total edges: {len(edge_sets_es)}")
    print(f"ES - Unique edges: {len(unique_edges_es)}")
    print(f"ES - Duplicates found: {len(edge_sets_es) - len(unique_edges_es)}")
    
    if len(edge_sets_es) == len(unique_edges_es):
        print("✅ No duplicate edges found in model_generation_es!")
        sf_es = simplicial_fraction(H_es, min_size=2)
        es_es = edit_simpliciality(H_es, min_size=2)
        fes_es = face_edit_simpliciality(H_es, min_size=2)
        print(f"ES - Simplicial fraction: {sf_es}")
        print(f"ES - Edit simpliciality: {es_es}")
        print(f"ES - Face edit simpliciality: {fes_es}")
        es_success = True
    else:
        print("❌ Duplicate edges detected in model_generation_es!")
        es_success = False
    
    print("\n" + "="*50)
    print("Testing model_generation_sf...")
    
    # Test model_generation_sf
    H_sf = model_generation_sf(sf=0.7, approx_num_E=200, num_node=1500, min_size=2, max_size=7)
    edges_sf = H_sf.edges.members()
    
    # Convert edges to frozensets for comparison
    edge_sets_sf = [frozenset(edge) for edge in edges_sf]
    unique_edges_sf = set(edge_sets_sf)
    
    print(f"SF - Total edges: {len(edge_sets_sf)}")
    print(f"SF - Unique edges: {len(unique_edges_sf)}")
    print(f"SF - Duplicates found: {len(edge_sets_sf) - len(unique_edges_sf)}")
    
    if len(edge_sets_sf) == len(unique_edges_sf):
        print("✅ No duplicate edges found in model_generation_sf!")
        sf_sf = simplicial_fraction(H_sf, min_size=2)
        es_sf = edit_simpliciality(H_sf, min_size=2)
        fes_sf = face_edit_simpliciality(H_sf, min_size=2)
        print(f"SF - Simplicial fraction: {sf_sf}")
        print(f"SF - Edit simpliciality: {es_sf}")
        print(f"SF - Face edit simpliciality: {fes_sf}")
        sf_success = True
    else:
        print("❌ Duplicate edges detected in model_generation_sf!")
        # Print duplicates for debugging
        edge_counts = {}
        for edge in edge_sets_sf:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
        
        duplicates = {edge: count for edge, count in edge_counts.items() if count > 1}
        print("Duplicate edges:", duplicates)
        sf_success = False
    
    return es_success and sf_success

# Adjust Main function if needed
if __name__ == "__main__":
    # print("Testing edge rewiring model generation...")
    # test_no_duplicate_edges()
    H = model_generation_es(
        es=0.04758,
        approx_num_C=6936,  # Set high to allow target_num_edges to work
        num_max_hyperedge=304,
        num_node=516,
        min_size=2,
        max_size=None,
        adjust_es=False,
    )
    print(xgi.number_connected_components(H))