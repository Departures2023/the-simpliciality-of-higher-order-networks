import copy
from itertools import combinations
import math
import sys
import os

from matplotlib import pyplot as plt
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
def generate_C_distribution(min_size, max_size, C_avg, num_max_hyperedge, target_sum):
    """
    Generate a list of n random numbers, each at least min_value, such that their sum is target_sum.

    Args:
        min_size (int): min size of an edge
        max_size (int): max size of an edge
        C_avg (float): average number of combinations (induced edges) within a maximal hyperedge
        num_max_hyperedge (int): number of maximal hyperedges
        target_sum (int): total number of combinations (induced edges) within all maximal hyperedges

    Returns:
        _type_: _description_
    """
    
    # Q1: HOW TO CHOOSE THE VALUE OF STANDARD DEVIATION?
    std = 0.5 * C_avg
    # Q2: CHECK IF I USE THE RIGHT EQUATION FOR UPPER AND LOWER BOUND?
    # Don't override max_size with target_sum - use the actual max_size passed in

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
            lst_choose_from = list(set([x for x in range(0, num_max_hyperedge)]) - set(idx_exclude))
            if (len(lst_choose_from) == 0):
                break
            idx = random.choice(lst_choose_from)
            C_distribution[idx] -= 1
        return C_distribution
    elif excess < 0:
        excess = abs(excess)
        for i in range(int(excess)):
            # exclude indices where the number is already equal to the corresponding C_distribution value
            idx_exclude = [i for i in range(num_max_hyperedge) if max_size == C_distribution[i]]
            lst_choose_from = list(set([x for x in range(0, num_max_hyperedge)]) - set(idx_exclude))
            if (len(lst_choose_from) == 0):
                break
            idx = random.choice(lst_choose_from)
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
    
    # CHOICE1: DIRECTLY RETURN ERROR
    # if target_sum > C_distribution.sum() or target_sum < length * min_edge_num:
    #     raise ValueError("Impossible to generate numbers: Value Error.")
    
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




# Function to generate a hypergraph with a given edit simpliciality, number of maximal hyperedges, and number of nodes
def model_generation_es(es, approx_num_C, num_max_hyperedge, num_node, min_size=2, max_size=None, adjust_es=False):
    # Checking if input parameters are valid
    if max_size is None:
        max_size = num_node
        
    if approx_num_C < num_max_hyperedge:
        print(f"❌ Warning: approx_num_C ({approx_num_C}) is smaller than num_max_hyperedge ({num_max_hyperedge})")
        print(f"Adjusting approx_num_C to {num_max_hyperedge}")
        approx_num_C = num_max_hyperedge
        
    # Calculate the maximum possible combinations for the given parameters
    max_possible_C = approximate_C_upperbound(num_node, min_size, max_size, num_max_hyperedge)
    
    if approx_num_C > max_possible_C:
        print(f"❌ Warning: approx_num_C ({approx_num_C}) is larger than maximum possible combinations ({max_possible_C})")
        print(f"Adjusting approx_num_C to {max_possible_C}")
        approx_num_C = max_possible_C

    # |C| of the graph
    C_total = int(approx_num_C)
    
    # |H| of the graph
    edge_total = int(approx_num_C * es)
    
    # TODO: Check the performance of model generation with and without adjust_es
    # New implementation of edit simpliciality (es = (|E| - num_max_hyperedge)/(|C| - num_max_hyperedge))
    if adjust_es:
        edge_total = int((C_total - num_max_hyperedge) * es) + num_max_hyperedge
        if not (C_total >= edge_total and edge_total >= num_max_hyperedge):
            print(f"❌ Warning: C_total ({C_total}) is smaller than edge_total ({edge_total}) or num_max_hyperedge ({num_max_hyperedge})")
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
    
    # Print statements for debugging
    # print("edge_total:", edge_total)
    # print("C_total:", C_total)
    # print("num_max_hyperedge:", num_max_hyperedge)
    # print("C_avg:", C_avg)
    start_time_1 = time.time()
    
    # Q3: NEED IMPROVEMENT - BETTER DISTRIBUTION METHOD?
    # Generate the distribution of C values (Union of powerset(maximal hyperedges))
    C_distribution = generate_C_distribution(
        min_size=possible_combinations(min_size), 
        max_size=C_total, 
        C_avg=C_avg, 
        num_max_hyperedge=num_max_hyperedge, 
        target_sum=C_total
    )
    # Print statements for debugging
    print("C_distribution:", C_distribution)
    
    # Generate the distribution of numbers of edges actually connected
    edge_distribution  = generate_edge_distribution(
        min_edge_num=min_size, 
        C_distribution=C_distribution, 
        target_sum=edge_total
    )
    
    # Print statements for debugging
    # print("edge_distribution:", edge_distribution)
    
    # Convert the distribution of C values to the number of nodes in maximal hyperedges
    maximal_edge_size_list = [combination_to_size(i) for i in C_distribution]
    # Avoid adding repeating edges - use set for consistent comparison
    edge_to_exclude = set()
    # Print statements for debugging
    # print("maximal_edge_size_list:", maximal_edge_size_list)
    
    maximal_edge_set = set()
    final_possible_edge_list = []
    end_time_1 = time.time()
    time_1 = end_time_1 - start_time_1
    print(f"Time taken for C_distribution, edge_distribution, maximal_edge_size_list: {time_1} seconds")
    start_time_2 = time.time()
    for i in range(num_max_hyperedge):
        # Randomly select nodes for the maximal hyperedge
        selected_nodes = random.sample(nodes, maximal_edge_size_list[i])
        selected_nodes_set = frozenset(selected_nodes)  # Convert to frozenset for consistent comparison
        
        # Avoid adding repeating nodes and make sure the selected nodes are not a subset of any existing maximal hyperedge
        # Fixed logic: use OR instead of AND, and fix subset comparison
        while (selected_nodes_set in edge_to_exclude or any(selected_nodes_set.issubset(existing_edge) for existing_edge in maximal_edge_set)):
            selected_nodes = random.sample(nodes, maximal_edge_size_list[i])
            selected_nodes_set = frozenset(selected_nodes)
            
        # Add the maximal hyperedge to the hypergraph
        H.add_edge(selected_nodes)
        maximal_edge_set.add(selected_nodes_set)
        edge_to_exclude.add(selected_nodes_set)

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
    end_time_2 = time.time()
    time_2 = end_time_2 - start_time_2
    print(f"Time taken for adding edges: {time_2} seconds")
    
    # Final adjustment of the hypergraph
    start_time_3 = time.time()
    edges = H.edges.filterby("size", min_size, "geq").members()
    print("edges:", len(edges))
    print("maximal_edge_set:", len(maximal_edge_set))
    # Exclude the maximal hyperedges (edges constructed from selected_nodes)
    edges = [edge for edge in edges if frozenset(edge) not in maximal_edge_set]
    print("new edges:", len(edges))
    
    H, curr_es = final_edge_adjustment_es(
        H, 
        edges, 
        final_possible_edge_list, 
        edge_to_exclude=edge_to_exclude,
        expected_es=es
    )
    end_time_3 = time.time()
    time_3 = end_time_3 - start_time_3
    print(f"Time taken for final adjustment: {time_3} seconds")
    es_diff = curr_es - es
    return H, es_diff, time_1, time_2, time_3
    
# Function to slightly adjust the hypergraph to match the expected edit simpliciality
def final_edge_adjustment_es(H, edges, final_possible_edge_list, edge_to_exclude, expected_es):
    # Calculate the current edit simpliciality
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
            # Check if the edit simpliciality is close to the expected value only every 2 iterations
            if count == 2:
                count = 0
                curr_es = edit_simpliciality(H, min_size=2)
                # if curr_es >= expected_es:
                if (curr_es >= expected_es) or (abs(curr_es - expected_es) < 0.002):
                    return H, curr_es
    elif curr_es > expected_es:
        # Remove edges from the hypergraph untul the edit simpliciality is equal to the expected value
        # curr_es > expected_es
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
            if count == 2:
                count = 0
                curr_es = edit_simpliciality(H, min_size=2)
        return H, curr_es
    else:
        print(f"❌ Warning: Input parameters are not good, please check the input parameters")
        return H, curr_es
                    
    
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
    
    
    return es_success

def save_general_data(es, es_diff, preparation_time, edge_adding_time, final_adj_time, filename):
    # gets all of the stats 
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    lines = [
        f"expected_es: {es}",
        f"es_diff: {es_diff:.5f}",
        f"preparation_time: {preparation_time:.5f}",
        f"edge_adding_time: {edge_adding_time:.5f}",
        f"final_adj_time: {final_adj_time:.5f}",
        "=" * 50
    ]
    # opens and writes to the file
    with open(filename, 'a', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

# Adjust Main function if needed
if __name__ == "__main__":
    print("Testing edge rewiring model generation...")
    es_lst = np.linspace(0.1, 0.95, num=10)
    es_diff_lst = []
    preparation_time_lst = []
    edge_adding_time_lst = []
    final_adj_time_lst = []
    for es in es_lst:
        tmp_preparation_time_lst = []
        tmp_edge_adding_time_lst = []
        tmp_final_adj_time_lst = []
        for i in range(5):
            _, es_diff, preparation_time, edge_adding_time, final_adj_time = model_generation_es(es, 9000, 300, 1000, 2, 11, False)
            es_diff_lst.append(abs(es_diff))
            tmp_preparation_time_lst.append(preparation_time)
            tmp_edge_adding_time_lst.append(edge_adding_time)
            tmp_final_adj_time_lst.append(final_adj_time)
        avg_preparation_time = np.mean(tmp_preparation_time_lst)
        avg_edge_adding_time = np.mean(tmp_edge_adding_time_lst)
        avg_final_adj_time = np.mean(tmp_final_adj_time_lst)
        avg_es_diff = np.mean(es_diff_lst)
        preparation_time_lst.append(avg_preparation_time)
        edge_adding_time_lst.append(avg_edge_adding_time)
        final_adj_time_lst.append(avg_final_adj_time)
        save_general_data(es, avg_es_diff, avg_preparation_time, avg_edge_adding_time, avg_final_adj_time, "experiment_result/model_generation_es/model_generation_time_exper.txt")
    plt.plot(es_lst, preparation_time_lst, label="Preparation time", color="red", marker="o")
    plt.plot(es_lst, edge_adding_time_lst, label="Edge adding time", color="blue", marker="o")
    plt.plot(es_lst, final_adj_time_lst, label="Final adjustment time", color="green", marker="o")
    plt.legend()
    plt.xlabel("Edit simpliciality")
    plt.ylabel("Time (seconds)")
    plt.title("Time taken for model generation")
    plt.show()
        