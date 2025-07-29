import xgi
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
# __file__ is not defined in Jupyter, use current working directory instead
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'edge_rewiring')))
from sod.simpliciality import new_edit_simpliciality, face_edit_simpliciality, simplicial_fraction
from model_generation import *
from datetime import datetime

datasets = [
    "contact-primary-school",
    "contact-high-school",
    "hospital-lyon",
    "email-enron",
    "email-eu",
    "ndc-substances",
    "diseasome",
    "disgenenet",
    "congress-bills",
    "tags-ask-ubuntu",
]

def get_sizes_of_maximal_hyperedges(H):
    maximal_hyperedges = H.edges.maximal().filterby("size", 2, "geq").members()
    sizes = [len(member) for member in maximal_hyperedges]
    max_size = max(sizes)
    count = [0] * (max_size - 2 + 1)
    for size in sizes:
        if size >= 2:
            count[size - 2] += 1
            
    print(f"Count of maximal hyperedges by size: {count}")
    return count


dataset = datasets[int(sys.argv[1])]

print(f"Running max_hyperedge test on dataset: {dataset}")

H_og = (xgi.load_xgi_data(dataset, max_order=11))
es = new_edit_simpliciality(H_og)
num_edges = H_og.num_edges
num_max_hyperedge = len(H_og.edges.maximal().filterby("size", 2, "geq"))
maximal_edge_sizes = [len(e) for e in H_og.edges.maximal().filterby("size", 2, "geq").members()]
#print("maximal_edge_sizes: ", maximal_edge_sizes)
C_distribution = np.array([possible_combinations(i) for i in maximal_edge_sizes])
approx_num_C = (num_edges - num_max_hyperedge + es * num_max_hyperedge) / es
num_node = H_og.num_nodes

H = model_generation_es(
        es=es,
        approx_num_C=approx_num_C,
        num_max_hyperedge=num_max_hyperedge,
        num_node=num_node,
        min_size=2,
        max_size=None,
        adjust_es=True,
        C_distribution=C_distribution
    )
H.cleanup()
es_new = new_edit_simpliciality(H)
error_es = abs(es_new - es)

mean_degree = sum(dict(H.degree()).values()) / H.num_nodes
tau = {k: 0.01/k for k in xgi.unique_edge_sizes(H)}

edges = 4000 + (int(es * 10 - 1) - 1) * 1000
num_max_hyperedge_new = len(H.edges.maximal().filterby("size", 2, "geq"))
approx_num_C_new = (H.num_edges - num_max_hyperedge_new + es_new * num_max_hyperedge)/es_new

print(f"es = {es_new}, node = {H.num_nodes}, edges = {H.num_edges}, num_max_hyperedge = {num_max_hyperedge_new}, approx_c = {approx_num_C_new} \n"
        f"error : es = {es_new - es}, node = {H.num_nodes - num_node}, edges = {H.num_edges - edges}, num_max_hyperedge = {num_max_hyperedge_new - num_max_hyperedge}, approx_c = {approx_num_C_new - approx_num_C} \n")

original_counts = get_sizes_of_maximal_hyperedges(H_og)
generated_counts = get_sizes_of_maximal_hyperedges(H)

print(f"Original Maximal Hyperedge Sizes: {original_counts}")
print(f"Generated Maximal Hyperedge Sizes: {generated_counts}")

# Determine max size
max_size = max(len(original_counts), len(generated_counts))

# Pad shorter list with zeros
original_counts_padded = original_counts + [0] * (max_size - len(original_counts))
generated_counts_padded = generated_counts + [0] * (max_size - len(generated_counts))

sizes = np.arange(2, max_size + 2)

width = 0.4  # width of bars

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(sizes - width/2, original_counts_padded, width=width, label='Original')
ax.bar(sizes + width/2, generated_counts_padded, width=width, label='Generated')

ax.set_xlabel('Maximal Hyperedge Size')
ax.set_ylabel('Counts')
ax.set_title('Counts of Maximal Hyperedges by Size')
ax.legend()

plt.xticks(sizes)
plt.show()