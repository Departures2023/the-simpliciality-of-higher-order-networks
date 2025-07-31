import xgi
import hypercontagion as hc
import matplotlib.pyplot as plt
import time
import numpy as np
import random
from model_generation import *
from functools import partial
import multiprocessing as mp
import datetime
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
def jaccard_similarity(set1, set2):
    # intersection of two sets
    intersection = len(set1.intersection(set2))
    # Unions of two sets
    union = len(set1.union(set2))
    
    return intersection / union
dataset = datasets[int(sys.argv[1])]
H_og = (xgi.load_xgi_data(dataset, max_order=11))
H_og.cleanup
es = new_edit_simpliciality(H_og)
num_edges = H_og.num_edges
num_max_hyperedge = len(H_og.edges.maximal().filterby("size", 2, "geq"))
maximal_edge_sizes = [len(e) for e in H_og.edges.maximal().filterby("size", 2, "geq").members()]
#print("maximal_edge_sizes: ", maximal_edge_sizes)
C_distribution = np.array([possible_combinations(i) for i in maximal_edge_sizes])
print("C_distribution dataset: ", sum(C_distribution))
approx_num_C = sum(C_distribution)
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


og_edges = {frozenset(H_og.edges.members(edge_id)) for edge_id in H_og.edges}

cur_edges = {frozenset(H.edges.members(edge_id)) for edge_id in H.edges}

jaccard = jaccard_similarity(cur_edges, og_edges)

cc_H = 0
cc_H += sum(list(xgi.clustering_coefficient(H).values()))

CC_og = 0
CC_og += sum(list(xgi.clustering_coefficient(H_og).values()))

centrality_H = 0
centrality_H += sum(list(xgi.clique_eigenvector_centrality(H).values()))
centrality_og = 0
centrality_og += sum(list(xgi.clique_eigenvector_centrality(H_og).values()))

print(f"centrality_H: {centrality_H}, centrality_og: {centrality_og} clustering_og: {CC_og} clustering_H: {cc_H}")

fig, axes = plt.subplots(1, 2, figsize=(20, 10))  # 1 row, 2 columns
# Draw H on the first subplot
xgi.draw(H, ax=axes[0], node_size=5, edge_size=1, edge_color="black", node_color="lightblue")
axes[0].set_title("H")

# Draw H_og on the second subplot
xgi.draw(H_og, ax=axes[1], node_size=5, edge_size=1, edge_color="red", node_color="lightgreen")
axes[1].set_title("H_og")

plt.show()
print(f"Jaccard similarity between original and current edges: {jaccard:.4f}")