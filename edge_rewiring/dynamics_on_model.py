import xgi
import hypercontagion as hc
import matplotlib.pyplot as plt
import time
import numpy as np
import random
import os
from model_generation import *
from sod.simpliciality import edit_simpliciality, face_edit_simpliciality, simplicial_fraction

output_dir = r'experiment_result\dynamics_on_model\fig'

def dynamics_on_model(es, approx_num_C, num_max_hyperedge, num_node, min_size=2, max_size=None, adjust_es=False, \
    compare_interval_smaller_case=2, compare_interval_bigger_case=2):
    # Generate the hypergraph with specific target
    # For connectivity=1, we need enough edges to connect everything
    H = model_generation_es(
        es=es, 
        approx_num_C=approx_num_C, 
        num_max_hyperedge=num_max_hyperedge, 
        num_node=num_node, 
        min_size=min_size,
        max_size=max_size, 
        adjust_es=adjust_es, 
        compare_interval_smaller_case=compare_interval_smaller_case, 
        compare_interval_bigger_case=compare_interval_bigger_case, 
    )
    if adjust_es:
        curr_es = new_edit_simpliciality(H, min_size=2)
    else:
        curr_es = edit_simpliciality(H, min_size=2)
    print("curr_es:", curr_es)
    print("num_nodes", H.num_nodes)
    print("num_edges", H.num_edges)
    print("num_maximal_edges", len(H.edges.maximal()))
    print("xgi.number_connected_components(H1)", xgi.number_connected_components(H))
    print("H.edges.members()", H.edges.members())
    # H = xgi.load_xgi_data("contact-primary-school")
    H.cleanup(connected=True)
    if adjust_es:
        curr_es = new_edit_simpliciality(H, min_size=2)
    else:
        curr_es = edit_simpliciality(H, min_size=2)
    print("curr_es:", curr_es)
    print("num_nodes", H.num_nodes)
    print("num_edges", H.num_edges)
    print("num_maximal_edges", len(H.edges.maximal()))
    print("xgi.number_connected_components(H1)", xgi.number_connected_components(H))
    gamma = 0.05
    max_size = 1.2*max(xgi.unique_edge_sizes(H))
    tau = {i: 1 for i in xgi.unique_edge_sizes(H)}
    t1, S1, I1, R1 = hc.discrete_SIR(H, tau, gamma, tmin=0, tmax=100, dt=1, rho=0.1)
    num_node = H.num_nodes
    plt.figure()
    plt.plot(t1, S1 / num_node, "g--", label="S (discrete)")
    plt.plot(t1, I1 / num_node, "r--", label="I (discrete)")
    plt.plot(t1, R1 / num_node, "b--", label="R (discrete)")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Fraction of population")
    plt.savefig(os.path.join(output_dir, f"{es}_SIR.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    # es_list = np.linspace(0.15, 0.95, num=5)
    # for es in es_list:
    #     dynamics_on_model(es, 9000, 300, 1000, 2, 11, True, 2, 2)
    dynamics_on_model(
        es=0.2,
        approx_num_C=300,  # Set high to allow target_num_edges to work
        num_max_hyperedge=25,
        num_node=2000,
        min_size=2,
        max_size=None,
        adjust_es=True,
        compare_interval_smaller_case=1,
        compare_interval_bigger_case=1,
    )