import powerlaw
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import Counter
# import numpy as np
# import seaborn as sns
import xgi
# from matplotlib import cm
# from draw import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sod import *
from sod.simpliciality import edit_simpliciality
# from sod.simpliciality import edge_rewiring
from edge_rewiring import edge_rewiring_alg
from edge_rewiring import edge_rewiring_exper

def filter_degrees (H) :
    degree_dict = H.degree()
    degree_list = list(degree_dict.values())
    sorted_degrees = sorted(degree_list)
    filtered_degrees = [d for d in sorted_degrees if d > 0]

    return filtered_degrees

def has_powerlaw(H, p_value = 0.10) :
    filtered_degrees = filter_degrees(H)
    # degree_dict = H.degree()
    # degree_list = list(degree_dict.values())
    # sorted_degrees = sorted(degree_list)
    # filtered_degrees = [d for d in sorted_degrees if d > 0]
    # print(filtered_degrees)

    fit = powerlaw.Fit(filtered_degrees)

    R, p = fit.distribution_compare('truncated_power_law', 'exponential')
    print(f"  R = {R:.4f}, p = {p:.4f}")
    # plt.figure(figsize=(8, 6))
    # powerlaw.plot_pdf(filtered_degrees, color='b', label='Empirical Data')
    # fit.power_law.plot_pdf(color='r', linestyle='--', label='Power-law Fit')
    # plt.xlabel("Degree")
    # plt.ylabel("P(degree)")
    # plt.title("Power-law Fit to Hypergraph Degree Distribution")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    return R > 0 and p > p_value

def degree_distribution(H) :
    # Check if H is loaded correctly
    if H is None:
        print("Failed to load the hypergraph.")
        sys.exit(1)
    # Get degrees
    degree_list = list(H.degree().values())

    # Count how many nodes have each degree
    degree_counts = Counter(degree_list)

    # Sort degrees for plotting
    degrees = sorted(degree_counts.keys())
    counts = [degree_counts[d] for d in degrees]
    # probabilities = [counts / len(H) for count in degree_counts.values()]
    # print(probabilities)

    data = list(H.degree().values()) # Get degree data
    # print(data)
    # print(H.nodes)
    fit = powerlaw.Fit(data, xmin=1)
    # print(H.nodes)
    print(f"Power-law exponent (alpha): {fit.alpha}")
    print(f"Standard error on alpha: {fit.sigma}")
    fig = fit.plot_pdf(color='b', linewidth=2)
    fit.power_law.plot_pdf(color='g', linestyle='--', ax=fig)
    # fig = fit.plot_pdf(color='b', marker='o', linestyle='None', label="Empirical")
    # fit.power_law.plot_pdf(color='g', linestyle='--', ax=fig, label="Power law fit")
    # plt.loglog(degrees, probabilities, marker='o', linestyle='-', color='b')
    plt.legend()
    plt.xlabel("Degree")
    plt.ylabel("Probability")
    plt.title("Power-law degree distribution")
    plt.show()