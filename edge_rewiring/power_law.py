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