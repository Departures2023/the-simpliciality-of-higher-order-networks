# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
import xgi
# from matplotlib import cm
# from draw import *
from sod import *
from sod.simpliciality import edit_simpliciality

import os
from sod.simpliciality import edge_rewiring

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
dir = {
    "contact-primary-school": "experiment_result/contact-primary-school.txt",
    "contact-high-school": "experiment_result/contact-high-school.txt",
    "hospital-lyon": "experiment_result/hospital-lyon.txt",
    "email-enron": "experiment_result/email-enron.txt",
    "email-eu": "experiment_result/email-eu.txt",
    "ndc-substances": "experiment_result/ndc-substances.txt",
    "diseasome": "experiment_result/diseasome.txt",
    "disgenenet": "experiment_result/disgenenet.txt",
    "congress-bills": "experiment_result/congress-bills.txt",
    "tags-ask-ubuntu": "experiment_result/tags-ask-ubuntu.txt",
}
max_order = 11
min_size = 2
for i in range(9):
    print("dataset: ", datasets[i])
    for j in range (3):
        H = xgi.load_xgi_data(datasets[i], max_order=max_order)
        H.cleanup(singletons=True)

        H, stats = edge_rewiring.rewire_Alg1_expr(H, min_size, max_order)
        H.cleanup(singletons=True)
        edge_rewiring.save_expr_data(datasets[i], j, stats, dir[datasets[i]])
        print(stats)