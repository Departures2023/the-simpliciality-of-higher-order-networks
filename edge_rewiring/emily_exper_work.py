# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
import xgi
# from matplotlib import cm
# from draw import *
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sod import *
from sod.simpliciality import edit_simpliciality
import threading
from edge_rewiring import edge_rewiring_alg
from colorama import init
from termcolor import colored

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


def edge_rewiring_exper(int):
    for j in range (3):
        H = xgi.load_xgi_data(datasets[int], max_order=max_order)
        H.cleanup(singletons=True)

        H, stats, list_all = edge_rewiring_alg.rewire_Alg1_expr(H, min_size, max_order)
        H.cleanup(singletons=True)
        edge_rewiring_alg.save_expr_data(datasets[int], j, stats, dir[datasets[int]])
        print(colored(datasets[int], 'blue'), stats)  

def edge_rewiring_exper_avg(dataset, num):
    name = dataset
    avg_num_max_hyperedges = 0
    num_failures = 0
    avg_time = 0
    avg_num_missing_subfaces = 0
    avg_delta_sf = 0
    avg_delta_fes = 0
    print(colored(name, 'blue'))
    for i in range(num):
        result_tuple = (edge_rewiring_exper(dataset))
        print(type(result_tuple))
        stats_list = result_tuple[2]
        print(type(stats_list))


        
if __name__ =="__main__":
    t1 = threading.Thread(target=edge_rewiring_exper_avg, args=(0, 1))
    t2 = threading.Thread(target=edge_rewiring_exper_avg, args=(1, 1))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print(colored("Done!", 'red')) 
    