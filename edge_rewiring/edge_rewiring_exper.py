# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
import xgi
# from matplotlib import cm
# from draw import *
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'edge_rewiring')))
from edge_rewiring import edge_rewiring_alg
from sod import *
from sod.simpliciality import edit_simpliciality
import threading
from colorama import Fore
from edge_rewiring import *
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
    
        
        
# Use a single dataset to run the algorithm multiple times
def loop_Alg1_expr(index, iter, min_size, max_size):
    total_time = 0
    delta_ES = 0
    delta_FES = 0
    delta_SF = 0
    for i in range(iter):
        H, stats = edge_rewiring_alg.rewire_Alg1_expr(graphs[index], min_size, max_size)
        H.cleanup(singletons=True)
        graphs[index] = H
        # Save the experiment data
        edge_rewiring_alg.save_expr_data(datasets[index], i, stats, dir[datasets[index]])
        total_time += stats["total_time"]
        delta_SF += stats["delta_SF"]
        delta_ES += stats["delta_ES"]
        delta_FES += stats["delta_FES"]
        print(colored(datasets[index], 'blue'), stats)  
        
    print( Fore.LIGHTGREEN_EX + str(datasets[index]) + ": total_time = " + str(total_time) + 
          " delta_SF = " + str(delta_SF) + 
          " delta_ES = " + str(delta_ES) + 
          " delta_FES = " + str(delta_FES))
        
        
if __name__ == "__main__":
    print("Starting edge rewiring experiments...")
    global graphs
    graphs = []
    max_size = 11
    min_size = 2
    
    # Load the datasets and clean them
    for i in range (10):
        graphs.append(xgi.load_xgi_data(datasets[i], max_order=max_size))
        graphs[i].cleanup(singletons=True)
    
    # Create threads to run the algorithm in parallel
    threads = []
    for i in range(10):
        thread = threading.Thread(target=loop_Alg1_expr, args=(i, 3, min_size, max_size,))
        threads.append(thread)
        thread.start()
        
    for thread in threads:
        thread.join()

    print("All threads finished")