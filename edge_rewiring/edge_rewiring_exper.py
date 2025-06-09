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
from colorama import Fore
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


def edge_rewiring_exper(int, times):
    total_time = 0
    delta_ES = 0
    for j in range (3):
        H = xgi.load_xgi_data(datasets[int], max_order=max_order)
        H.cleanup(singletons=True)

        H, stats = edge_rewiring_alg.rewire_Alg1_expr(H, min_size, max_order)
        H.cleanup(singletons=True)
        edge_rewiring_alg.save_expr_data(datasets[int], j, stats, dir[datasets[int]])
        total_time += stats["total_time"]
        delta_SF += stats["delta_SF"]
        delta_ES += stats["delta_ES"]
        delta_FES += stats["delta_FES"]
        print(colored(datasets[int], 'blue'), stats)  
        
    print( Fore.LIGHTGREEN_EX + str(datasets[int]) + ": total_time = " + str(total_time) + 
          " delta_SF = " + str(delta_SF) + 
          " delta_ES = " + str(delta_ES) + 
          " delta_FES = " + str(delta_FES))
    
        
        
if __name__ =="__main__": 
    threads = []
    for i in range(10):
        thread = threading.Thread(target=edge_rewiring_exper, args=(i, 3, ))
        threads.append(thread)
        thread.start()
        
    for thread in threads:
        thread.join()

    print("Done!") 
    
    ''' t1 = threading.Thread(target=edge_rewiring_exper, args=(0,))
    t2 = threading.Thread(target=edge_rewiring_exper, args=(1,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()'''
    