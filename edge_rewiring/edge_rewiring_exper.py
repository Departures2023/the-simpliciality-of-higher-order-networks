#im adding this comment to test git
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
        
# Use a single dataset to run the algorithm multiple times, and that process multiple times
def loop_Alg1_expr(index, iter, times, min_size, max_size, latex_list_one, latex_list_two):
    total_success = 0
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
        #print(colored(datasets[index], 'blue'), stats)  
        
    print( Fore.LIGHTGREEN_EX + str(datasets[index]) + ": total_time = " + str(total_time) + 
          " delta_SF = " + str(delta_SF) + 
          " delta_ES = " + str(delta_ES) + 
          " delta_FES = " + str(delta_FES))
        
        

def loop_Alg1_expr_2(index, iter, times, min_size, max_size, latex_list_one, latex_list_two):
    total_success = 0
    total_time = 0
    total_num_missing_subfaces = 0
    total_delta_ES = 0
    total_delta_FES = 0
    total_delta_SF = 0

    # Where times is the number of times we want the process to run 
    for j in range(times):
        num_max_hyperedges = 0
        success = 0
        failures = 0
        time = 0
        num_missing_subfaces = 0
        delta_ES = 0
        delta_FES = 0
        delta_SF = 0
        max_failures = 0
        min_failures = iter
        max_delta_FES = 0
        min_delta_FES = iter
        max_delta_SF = 0
        min_delta_SF = iter

        # Where iter is the number of iterations we want to run the algorithm
        for i in range(iter):
            H, stats = edge_rewiring_alg.rewire_Alg1_expr(graphs[index], min_size, max_size)
            H.cleanup(singletons=True)
            graphs[index] = H
            # Save the experiment data
            edge_rewiring_alg.save_expr_data(datasets[index], i, stats, dir[datasets[index]])
            # Updates all values each time, checking the if statements
            num_max_hyperedges += stats["num_maximal_hyperedge"]
            num_missing_subfaces += stats["num_missing_subface"]
            success += stats["success_update"]
            if stats["success_update"] == 0:
                failures += 1
            time += stats["total_time"]
            delta_ES += stats["delta_ES"]
            delta_SF += stats["delta_SF"]
            if (stats["delta_SF"] > max_delta_SF):
                max_delta_SF = stats["delta_SF"]
            if (stats["delta_SF"]  < min_delta_SF):
                min_delta_SF = stats["delta_SF"]
            delta_FES += stats["delta_FES"]
            if (stats["delta_FES"] > max_delta_FES):
                max_delta_FES = stats["delta_FES"]
            if (stats["delta_FES"] < min_delta_FES):
                min_delta_FES = stats["delta_FES"]
            print(colored(datasets[index], 'blue'), stats)

        # Checks min and max failures, updates total values
        if (failures > max_failures):
            max_failures = failures
        if (failures < min_failures):
            min_failures = failures
        total_success += success
        total_time += time
        total_num_missing_subfaces += num_missing_subfaces
        total_delta_ES += delta_ES
        total_delta_FES += delta_FES
        total_delta_SF += delta_SF

    # Calculates averages and does necessary rounding
    avg_time = round(total_time / times, 2)
    total_failures = iter * times - total_success
    avg_failures = total_failures / times
    failure_rate = avg_failures / iter
    avg_num_missing_subfaces = total_num_missing_subfaces / times
    avg_delta_SF = round((total_delta_SF / times), 5)
    avg_delta_ES = round((total_delta_ES / times), 5)
    avg_delta_FES = round((total_delta_FES / times), 5)

    # Prints results of each dataset
    print( Fore.LIGHTGREEN_EX + str(datasets[index]) + ": \n" + 
          " average time = " + str(avg_time) + "\n" + 
          " average failures = " + str(avg_failures) + "\n" + 
          " failure rate = " + str(failure_rate) + "\n" +
          " min failures = " + str(min_failures) + "\n" +
          " max failures = " + str(max_failures) + "\n" +
          " number of edges that meet requirements = " + str(num_max_hyperedges) + "\n" +
          " average number missing subfaces = " + str(avg_num_missing_subfaces) + "\n" + 
          " average delta_ES = " + str(avg_delta_ES)  + "\n" + 
          " average delta_SF = " + str(avg_delta_SF) + "\n" + 
          " min delta_SF = " + str(round((min_delta_SF), 5)) + "\n" +
          " max delta_SF = " + str(round((max_delta_SF), 5)) + "\n" +
          " average delta_FES = " + str(avg_delta_FES) + "\n" +
          " min delta_FES = " + str(round((min_delta_FES), 5)) + "\n" +     
          " max delta_FES = " + str(round((max_delta_FES), 5)) + "\n")
    
    # Appends results to the latex lists, these produce printed latex that can be copied into a latex document
    latex_list_one.append(
        datasets[index] + " & " +
        str(avg_time) + " & " + 
        str(avg_failures) + " & " +
        str(failure_rate) + " & " +
        str(min_failures) + " & " +
        str(max_failures) + " & " +
        str(num_max_hyperedges) + " & " +
        str(avg_num_missing_subfaces) + " \\\\")
    latex_list_one.append("\hline")
    
    latex_list_two.append(
        datasets[index] + " & " +
        str(avg_delta_ES) + " & " +
        str(avg_delta_SF) + " & " +
        str(round((min_delta_SF), 5)) + " & " +
        str(round((max_delta_SF), 5)) + " & " +
        str(avg_delta_FES) + " & " +
        str(round((min_delta_FES), 5)) + " & " +
        str(round((max_delta_FES), 5)) + 
        " \\\\")
    latex_list_two.append("\hline")

if __name__ == "__main__":
    print("Starting edge rewiring experiments...")
    #global graphs
    #graphs = []
    max_size = 11
    min_size = 2
    latex_list_one = []
    latex_list_two = []

    # Load the datasets and clean them
    ''' for i in range (10):
        graphs.append(xgi.load_xgi_data(datasets[i], max_order=max_size))
        graphs[i].cleanup(singletons=True)
    '''
    # Create threads to run the algorithm in parallel
    threads = []
    for i in range(10):
        #thread = threading.Thread(target=loop_Alg1_expr, args=(i, 3, min_size, max_size,))
        thread = threading.Thread(target=loop_Alg1_expr_2, args=(i, 1, 3, min_size, max_size, latex_list_one, latex_list_two))
        threads.append(thread)
        thread.start()
        
    for thread in threads:
        thread.join()

    '''print(colored("All threads finished!", 'red'))
    print(*latex_list_one, sep="\n")
    print("\n \n \n ***** \n \n \n")
    print(*latex_list_two, sep="\n")
    print(colored("\n Done!", "red"))'''