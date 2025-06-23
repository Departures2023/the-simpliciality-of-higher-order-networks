import xgi
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
from multiprocessing import Process, Manager, Queue
import time

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
        
"""
Construct_New_Graph
Inputs: 
    index - index of dataset we want to use
    iter - number of rewirings we want to occur
    min size - minimum size used in main 
    max size - maximum size used in main
    total - dictionary of statistics

Output:
    Updates total successes
    Updates total time
    Updates number of missing subfaces 
    Updates number of maximal hyper-edges

Runs one trial of the edge rewiring algorithm on the given dataset, where iter is the number of
 edge rewirings we want.
"""
def Construct_New_Graph(index, iter, min_size, max_size, total, graph):
    # Initialize variables to keep track of statistics
    success = 0
    failures = 0
    time = 0
    num_missing_subfaces = 0
    #For given number of iterations, we do an edge rewiring
    for i in range(iter):
        # Makes a second graph
        G = graph[0].copy()
        # Runs rewiring, saving it as H and the statistics in stats
        H, stats = edge_rewiring_alg.rewire_Alg1_expr(G, min_size, max_size)
        # Removes singletons if there are any
        H.cleanup(singletons=True)
        # Sets new graph to rewired graph
        G = H
        # Save the experiment data
        edge_rewiring_alg.save_expr_data(datasets[index], i, stats, dir[datasets[index]])
        # Updates if there was a sucess or not
        success += stats["success_update"]
        if stats["success_update"] == 0:
            failures += 1
        time += stats["total_time"]
        #print(colored(datasets[index], 'blue'), stats)

    # Checks min and max failures, updates total values
    if (failures > total["max_failures"]):
        total["max_failures"] = failures
    if (failures < total["min_failures"]):
        total["min_failures"] = failures

    # Updates statistics        
    total["total_success"] += success
    total["total_time"] += time
    total["total_num_missing_subfaces"] += num_missing_subfaces
    total["num_max_hyperedges"] = stats["num_maximal_hyperedge"]

"""
Process_Dataset
Inputs: 
    index - dataset index
    trials - number of trials we would like to complete
    rewiring times - number of rewirings, used in construct new graph
    min size - minimum size used in main
    max size - maximum size used in main
    latex list - latex list one for formatting

Output:
    Updates latex list
    Updates total, including total success, total time, number of missing surfaces, number of maximal
     hyper-edges, min failures and max failures Prints the results from each dataset

For a single dataset, runs Construct_New_Graph the given number of trials
"""     
def process_dataset (index, trials, rewiring_times, min_size, max_size, latex_list, graphs):   
    
    with Manager() as manager:
        graph = manager.list()
        total = manager.dict({
            'total_success': 0,
            'total_time': 0,
            'total_num_missing_subfaces': 0,
            'num_max_hyperedges': 0,
            'min_failures': rewiring_times,
            'max_failures': 0
            })
        graph.append(graphs[index])
    # Create threads to run the algorithm in parallel
        processes = []

        # Where trials is the number of processes we want to run
        for i in range(trials):     
            # Runs Construct_New_Graph in its own thread      
            p = Process(target=Construct_New_Graph, args=(index, rewiring_times, min_size, 
                                                                    max_size, total, graph))
            processes.append(p)
            p.start()

        # For all threads, joins them to syncronize    
        for p in processes:
            p.join()           
    
        # Updates statistics
        total_success = total["total_success"]
        total_time = total["total_time"]
        max_failures = total["max_failures"]
        min_failures = total["min_failures"]
        num_max_hyperedges = total["num_max_hyperedges"]

        # Calculates averages and does necessary rounding
        avg_time = round(total_time / trials, 2)
        total_failures = rewiring_times * trials - total_success
        avg_failures = total_failures / trials
        failure_rate = round((avg_failures / rewiring_times), 5)
        
        # Prints results of each dataset
        print( Fore.LIGHTGREEN_EX + str(datasets[index]) + ": \n" +
            " average time = " + str(avg_time) + "\n" + 
            " average failures = " + str(avg_failures) + "\n" + 
            " failure rate = " + str(failure_rate) + "\n" +
            " min failures = " + str(min_failures) + "\n" +
            " max failures = " + str(max_failures) + "\n" )
            #" number of edges that meet requirements = " + str(num_max_hyperedges) + "\n" +
            #" average number missing subfaces = " + str(avg_num_missing_subfaces) + "\n" +
        
        # Appends results to the latex lists, these produce printed latex that can be copied into a latex document
        latex_list.append(
            datasets[index] + " & " +
            str(avg_time) + " & " + 
            str(avg_failures) + " & " +
            str(failure_rate) + " & " +
            str(min_failures) + " & " +
            str(max_failures) + " & " +
            str(num_max_hyperedges) + " & " +
            #str(avg_num_missing_subfaces) + 
            " \\\\")
        latex_list.append("\hline")   

"""
main
Arguments: 
    1. trials: how many trials do you want
    2. rewiring times: how many rewirings do you want do

Output:
    Prints Latex Code for the results of the experiments

Runs Process_Dataset for each dataset in parallel, the given number of times.
""" 
if __name__ == "__main__":
    start = time.time()
    # Checks if arguments are given, if not prints error and exits
    if len(sys.argv) < 3:
        print("Usage Error: <processes> <rewiring_times>")
        sys.exit()
    print("Starting edge rewiring experiments...")
    #Initializes graphs and needed values
    '''global graphs
    graphs = []'''
    max_size = 11
    min_size = 2
    datasets_size = 10
    
    trials = int(sys.argv[1])
    rewiring_times = int(sys.argv[2])

    # For all of the datasets
    '''for i in range (datasets_size):
        # Uploads datasets
        graphs.append(xgi.load_xgi_data(datasets[i], max_order=max_size))
        # Removes any singletons
        graphs[i].cleanup(singletons=True)'''

    with Manager() as manager:
        latex_list = manager.list()
        graphs = manager.list() 
        for i in range (datasets_size):
            graphs.append(xgi.load_xgi_data(datasets[i], max_order=max_size))
            graphs[i].cleanup(singletons=True)
    # Create threads to run the algorithm in parallel
        processes = []

        # For all datasets
        for i in range(datasets_size):       
            # Threads process_dataset so each process runs in parallel
            p = Process(target=process_dataset, args=(i, trials, rewiring_times, min_size, max_size, latex_list, graphs))
            processes.append(p)
            p.start()              

        # For all threads, joins them to syncronize    
        for p in processes:
            p.join()

        # Prints the results of the experiments
        print(colored("All threads finished!", 'red'))
        print(*latex_list, sep="\n")
        print(colored("\n Done!", "red"))

    end = time.time()
    all = end - start
    print("time for all processes is ", all)