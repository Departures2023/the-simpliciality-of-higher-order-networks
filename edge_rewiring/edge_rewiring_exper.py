import xgi
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'edge_rewiring')))
from edge_rewiring import edge_rewiring_alg
from sod import *
from sod.simpliciality import edit_simpliciality
from colorama import Fore
from edge_rewiring import *
from colorama import init
from termcolor import colored
from multiprocessing import Process, Manager
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
def Construct_New_Graph(index, iter, min_size, max_size, total):
    # Initialize variables to keep track of statistics
    success = 0
    failures = 0
    total_time_taken = 0
    num_missing_subfaces = 0
    
    # Load the graph directly in this process
    try:
        import xgi
        G = xgi.load_xgi_data(datasets[index], max_order=max_size)
        G.cleanup(singletons=True)
    except Exception as e:
        print(f"Error loading dataset {datasets[index]}: {e}")
        return
    
    #For given number of iterations, we do an edge rewiring
    for i in range(iter):
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
        total_time_taken += stats["total_time"]
        #print(colored(datasets[index], 'blue'), stats)

    # Checks min and max failures, updates total values
    if (failures > total["max_failures"]):
        total["max_failures"] = failures
    if (failures < total["min_failures"]):
        total["min_failures"] = failures

    # Updates statistics  
    total["total_cc"] += sum(list(xgi.clustering_coefficient(G).values()))
    es = edit_simpliciality(G, min_size)
    total["total_es"] += es    
    total["centrality"] += sum(list(xgi.clique_eigenvector_centrality(G).values()))
    total["total_success"] += success
    total["total_time"] += total_time_taken
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
def process_dataset (index, trials, rewiring_times, min_size, max_size, latex_list_one, 
                     latex_list_two, og_cc, og_clique_centrality, og_es):   
    
    with Manager() as manager:
        total = manager.dict({
            'total_success': 0,
            'total_time': 0,
            'total_num_missing_subfaces': 0,
            'num_max_hyperedges': 0,
            'min_failures': rewiring_times,
            'max_failures': 0,
            'total_cc': 0,
            'centrality': 0,
            'total_es': 0
            })
        
        # Create processes to run the algorithm in parallel
        processes = []
        # Where trials is the number of processes we want to run
        for i in range(trials):     
            # Runs Construct_New_Graph in its own process      
            p = Process(target=Construct_New_Graph, args=(index, rewiring_times, min_size, 
                                                                    max_size, total))
            processes.append(p)
            p.start()

        # For all processes, joins them to synchronize    
        for p in processes:
            p.join()           
    
        # Updates statistics
        total_success = total["total_success"]
        total_time = total["total_time"]
        max_failures = total["max_failures"]
        min_failures = total["min_failures"]
        num_max_hyperedges = total["num_max_hyperedges"]
        total_cc = total["total_cc"]
        centrality = total["centrality"]
        total_es = total["total_es"]
        # Calculates averages and does necessary rounding
        avg_time = round(total_time / trials, 2)
        total_failures = rewiring_times * trials - total_success
        avg_failures = total_failures / trials
        failure_rate = round((avg_failures / rewiring_times), 5)
        avg_cc = round((total_cc / len(graphs[index].nodes)) / trials, 5)
        delta_cc = round((avg_cc - og_cc), 5)
        centrality = round(((centrality / len(graphs[index].nodes)) / trials), 5)
        delta_clique_centrality = round(og_clique_centrality - centrality, 5)
        es = (total_es / trials)
        delta_es = round((es - og_es), 5)

        # Prints results of each dataset
        print( Fore.LIGHTGREEN_EX + str(datasets[index]) + ": \n" +
            " average time = " + str(avg_time) + "\n" + 
            " average failures = " + str(avg_failures) + "\n" + 
            " failure rate = " + str(failure_rate) + "\n" +
            " min failures = " + str(min_failures) + "\n" +
            " max failures = " + str(max_failures) + "\n" + 
            " average clustering coefficient = " + str(avg_cc) + "\n" + 
            " change in clustering coefficient = " + str(delta_cc) + "\n" +
            " clique eigenvector centrality = " + str(centrality) + "\n" +
            " change in clique eigenvector centrality = " + str(delta_clique_centrality) + "\n" + 
            " edit simpliciality = " + str(es) + "\n" +
            " og edit simpliciality = " + str(og_es) + "\n" +
            " change in edit simpliciality = " + str(delta_es) + "\n")
        
        # Appends results to the latex lists, these produce printed latex that can be copied into a latex document
        latex_list_one.append(
            datasets[index] + " & " +
            str(es) + " & " +
            str(delta_es) + " & " +
            str(avg_time) + " & " + 
            str(avg_failures) + " & " +
            str(failure_rate) + " & " +
            str(min_failures) + " & " +
            str(max_failures) + " & " +
            str(num_max_hyperedges) +
        " \\\\")
        latex_list_one.append("\hline") 

        latex_list_two.append(
            datasets[index] + " & " +
            str(avg_cc) + " & " +
            str(delta_cc) + " & " +
            str(centrality) + " & " +
            str(delta_clique_centrality) +
            " \\\\")
        latex_list_two.append("\hline") 

"""
main
Arguments: 
    1. trials: how many trials do you want
    2. rewiring times: how many rewirings do you want do
    3. start_dataset_index: index of first dataset to process
    4. end_dataset_index: index after last dataset to process

Output:
    Prints Latex Code for the results of the experiments

Processes datasets sequentially, running trials in parallel for each dataset.
""" 
if __name__ == "__main__":
    start = time.time()
    # Checks if arguments are given, if not prints error and exits
    if len(sys.argv) < 5:
        print("Usage: python edge_rewiring_exper.py <trials> <rewiring_times> <start_dataset_index> <end_dataset_index>")
        print("  trials: number of parallel trials per dataset")
        print("  rewiring_times: number of rewirings per trial")
        print("  start_dataset_index: index of first dataset (0-9)")
        print("  end_dataset_index: index after last dataset (1-10)")
        print("\nDatasets (by index):")
        for i, dataset in enumerate(datasets):
            print(f"  {i}: {dataset}")
        sys.exit()
    print("Starting edge rewiring experiments...")
    #Initializes graphs and needed values
    max_size = 11
    min_size = 2
    
    # Parse and validate arguments
    try:
        trials = int(sys.argv[1])
        rewiring_times = int(sys.argv[2])
        begin_dataset = int(sys.argv[3])
        end_dataset = int(sys.argv[4])
    except ValueError:
        print("Error: All arguments must be integers")
        sys.exit(1)
    
    if trials <= 0:
        print("Error: trials must be positive")
        sys.exit(1)
    if rewiring_times <= 0:
        print("Error: rewiring_times must be positive")
        sys.exit(1)
    if begin_dataset < 0 or begin_dataset >= len(datasets):
        print(f"Error: start_dataset_index must be between 0 and {len(datasets)-1}")
        sys.exit(1)
    if end_dataset <= begin_dataset or end_dataset > len(datasets):
        print(f"Error: end_dataset_index must be between {begin_dataset+1} and {len(datasets)}")
        sys.exit(1)

    global graphs
    graphs = []
    latex_list_one = []
    latex_list_two = []
          
    for i in range (10):
        if (i < begin_dataset or i >= end_dataset):
            graphs.append(0)
        else:
            graphs.append(xgi.load_xgi_data(datasets[i], max_order=max_size))              
            graphs[i].cleanup(singletons=True)

    for i in range(begin_dataset, end_dataset):     
        og_cc = sum(list(xgi.clustering_coefficient(graphs[i]).values())) / len(graphs[i].nodes)
        og_clique_centrality = sum(list(xgi.clique_eigenvector_centrality(graphs[i]).values())) / len(graphs[i].nodes)
        og_es = edit_simpliciality(graphs[i], min_size=min_size)

        process_dataset(i, trials, rewiring_times, min_size, max_size, latex_list_one, latex_list_two, og_cc, og_clique_centrality, og_es)

    # Prints the results of the experiments
    end = time.time()
    total_time = end - start
    print(colored("\n Done! - Time:" + str(total_time) + "\n", "red"))
    print(*latex_list_one, sep="\n")
    print("\n\n\n ***** \n\n\n")
    print(*latex_list_two, sep="\n")