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
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import gc
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
    # Makes a second graph
    G = graph[0]
    
    og_edges = {frozenset(graph[0].edges.members(edge_id)) for edge_id in graph[0].edges}
    
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
        time += stats["total_time"]
        cur_edges = {frozenset(G.edges.members(edge_id)) for edge_id in G.edges}
        jaccard = jaccard_similarity(cur_edges, og_edges)
        total["Jaccard"].append((i + 1, jaccard))
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
    total["total_time"] += time
    total["total_num_missing_subfaces"] += num_missing_subfaces 
    total["num_max_hyperedges"] = stats["num_maximal_hyperedge"] #only equal as number doesn't change
    sf = simplicial_fraction(G, min_size)
    total["total_sf"] += sf
    fes = face_edit_simpliciality(G, min_size)
    total["total_fes"] += fes
    total["total_degree_assortativity"] += xgi.degree_assortativity(G)
    deg_list = list(xgi.degree_counts(G))
    degrees = []
    for degree_val, count in enumerate(deg_list):
        degree = count * degree_val
        degrees.append(degree)
    total["total_degree_count"] += (sum(degrees)) / len(degrees)

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
                     latex_list_two, latex_list_three, og_cc, og_clique_centrality, og_es, og_sf, og_fes, og_da, og_dc):   
    
    with Manager() as manager:
        graph = manager.list()
        total = manager.dict({
            'total_success': 0,
            'total_time': 0,
            'total_num_missing_subfaces': 0,
            'num_max_hyperedges': 0,
            'min_failures': rewiring_times,
            'max_failures': 0,
            'total_cc': 0,
            'centrality': 0,
            'total_es': 0,
            'total_sf': 0,
            'total_fes': 0,
            'total_degree_assortativity': 0,
            'total_degree_count': 0
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
        total_cc = total["total_cc"]
        centrality = total["centrality"]
        total_es = total["total_es"]
        total_sf = total["total_sf"]
        total_fes = total["total_fes"]
        total_deg_assort = total["total_degree_assortativity"]
        total_degree_count = total["total_degree_count"]
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
        #avg_Jaccard = round(sum(total["Jaccard"]) / trials, 5)        
        jaccard_index.extend(total["Jaccard"])
        sf = (total_sf / trials)
        delta_sf = round((sf - og_sf), 5)
        fes = (total_fes / trials)
        delta_fes = round((fes - og_fes), 5)
        degree_assortativity = round((total_deg_assort / trials), 5)
        delta_da = round(degree_assortativity - og_da, 5)
        avg_degree = round((total_degree_count / trials), 5)
        delta_avg_degree = round((avg_degree - og_dc), 5)


        # Prints results of each dataset
        print( Fore.LIGHTGREEN_EX + str(datasets[index]) + ": \n" +
            " average time = " + str(avg_time) + "\n" + 
            " failure rate = " + str(failure_rate) + "\n" + 
            " edges fit requirements = " + str(num_max_hyperedges) + "\n" +
            " average clustering coefficient = " + str(avg_cc) + "\n" + 
            " change in clustering coefficient = " + str(delta_cc) + "\n" +
            " clique eigenvector centrality = " + str(centrality) + "\n" +
            " change in clique eigenvector centrality = " + str(delta_clique_centrality) + "\n" + 
            " edit simpliciality = " + str(es) + "\n" +
            " og edit simpliciality = " + str(og_es) + "\n" +
            " change in edit simpliciality = " + str(delta_es) + "\n" + 
            " simplicial fraction = " + str(sf) + "\n" +
            " og simplicial fraction = " + str(og_sf) + "\n" +
            " change in simplicial fraction = " + str(delta_sf) + "\n" +
            " face edit simpliciality = " + str(fes) + "\n" +
            " og face edit simpliciality = " + str(og_fes) + "\n" +
            " change in face edit simpliciality = " + str(delta_fes) + "\n" + 
            " degree assortativity = " + str(degree_assortativity) + "\n" +
            " og degree assortativity = " + str(og_da) + "\n" +
            " change in degree assortativity = " + str(round(degree_assortativity - og_da, 5)) + "\n" + 
            " average degree count = " + str(avg_degree) + "\n" + 
            " og average degree count = " + str(og_dc) + "\n" + 
            " change in average degree count = " + str(delta_avg_degree))
               
        # Appends results to the latex lists, these produce printed latex that can be copied into a latex document
        latex_list_one.append(
            datasets[index] + " & " +
            str(avg_time) + " & " + 
            str(failure_rate) + " & " +
            str(round(es, 5)) + " & " +
            str(delta_es) + " & " +
            str(round(sf, 5)) + " & " +
            str(delta_sf) + " & " +
            str(round(fes, 5)) + " & " +
            str(delta_fes) + 
        " \\\\")
        latex_list_one.append("\hline") 

        latex_list_two.append(
            datasets[index] + " & " +
            str(num_max_hyperedges) + " & " +
            str(avg_cc) + " & " +
            str(delta_cc) + " & " +
            str(centrality) + " & " +
            str(delta_clique_centrality) +
            " \\\\")
        latex_list_two.append("\hline") 

        latex_list_three.append(
            datasets[index] + " & " +
            str(degree_assortativity) + " & " +
            str(delta_da) + " & " +
            str(avg_degree) + " & " + 
            str(delta_avg_degree) +
            " \\\\")
        latex_list_three.append("\hline")
    
   
'''Calculates the Jaccard similarity between two sets.
Args:
    set1: First set.
    set2: Second set.
Returns:
    Jaccard similarity as a float.
'''  
def jaccard_similarity(set1, set2):
    # intersection of two sets
    intersection = len(set1.intersection(set2))
    # Unions of two sets
    union = len(set1.union(set2))
    
    return intersection / union
        
   
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
        print("Usage Error: <processes> <rewiring_times> <the number of first dataset> <the number of the end + 1> \ncontact-primary-school, contact-high-school, hospital-lyon, \nemail-enron, email-eu, ndc-substances, \ndiseasome, disgenenet, congress-bills, \ntags-ask-ubuntu")
        sys.exit()
    print("Starting edge rewiring experiments...")
    #Initializes graphs and needed values
    max_size = 11
    min_size = 2
    begin_dataset = int(sys.argv[3])
    end_dataset = int(sys.argv[4])
    trials = int(sys.argv[1])
    rewiring_times = int(sys.argv[2])

    global graphs, jaccard_index
    jaccard_index = []
    graphs = []
    latex_list_one = []
    latex_list_two = []
    latex_list_three = []
          
    for i in range (10):
        if (i < begin_dataset or i >= end_dataset):
            graphs.append(0)
        else:
            graphs.append(xgi.load_xgi_data(datasets[i], max_order=max_size))              
            graphs[i].cleanup(singletons=True)

    # Start the thread pool executor to run the process_dataset function in parallel  
    
    # Instead of submitting all at once, process one at a time
    for i in range(begin_dataset, end_dataset):
        og_cc = sum(list(xgi.clustering_coefficient(graphs[i]).values())) / len(graphs[i].nodes)
        og_clique_centrality = sum(list(xgi.clique_eigenvector_centrality(graphs[i]).values())) / len(graphs[i].nodes)
        og_es = edit_simpliciality(graphs[i], min_size)
        og_sf = simplicial_fraction(graphs[i], min_size)
        og_fes = face_edit_simpliciality(graphs[i], min_size)
        og_da = xgi.degree_assortativity(graphs[i])
        deg_list = list(xgi.degree_counts(graphs[i]))
        degrees = []
        for degree_val, count in enumerate(deg_list):
            degree = count * degree_val
            degrees.append(degree)
        og_dc = (sum(degrees)) / len(degrees)

        with ThreadPoolExecutor(min(32, os.cpu_count() + 4)) as executor:
            future = executor.submit(
                process_dataset,
                i,
                trials,
                rewiring_times,
                min_size,
                max_size,
                latex_list_one,
                latex_list_two,
                latex_list_three,
                og_cc,
                og_clique_centrality,
                og_es, 
                og_sf,
                og_fes, 
                og_da, 
                og_dc
            )
            future.result()
        # frees memory hopefully
        del graphs[i]
        gc.collect()

    # Prints the results of the experiments
    end = time.time()
    total_time = end - start

    # Prepare data for plotting        
    x_vals = [x for x, _ in jaccard_index]
    y_vals = [y for _, y in jaccard_index]


    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-')
    plt.title(f'Jaccard Index vs Rewiring Iterations for {datasets[6]}')
    plt.xlabel('Rewiring Iteration')
    plt.ylabel('Jaccard Index')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'jaccard_vs_rewiring_{datasets[6]}.png') 
    plt.close()  # Close the plot to free memory        
    # Prints the results of the experiments
    print(colored("\n Done! - Time:" + str(total_time) + "\n", "red"))
    print(*latex_list_one, sep="\n")
    print("\n\n\n ***** \n\n\n")
    print(*latex_list_two, sep="\n")
    print("\n\n\n ***** \n\n\n")
    print(*latex_list_three, sep="\n")
