import gc
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
from termcolor import colored
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import copy

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
    graph - graph we are using

Output: Returns the following updates statistics after the given number of rewirings
    success
    time
    failures
    total_es
    total_sf
    total_fes
    num_missing_subfaces
    total_cc
    total_centrality
    total_degree_assortativity
    total_degree_count
    num_max_hyperedges

Runs one trial of the edge rewiring algorithm on the given dataset, where iter is the number of
 edge rewirings we want.
"""
def Construct_New_Graph(index, iter, min_size, max_size, graph):
    # Initialize variables to keep track of statistics
    success = 0
    failures = 0
    time = 0
    num_missing_subfaces = 0

    #For given number of iterations, we do an edge rewiring
    for i in range(iter):
        # Runs rewiring, saving it as H and the statistics in stats
        H, stats = edge_rewiring_alg.rewire_Alg1_expr(graph, min_size, max_size)
        # Removes singletons if there are any
        H.cleanup(singletons=True)
        # Sets new graph to rewired graph
        graph = H
        # Save the experiment data
        edge_rewiring_alg.save_expr_data(datasets[index], i, stats, dir[datasets[index]])
        # Updates if there was a sucess or not
        success += stats["success_update"]
        if stats["success_update"] == 0:
            failures += 1
        time += stats["total_time"]

    # Updates statistics  
    es = edit_simpliciality(graph, min_size)
    sf = simplicial_fraction(graph, min_size)
    fes = face_edit_simpliciality(graph, min_size)
    deg_list = list(xgi.degree_counts(graph))
    degrees = []
    for degree_val, count in enumerate(deg_list):
        degree = count * degree_val
        degrees.append(degree)

    #returns statistics so we can use them later
    return {
        'success' : success,
        'time': time,
        'failures': failures,
        'total_es': es,
        'total_sf': sf,
        'total_fes': fes,
        'num_missing_subfaces': num_missing_subfaces,
        'total_cc': sum(list(xgi.clustering_coefficient(graph).values())),
        'total_centrality': sum(list(xgi.clique_eigenvector_centrality(graph).values())),
        'total_degree_assortativity': xgi.degree_assortativity(graph),
        'total_degree_count': (sum(degrees)) / len(degrees),
        'num_max_hyperedges': stats["num_maximal_hyperedge"]
    }

"""
run_process
Inputs: 
    index - index of dataset we want to use
    iter - number of rewirings we want to occur
    min size - minimum size used in main 
    max size - maximum size used in main
    graph - graph we are using

Output: 
    Runs Construct_New_Graph on a deepcopy of the original graph

Helper function for process_dataset
"""
def run_process (hypergraph, index, rewiring_times, min_size, max_size):
        G = copy.deepcopy(hypergraph)
        #TODO: do we need {} for total anymore, I don't think so
        return Construct_New_Graph(index, rewiring_times, min_size, max_size, G)
    

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
def process_dataset (hypergraph, index, trials, rewiring_times, min_size, max_size, latex_list_one, 
                     latex_list_two, latex_list_three, og_cc, og_clique_centrality, og_es, og_sf, og_fes, og_da, og_dc):   
    
    #Saves results returned
    results = []
    #TODO: 8, idk?
    #Uses Process Pool Executor to run trials in parallel
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
        futures = [executor.submit(run_process, hypergraph, index, rewiring_times, min_size, max_size) for t in range(trials)]
        for future in as_completed(futures):
            results.append(future.result())

    #Saves statistics returned from each trial
    total_time = sum(r['time'] for r in results)
    total_failures = sum(r['failures'] for r in results)
    total_cc = sum(r['total_cc'] for r in results)
    total_centrality = sum(r['total_centrality'] for r in results)
    total_es = sum(r['total_es'] for r in results)
    total_sf = sum(r['total_sf'] for r in results)
    total_fes = sum(r['total_fes'] for r in results)
    total_deg_assort = sum(r['total_degree_assortativity'] for r in results)
    total_degree_count = sum(r['total_degree_count'] for r in results)
    num_max_hyperedges = sum(r['num_max_hyperedges'] for r in results)

    # Calculates averages and does necessary rounding
    avg_time = round(total_time / trials, 2)
    avg_cc = round((total_cc / trials) / len(hypergraph.nodes), 5)
    avg_failures = total_failures / trials
    failure_rate = round((avg_failures / rewiring_times), 5)  
    delta_cc = round((avg_cc - og_cc), 5)
    centrality = round((total_centrality / trials) / len(hypergraph.nodes), 5)
    delta_clique_centrality = round(og_clique_centrality - centrality, 5)
    es = round((total_es / trials), 5)
    delta_es = round((es - og_es), 5)
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
        " edit simpliciality = " + str(es) + "\n" +
        " og edit simpliciality = " + str(og_es) + "\n" +
        " change in edit simpliciality = " + str(delta_es) + "\n" + 
        " simplicial fraction = " + str(sf) + "\n" +
        " og simplicial fraction = " + str(og_sf) + "\n" +
        " change in simplicial fraction = " + str(delta_sf) + "\n" +
        " face edit simpliciality = " + str(fes) + "\n" +
        " og face edit simpliciality = " + str(og_fes) + "\n" +
        " change in face edit simpliciality = " + str(delta_fes) + "\n" + 
        " edges fit requirements = " + str(num_max_hyperedges) + "\n" +
        " average clustering coefficient = " + str(avg_cc) + "\n" + 
        " change in clustering coefficient = " + str(delta_cc) + "\n" +
        " clique eigenvector centrality = " + str(centrality) + "\n" +
        " change in clique eigenvector centrality = " + str(delta_clique_centrality) + "\n" + 
        " degree assortativity = " + str(degree_assortativity) + "\n" +
        " og degree assortativity = " + str(og_da) + "\n" +
        " change in degree assortativity = " + str(round(degree_assortativity - og_da, 5)) + "\n" + 
        " average degree count = " + str(avg_degree) + "\n" + 
        " og average degree count = " + str(og_dc) + "\n" + 
        " change in average degree count = " + str(delta_avg_degree) + "\n")
               
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
    trials = int(sys.argv[1])
    rewiring_times = int(sys.argv[2])
    begin_dataset = int(sys.argv[3])
    end_dataset = int(sys.argv[4])

    graphs = []
    latex_list_one = []
    latex_list_two = []
    latex_list_three = []
          
    for i in range(begin_dataset, end_dataset):
        #appends only the graphs we need
        graphs.append(xgi.load_xgi_data(datasets[i], max_order=max_size))              

    for i in range(len(graphs)):
        #Calculates all of the statistics for the original graph
        og_cc = sum(list(xgi.clustering_coefficient(graphs[i]).values())) / len(graphs[i].nodes)
        og_clique_centrality = sum(list(xgi.clique_eigenvector_centrality(graphs[i]).values())) / len(graphs[i].nodes)
        og_es = round((edit_simpliciality(graphs[i], min_size)), 5)
        og_sf = simplicial_fraction(graphs[i], min_size)
        og_fes = face_edit_simpliciality(graphs[i], min_size)
        og_da = xgi.degree_assortativity(graphs[i])
        deg_list = list(xgi.degree_counts(graphs[i]))
        degrees = []
        for degree_val, count in enumerate(deg_list):
            degree = count * degree_val
            degrees.append(degree)
        og_dc = (sum(degrees)) / len(degrees)

        #Uses thread Pool Executor to run process_dataset on all of the datasets
        with ThreadPoolExecutor(min(32, os.cpu_count() + 4)) as executor:
            future = executor.submit(
                process_dataset,
                graphs[i],
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
    print(colored("\n Done! - Time:" + str(total_time) + "\n", "red"))
    print(*latex_list_one, sep="\n")
    print("\n\n\n ***** \n\n\n")
    print(*latex_list_two, sep="\n")
    print("\n\n\n ***** \n\n\n")
    print(*latex_list_three, sep="\n")
