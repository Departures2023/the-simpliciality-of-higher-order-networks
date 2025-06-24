import xgi
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sod.simpliciality import edit_simpliciality, face_edit_simpliciality, simplicial_fraction
import matplotlib.pyplot as plt
from edge_rewiring import model_generation
from multiprocessing import Process, Manager
def es_new (es, approx_num_C, num_max_hyperedge, num_node, min_size, max_size, lst):
    H_es = model_generation.model_generation_es(es, approx_num_C, num_max_hyperedge, num_node, min_size, max_size)
    es = edit_simpliciality(H_es, min_size=2)
    lst.append(es)
    

if __name__ == "__main__":
    with Manager() as manager:
        lst = manager.list()   
        processes = []
        for i in range(1, 200, 10): 
            p = Process(target=es_new, args=(0.4, 30, i, 200, 2, 10, lst))
            processes.append(p)
            p.start() 
            
        for p in processes:
            p.join()  
            
        print(lst)
        
        plt.plot(range(1, 200, 10), lst, marker='o')
        plt.show()