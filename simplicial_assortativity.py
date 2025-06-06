import json
import os
from sys import platform

import xgi
from joblib import Parallel, delayed

from sod import *


def get_simplicial_assortativity(dataset, max_order, metric):
    H = xgi.load_xgi_data(dataset, max_order=max_order)
    H.cleanup()

    a = simplicial_assortativity(H, metric)
    print(f"{dataset}-{metric} completed!", flush=True)
    return dataset, metric, a


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

max_order = 2
metrics = ["sf", "es", "fes"]

if platform == "linux" or platform == "linux2":
    num_processes = len(os.sched_getaffinity(0))
elif platform == "darwin" or platform == "win32":
    num_processes = os.cpu_count()

print(f"{num_processes} processes", flush=True)

arglist = []
for d in datasets:
    for m in metrics:
        arglist.append((d, max_order, m))

data = Parallel(n_jobs=num_processes)(
    delayed(get_simplicial_assortativity)(*arg) for arg in arglist
)

a_data = defaultdict(dict)
for d, metric, a in data:
    a_data[d][metric] = a

with open("Data/empirical_simplicial_assortativity.json", "w") as file:
    datastring = json.dumps(a_data, indent=2)
    file.write(datastring)
