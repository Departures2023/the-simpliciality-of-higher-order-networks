from concurrent.futures import ProcessPoolExecutor
import xgi
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
import random
import time
import pandas as pd
import seaborn as sns
from model_generation import model_generation_es
from edge_rewiring import model_generation, power_law
from edge_rewiring.power_law import has_powerlaw
from statsmodels.stats.proportion import proportion_confint

random.seed(42)
np.random.seed(42)

trials = 400
es_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def run_trial(args):
    es, i = args
    H = model_generation.model_generation_es(
        es=es,
        approx_num_C=700,
        num_max_hyperedge=100,
        num_node=600,
        min_size=2,
        max_size=10
    )
    result = has_powerlaw(H, p_value=0.10)
    return {
        "es": es,
        "trial": i,
        "follows_powerlaw": result[0] if isinstance(result, tuple) else result
    }

if __name__ == '__main__':
    all_stats = []
    summary_data = []

    for es in es_values:
        print(f"\nRunning trials for es = {es}")
        start_time = time.time()

        args = [(es, i) for i in range(trials)] 
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(run_trial, args))

        end_time = time.time()
        success_count = sum(1 for r in results if r["follows_powerlaw"])
        all_stats.extend(results)

        lower, upper = proportion_confint(success_count, trials, method='wilson')
        summary_data.append({
            "ES": es,
            "Success": success_count,
            "Run Time": round(end_time - start_time, 2)
        })

        print(f"es = {es} → {success_count}/{trials} followed power-law")
        print(f"ES={es}: {success_count}/{trials}, CI=({lower:.2f}, {upper:.2f})")

    #Summary table
    summary_df = pd.DataFrame(summary_data)
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))
