# import xgi
# import matplotlib.pyplot as plt
# import numpy as np
# import powerlaw
# import random
# from model_generation import model_generation_es
# from edge_rewiring.power_law import has_powerlaw
# import time

# import random
# import numpy as np
# import pandas as pd
# import seaborn as sns

# random.seed(42)
# np.random.seed(42)

# # Out of 600 trials, 282 hypergraphs followed power-law (p ≥ 0.10).
# from edge_rewiring import model_generation, power_law
# from statsmodels.stats.proportion import proportion_confint

# es_values = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
# node_sizes = [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1200]
# max_hyperedge_values = [60, 70, 80, 90, 100, 110, 120, 130, 140]
# approx_C_values = [300,400,500,600,700,800,900,1000,1100]
# max_size_values = [5 , 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
# trials = 400
# powerlaw_stats = []
# all_stats = []
# start_time = 0
# end_time = 0


# for es in es_values: 
#     print(f"\nRunning trials for es = {es}")
#     success_count = 0
#     start_time = time.time()
#     for i in range(trials):
#         H = model_generation.model_generation_es(
#             es=0.3,
#             approx_num_C=700,
#             num_max_hyperedge=100,
#             num_node=600,
#             min_size=2,
#             max_size=10
#         )
#         # H.cleanup(singletons=True)
#         result = power_law.has_powerlaw(H, p_value=0.10)

#         # Perform fit and compare distributions
#         degree_data = power_law.filter_degrees(H)
#         fit = powerlaw.Fit(degree_data, verbose=False)
#         R, p = fit.distribution_compare('truncated_power_law', 'exponential')

#         all_stats.append({
#             "es": es,
#             "trial": i,
#             "follows_powerlaw": result[0] if isinstance(result, tuple) else result,
#             "alpha": fit.alpha,
#             "R": R,
#             "p": p,
#             "run_time": end_time - start_time
#         })

#         # powerlaw_stats.append((i, {"alpha": fit.alpha, "R": R, "p": p}))
#         # follows_powerlaw = result[0] if isinstance(result, tuple) else result

#         # # if isinstance(result, tuple):
#         # #     follows_powerlaw = result
#         # #     # fit_stats = powerlaw.Fit(H)
#         # # else:
#         # #     follows_powerlaw = result  

#         # if follows_powerlaw:
#         #     success_count += 1
#         #     print(f"Trial {i+1}: ✅ Follows power-law")
#         # else:
#         #     print(f"Trial {i+1}: ❌ Does NOT follow power-law")
#         # print(f"  α = {fit.alpha:.4f}, R = {R:.4f}, p = {p:.4f}")
#         if result[0] if isinstance(result, tuple) else result:
#             success_count += 1
#             # print(f"Trial {i+1}: ✅ Follows power-law")
#         # else:
#         #     # xgi.write_json(H, f"failed_graph_{i}.json")
#         #     print(f"Trial {i+1}: ❌ Does NOT follow power-law")
#         # print(f"  α = {fit.alpha:.4f}, R = {R:.4f}, p = {p:.4f}")
#     end_time = time.time()
#     print(f"Evaluation time: {end_time-start_time}")
#     print(f"es = {es} → {success_count}/{trials} followed power-law")
#     lower, upper = proportion_confint(success_count, trials, method='wilson')
#     print(f"ES={es}: {success_count}/{trials}, CI=({lower:.2f}, {upper:.2f})")

#     # print(f"fit_stats: {fit_stats}")
#     # print(f"\nOut of {trials} trials, {success_count} hypergraphs followed power-law (p ≥ 0.10).")

#     # Plot is not working
#     # p_values = [stat["p"] for _, stat in powerlaw_stats]

#     # plt.hist(p_values, bins=20, edgecolor='black')
#     # plt.title("Power-law Fit p-values Across All Trials")
#     # plt.xlabel("p-value")
#     # plt.ylabel("Frequency")
#     # plt.show()

# df = pd.DataFrame(all_stats)
# sns.boxplot(x="es", y="p", data=df)
# plt.axhline(0.10, color="red", linestyle="--", label="p=0.10 threshold")
# plt.title("Power-law Fit p-values by Edit Simpliciality")
# plt.legend()
# plt.show()





import multiprocessing as mp
import pandas as pd
import numpy as np
import time
import powerlaw
from model_generation import model_generation_es
from edge_rewiring.power_law import has_powerlaw
from statsmodels.stats.proportion import proportion_confint
import random

random.seed(42)
np.random.seed(42)

es = 0.3
trials_per_setting = 100

# Parameter grid to test
approx_C_values = [850, 875, 900, 925]
node_sizes = [600, 800, 1000]
hyperedge_counts = [150, 200, 250, 300]
max_sizes = [18, 19, 20, 21, 22]

# Create parameter combinations
param_grid = [
    (es, approx_C, num_node, num_hyperedges, max_size)
    for approx_C in approx_C_values
    for num_node in node_sizes
    for num_hyperedges in hyperedge_counts
    for max_size in max_sizes
]

# Trial function
def run_trials(args):
    es, approx_C, num_node, num_hyperedges, max_size = args
    success = 0

    for _ in range(trials_per_setting):
        try:
            H = model_generation_es(
                es=es,
                approx_num_C=approx_C,
                num_max_hyperedge=num_hyperedges,
                num_node=num_node,
                min_size=2,
                max_size=max_size
            )
            result = has_powerlaw(H, p_value=0.10)
            if result[0] if isinstance(result, tuple) else result:
                success += 1
        except Exception:
            continue

    lower, upper = proportion_confint(success, trials_per_setting, method='wilson')

    return {
        "approx_C": approx_C,
        "num_node": num_node,
        "num_hyperedge": num_hyperedges,
        "max_size": max_size,
        "success": success,
        "success_rate": success / trials_per_setting,
        "CI_lower": lower,
        "CI_upper": upper
    }

# Use multiprocessing Pool
if __name__ == '__main__':
    start_time = time.time()
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        results = pool.map(run_trials, param_grid)

    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="success_rate", ascending=False)

    print("Top 10 configurations by success rate:")
    print(df_sorted.head(10))

    df.to_csv("powerlaw_es03_experiment_results.csv", index=False)
    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")
