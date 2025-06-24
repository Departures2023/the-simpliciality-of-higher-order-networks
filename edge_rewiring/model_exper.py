import xgi
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
import random
from model_generation import model_generation_es
from edge_rewiring.power_law import has_powerlaw

import random
import numpy as np
import pandas as pd
import seaborn as sns
from multiprocessing import Pool, cpu_count

random.seed(42)
np.random.seed(42)

# Out of 600 trials, 282 hypergraphs followed power-law (p ≥ 0.10).
from edge_rewiring import model_generation, power_law

es_values = [0.1 * i for i in range(1, 10)]
trials = 10
powerlaw_stats = []
all_stats = []

def run_trial(args):
    es, trial_idx = args
    for es in es_values: 
        print(f"\nRunning trials for es = {es}")
        success_count = 0
        for i in range(trials):
            H = model_generation.model_generation_es(
                es=es,
                approx_num_C=700,
                num_max_hyperedge=100,
                num_node=600,
                min_size=2,
                max_size=10
            )
        # H.cleanup(singletons=True)

        result = power_law.has_powerlaw(H, p_value=0.10)

        # Perform fit and compare distributions
        degree_data = power_law.filter_degrees(H)
        fit = powerlaw.Fit(degree_data, verbose=False)
        R, p = fit.distribution_compare('truncated_power_law', 'exponential')

        all_stats.append({
            "es": es,
            "trial": i,
            "follows_powerlaw": result[0] if isinstance(result, tuple) else result,
            "alpha": fit.alpha,
            "R": R,
            "p": p
        })

        # powerlaw_stats.append((i, {"alpha": fit.alpha, "R": R, "p": p}))
        # follows_powerlaw = result[0] if isinstance(result, tuple) else result

        # # if isinstance(result, tuple):
        # #     follows_powerlaw = result
        # #     # fit_stats = powerlaw.Fit(H)
        # # else:
        # #     follows_powerlaw = result  

        # if follows_powerlaw:
        #     success_count += 1
        #     print(f"Trial {i+1}: ✅ Follows power-law")
        # else:
        #     print(f"Trial {i+1}: ❌ Does NOT follow power-law")
        # print(f"  α = {fit.alpha:.4f}, R = {R:.4f}, p = {p:.4f}")
        if result[0] if isinstance(result, tuple) else result:
            success_count += 1
            print(f"Trial {i+1}: ✅ Follows power-law")
        else:
            # xgi.write_json(H, f"failed_graph_{i}.json")
            print(f"Trial {i+1}: ❌ Does NOT follow power-law")
        print(f"  α = {fit.alpha:.4f}, R = {R:.4f}, p = {p:.4f}")
    print(f"es = {es} → {success_count}/{trials} followed power-law")

    # print(f"fit_stats: {fit_stats}")
    # print(f"\nOut of {trials} trials, {success_count} hypergraphs followed power-law (p ≥ 0.10).")

    # Plot is not working
    # p_values = [stat["p"] for _, stat in powerlaw_stats]

    # plt.hist(p_values, bins=20, edgecolor='black')
    # plt.title("Power-law Fit p-values Across All Trials")
    # plt.xlabel("p-value")
    # plt.ylabel("Frequency")
    # plt.show()
if __name__ == "__main__":
    arg_list = [(es, i) for es in es_values for i in range(trials)]

    with Pool(processes=4) as pool:
        results = pool.map(run_trial, arg_list)

    df = pd.DataFrame(results)

    sns.boxplot(x="es", y="p", data=df)
    plt.axhline(0.10, color="red", linestyle="--", label="p=0.10 threshold")
    plt.title("Power-law Fit p-values by Edit Simpliciality")
    plt.legend()
    plt.xlabel("Edit Simpliciality (es)")
    plt.ylabel("p-value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# df = pd.DataFrame(all_stats)
# sns.boxplot(x="es", y="p", data=df)
# plt.axhline(0.10, color="red", linestyle="--", label="p=0.10 threshold")
# plt.title("Power-law Fit p-values by Edit Simpliciality")
# plt.legend()
# plt.show()




# STUFF FROM MONDAY
# # Define a sweep of edit simpliciality values
# es_values = [0.1 * i for i in range(1, 10)]  # 0.1 to 0.9

# # Fixed parameters for the model
# num_node = 50
# approx_num_C = 100
# num_max_hyperedge = 10
# min_size = 2
# max_size = 5

# results = []

# for es in es_values:
#     try:
#         H = model_generation_es(
#             es=es,
#             approx_num_C=approx_num_C,
#             num_max_hyperedge=num_max_hyperedge,
#             num_node=num_node,
#             min_size=min_size,
#             max_size=max_size
#         )

#         follows_powerlaw, stats = has_powerlaw(H, p_value=0.10)

#         result = {
#             "es": es,
#             "follows_powerlaw": follows_powerlaw,
#             "alpha": stats["alpha"] if stats else None,
#             "sigma": stats["sigma"] if stats else None,
#             "xmin": stats["xmin"] if stats else None
#         }
#         results.append(result)

#         print(f"es={es:.1f}, follows_powerlaw={follows_powerlaw}, alpha={result['alpha']:.2f}" if stats else f"es={es:.1f}, no fit")

#     except Exception as e:
#         print(f"Error at es={es:.1f}: {e}")

# # Optional: plot results
# es_vals = [r["es"] for r in results if r["alpha"] is not None]
# alphas = [r["alpha"] for r in results if r["alpha"] is not None]

# plt.plot(es_vals, alphas, marker='o')
# plt.xlabel("Edit Simpliciality (es)")
# plt.ylabel("Power-law Exponent (alpha)")
# plt.title("Effect of Edit Simpliciality on Power-law Degree Fit")
# plt.grid(True)
# plt.show()
