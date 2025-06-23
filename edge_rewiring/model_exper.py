import xgi
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
import random
from edge_rewiring.model_generation import model_generation_es
from edge_rewiring.power_law import has_powerlaw

# Define a sweep of edit simpliciality values
es_values = [0.1 * i for i in range(1, 10)]  # 0.1 to 0.9

# Fixed parameters for the model
num_node = 50
approx_num_C = 100
num_max_hyperedge = 10
min_size = 2
max_size = 5

results = []

for es in es_values:
    try:
        H = model_generation_es(
            es=es,
            approx_num_C=approx_num_C,
            num_max_hyperedge=num_max_hyperedge,
            num_node=num_node,
            min_size=min_size,
            max_size=max_size
        )

        follows_powerlaw, stats = has_powerlaw(H, p_value=0.10)

        result = {
            "es": es,
            "follows_powerlaw": follows_powerlaw,
            "alpha": stats["alpha"] if stats else None,
            "sigma": stats["sigma"] if stats else None,
            "xmin": stats["xmin"] if stats else None
        }
        results.append(result)

        print(f"es={es:.1f}, follows_powerlaw={follows_powerlaw}, alpha={result['alpha']:.2f}" if stats else f"es={es:.1f}, no fit")

    except Exception as e:
        print(f"Error at es={es:.1f}: {e}")

# Optional: plot results
es_vals = [r["es"] for r in results if r["alpha"] is not None]
alphas = [r["alpha"] for r in results if r["alpha"] is not None]

plt.plot(es_vals, alphas, marker='o')
plt.xlabel("Edit Simpliciality (es)")
plt.ylabel("Power-law Exponent (alpha)")
plt.title("Effect of Edit Simpliciality on Power-law Degree Fit")
plt.grid(True)
plt.show()
