import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Define parameters
mu = 0      # Mean of log(X)
sigma = 0.5   # Standard deviation of log(X)

# Generate log-normal data
data = np.random.lognormal(mean=mu, sigma=sigma, size=1000)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(data, bins=50, density=True, alpha=0.6, color='b', edgecolor='black')

# Overlay log-normal PDF
x = np.linspace(min(data), max(data), 1000)
pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
plt.plot(x, pdf, 'r', linewidth=2, label="Log-Normal PDF")

plt.title("Log-Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()