import numpy as np

# Number of simulations
num_simulations = 1000000

# Generate num_simulations samples from standard normal distribution for X and Y
X = np.random.randn(num_simulations)
Y = np.random.randn(num_simulations)
Noise = np.random.randn(num_simulations)

# Compute the product of X and Y
Z1 = X * Y + Noise
Z2 = X * X *  Y


# Compute the standard deviation of the product
std_dev1 = np.std(Z1)
std_dev2 = np.std(Z2)



print("Estimated Standard Deviation of X*Y: ", std_dev1)
print("Estimated Standard Deviation of X*X*Y: ", std_dev2)
