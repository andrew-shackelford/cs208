"""
Andrew Shackelford
ashackelford@college.harvard.edu

CS 208 - Spring 2019
Homework 2, Problem 2
"""

import numpy as np
import matplotlib.pyplot as plt

# return a poisson dataset of size n with labmda = 10
def poisson_dataset(n):
    return np.random.poisson(lam=10, size=n)

# release the mean of x according to mechanism 2
def mech_2_release(x, epsilon, a, b):
    x = np.clip(x, a, b)
    n = x.shape[0]
    gs_q = (float(b) - float(a)) / float(n)
    s = gs_q / float(epsilon)
    z = np.random.laplace(scale=s)

    mean = np.mean(x)
    z_clamped = np.clip(z, -abs(b-a), abs(b-a))

    return mean + z_clamped

def main():
    # create dataset, variables
    data = poisson_dataset(200)
    a = 0
    epsilon = 0.5
    num_trials = 100
    true_result = np.mean(data)

    # test different upper bounds of b
    x, y = [], []
    for b in range(1, 40):
        se = []
        for i in range(num_trials):
            noisy_result = mech_2_release(data, epsilon, a, b)
            se.append(np.square(noisy_result - true_result))
        rmse = np.sqrt(np.mean(se))
        x.append(b)
        y.append(rmse)

    # plot results
    plt.plot(x, y)
    plt.title('RMSE vs. Upper Bound')
    plt.xlabel('Upper Bound')
    plt.ylabel('RMSE')
    plt.savefig('problem_2.png')

if __name__ == "__main__":
    main()
