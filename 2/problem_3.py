"""
Andrew Shackelford
ashackelford@college.harvard.edu

CS 208 - Spring 2019
Homework 2, Problem 3
"""

import numpy as np
import matplotlib.pyplot as plt

def poisson_dataset(n):
    return np.random.poisson(lam=10, size=n)

def regression_release(x, y, epsilon_partitions, x_a, x_b, y_a, y_b):
    # create and clip variables
    n = x.shape[0]
    x = np.clip(x, x_a, x_b)
    y = np.clip(y, y_a, y_b)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    xx, xy = [], []

    # calculate estimators
    for i in range(n):
        xy.append((x[i] - x_mean) * (y[i] - y_mean))
        xx.append(np.square(x[i] - x_mean))
    xx = np.sum(xx)
    xy = np.sum(xy)

    # calculate scales for xy and xx
    xy_scale = float((x_b-x_a) * (y_b-y_a)) / float(epsilon_partitions[0])
    xx_scale = float(np.square(x_b - x_a)) / float(epsilon_partitions[1])

    # calculate beta and noisy_beta
    beta = float(xy) / float(xx)
    noisy_xy = float(xy) + np.random.laplace(scale=xy_scale)
    noisy_xx = float(xx) + np.random.laplace(scale=xx_scale)
    noisy_beta = noisy_xy / noisy_xx

    # calculate scales for x_mean and y_mean
    x_mean_scale = float(x_b - x_a) / float(n) / float(epsilon_partitions[2])
    y_mean_scale = float(y_b - y_a) / float(n) / float(epsilon_partitions[3])

    # calculate alpha and noisy alpha
    alpha = y_mean - (beta * x_mean)
    noisy_x_mean = float(x_mean) + np.random.laplace(scale=x_mean_scale)
    noisy_y_mean = float(y_mean) + np.random.laplace(scale=y_mean_scale)
    noisy_alpha = noisy_y_mean - (noisy_beta * noisy_x_mean)

    return alpha, beta, noisy_alpha, noisy_beta

# return the value of the linear function
def linear_function(x_i, alpha=1., beta=1., sigma=1.):
    return float(beta) * float(x_i) + float(alpha) + np.random.normal(scale=sigma)

def monte_carlo(epsilon_partitions):
    # create variables
    x_a = 0
    x_b = 15
    y_a = -5
    y_b = 20
    sigma = 1
    n = 1000

    # create datasets and calculate regression
    X = poisson_dataset(n)
    Y = []
    for x_i in X:
        Y.append(linear_function(x_i))
    alpha, beta, noisy_alpha, noisy_beta = regression_release(X, Y, epsilon_partitions, x_a, x_b, y_a, y_b)

    # calculate residuals
    resid_x, resid_y = [], []
    noisy_resid_x, noisy_resid_y = [], []
    for i, x_i in enumerate(X):
        resid_y_i = np.square(Y[i] - beta * x_i - alpha)
        noisy_resid_y_i = np.square(Y[i] - noisy_beta * x_i - noisy_alpha)
        
        resid_x.append(x_i)
        resid_y.append(resid_y_i)
        noisy_resid_x.append(x_i)
        noisy_resid_y.append(noisy_resid_y_i)

    # calculate residual means
    resid = np.mean(resid_y)
    noisy_resid = np.mean(noisy_resid_y)

    return resid_x, resid_y, resid, noisy_resid_x, noisy_resid_y, noisy_resid

def part_b():
    # create variables
    epsilon_partitions = [0.25, 0.25, 0.25, 0.25]
    num_trials = 100

    # get residual plot
    resid_x, resid_y, _, noisy_resid_x, noisy_resid_y, _ = monte_carlo(epsilon_partitions)

    # get averages of mean-squared residuals
    resid_lst = []
    noisy_resid_lst = []
    for _ in range(num_trials):
        _, _, resid, _, _, noisy_resid = monte_carlo(epsilon_partitions)
        resid_lst.append(resid)
        noisy_resid_lst.append(noisy_resid)
    resid = np.mean(resid_lst)
    noisy_resid = np.mean(noisy_resid_lst)

    # plot non-private results
    plt.scatter(resid_x, resid_y)
    plt.title('Residuals for Non-Private Linear Regression\n' +
              'Average mean-squared residuals: ' +
              str(round(resid, 3)))
    plt.xlabel(r'$X_i$')
    plt.ylabel('Squared Residuals')
    plt.savefig('problem_3_non_private.png')
    plt.clf()

    # plot private results
    plt.scatter(noisy_resid_x, noisy_resid_y)
    plt.title('Residuals for Differentially Private Linear Regression\n' +
              'Average mean-squared residuals: ' +
              str(round(noisy_resid, 3)))
    plt.xlabel(r'$X_i$')
    plt.ylabel('Squared Residuals')
    plt.savefig('problem_3_differentially_private.png')
    plt.clf()

# normalize epsilon partitions to sum to 1
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def part_c():
    # create variables and grid
    results = {}
    num_trials = 100
    grids = [
                [1, 1, 1, 1],

                [2, 1, 1, 1],
                [1, 2, 1, 1],
                [1, 1, 2, 1],
                [1, 1, 1, 2],

                [2, 2, 1, 1],
                [3, 3, 2, 1],
                [3, 3, 1, 2],

                [1, 1, 2, 2],
                [1, 1, 2, 3],
                [1, 1, 3, 2],

                [1, 2, 3, 4],
                [1, 2, 4, 3],
                [1, 3, 2, 4],
                [1, 3, 4, 2],
                [1, 4, 2, 3],
                [1, 4, 3, 2],

                [2, 1, 3, 4],
                [2, 1, 4, 3],
                [2, 3, 1, 4],
                [2, 3, 4, 1],
                [2, 4, 1, 3],
                [2, 4, 3, 1],

                [3, 1, 2, 4],
                [3, 1, 4, 2],
                [3, 2, 1, 4],
                [3, 2, 4, 1],
                [3, 4, 1, 2],
                [3, 4, 2, 1],

                [4, 1, 2, 3],
                [4, 1, 3, 2],
                [4, 2, 1, 3],
                [4, 2, 3, 1],
                [4, 3, 1, 2],
                [4, 3, 2, 1],
            ]

    # run grid search
    for epsilon_partitions in grids:
        epsilon_partitions = softmax(epsilon_partitions)
        noisy_resids = []
        for _ in range(num_trials):
            noisy_resids.append(monte_carlo(epsilon_partitions)[-1])
        results[np.mean(noisy_resids)] = epsilon_partitions
        

    # print out latex formatted results table
    print("Results table:")
    for key in sorted(results.keys()):
        ep_1 = round(results[key][0], 2)
        ep_2 = round(results[key][1], 2)
        ep_3 = round(results[key][2], 2)
        ep_4 = round(results[key][3], 2)
        resid = round(key, 3)
        print(str(ep_1) + ' & ' + str(ep_2) + ' & ' + str(ep_3) + ' & ' + str(ep_4) + '&' + str(resid) + '\\\\ \\hline')

    # print out best partition
    print ("best epsilon partition is " +
          str(results[sorted(results.keys())[0]]) +
          " with residuals " +
          str(sorted(results.keys())[0]))

def part_c_optimized():
    # create variables
    epsilon_partitions = [0.40, 0.40, 0.1, 0.1]
    num_trials = 100

    # get averages of mean-squared residuals
    noisy_resid_lst = []
    for _ in range(num_trials):
        _, _, _, _, _, noisy_resid = monte_carlo(epsilon_partitions)
        noisy_resid_lst.append(noisy_resid)
    noisy_resid = np.mean(noisy_resid_lst)

    # print result
    print("mean squared residuals for " +
          str(epsilon_partitions) +
          " is " +
          str(noisy_resid))

def main():
    part_b()
    part_c()
    part_c_optimized()

if __name__ == "__main__":
    main()
