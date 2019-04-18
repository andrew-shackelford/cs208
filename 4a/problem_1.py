"""
Andrew Shackelford
ashackelford@college.harvard.edu

CS 208 - Spring 2019
Homework 4a, Problem 1
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.stats import norm
import pickle

false_rates_dict = {}

# read in the data from the CSV file
def read_data(file):
    with open(file, 'rb') as f:
        next(f) # skip labels
        X, Y = [], []
        reader = csv.reader(f)
        for row in reader:
            res_x = []
            x = map(int, row[:-2]) # map input to integers
            flip_x = map(lambda n: 1 ^ n, x) # flip bits
            res_x.extend(x)
            res_x.extend(flip_x)

            X.append(res_x)
            Y.append(int(row[-1]))

        return np.array(X), np.array(Y)

# generate true probabilities
def true_data(X, Y):
    p = np.zeros(X.shape)
    for j in range(X.shape[1]):
        X_j = X[:, j]
        for i in range(X.shape[0]):
            p[i][j] = int(X_j[i] == 0 and Y[i] == 1)
    return p

# return mean for true probabilities
def true_mechanism(p):
    p_mean = np.zeros(p.shape[1])
    for j in range(p.shape[1]):
        p_j = p[:, j]
        p_mean[j] = np.mean(p_j)
    return p_mean

# return centralized dp mean
def centralized_mechanism(p, epsilon):
    p_mean = true_mechanism(p)
    p_dp = np.zeros(p_mean.shape)

    sensitivity = 1. / float(p.shape[0])
    scale = float(p.shape[1]) * sensitivity / float(epsilon)
    for j in range(p.shape[1]):
        p_dp[j] = p_mean[j] + np.random.laplace(scale=scale)

    return p_dp

# return localized dp mean using randomized response
# if desired, use scale_result to try to estimate the population mean
# however, this is not needed with our threshold from our normal approx.
def local_mechanism(p, epsilon, scale_result=False):
    # divide epsilon by d
    epsilon = float(epsilon) / float(p.shape[1])

    # calcuate probabilities and scaling factor
    true_prob = np.exp(epsilon) / (np.exp(epsilon) + 1.)
    false_prob = 1. / (np.exp(epsilon) + 1.)
    c = (np.exp(epsilon) + 1.) / (np.exp(epsilon) - 1)
    probs = [true_prob, false_prob]

    # perform randomized response
    choices = np.random.choice([1, -1], size=p.shape[0]*p.shape[1], replace=True, p=probs)
    choices = choices.reshape(p.shape)
    p_dp = np.multiply(p, choices) # perform multiplication instead of if/else for runtime savings
    p_dp_mean = np.mean(p_dp, axis=0)

    # scale the results back to [0, 1] (+ noise), with scaling factor if desired
    p_res = np.zeros(p_dp_mean.shape)
    for j in range(p.shape[1]):
        if scale_result:
            p_res[j] = ((c * p_dp_mean[j]) + 1.) / 2.
        else:
            p_res[j] = (p_dp_mean[j] + 1.) / 2.

    return p_res

# calculate the threshold for centralized dp
def calculate_centralized_threshold(d, n, epsilon):
    d, n, epsilon = float(d), float(n), float(epsilon)

    ln_term = 2. - (2. * np.power(0.9, 1./d))
    coeff = -d / (n * epsilon)
    return coeff * np.log(ln_term)

# calculate the threshold for local dp
def calculate_local_threshold(d, n, epsilon):
    d, n, epsilon = float(d), float(n), float(epsilon) / float(d)

    inverse_cdf_term = np.power(0.9, 1./d)
    mu = 1. / (1. + np.exp(epsilon))
    variance = (np.exp(epsilon)) / (n * np.square(1. + np.exp(epsilon)))
    return norm.ppf(inverse_cdf_term, loc=mu, scale=np.sqrt(variance))

# given a threshold, return the x variables that pass
def threshold(p, t):
    return np.argwhere(p <= t).flatten()

# calculate the false pos and neg rates given the chosen indices
def calculate_false_rate(X, Y, indices):
    # use dictionary to avoid having to re-run expensive calculations
    global false_rates_dict
    if indices.tostring() in false_rates_dict:
        return false_rates_dict[indices.tostring()]

    num_false_pos, num_false_neg = 0, 0
    for i in range(X.shape[0]):
        if indices.shape[0] == 0:
            num_false_neg += 1
        else:
            predicted = np.min(X[i][indices])
            num_false_pos += max(0, predicted - Y[i])
            num_false_neg += max(0, Y[i] - predicted)

    false_rates_dict[indices.tostring()] = (num_false_pos, num_false_neg)
    return num_false_pos, num_false_neg

# print out x variables that pass as well as false pos and neg rates
# for true results, centralized dp, and local dp with given dataset
def test_success():
    # set constants, read in and scale data
    epsilon = 1.
    X, Y = read_data('CaPUMS5full.csv')
    p = true_data(X, Y)
    scaled_p = (p - 0.5) * 2

    # calculate true x variables, print out false positives and negatives for sanity check
    print "No Privacy:"
    result = threshold(true_mechanism(p), 0.00000001)
    print result
    print calculate_false_rate(X, Y, result)

    # perform centralized dp, print out results
    print "Centralized Mechanism:"
    t = calculate_centralized_threshold(p.shape[1], p.shape[0], epsilon)
    result = threshold(centralized_mechanism(p, epsilon), t)
    print result
    print calculate_false_rate(X, Y, result)

    # perform local dp, print out results
    print "Local Mechanism:"
    t = calculate_local_threshold(scaled_p.shape[1], scaled_p.shape[0], epsilon)
    result = threshold(local_mechanism(scaled_p, epsilon), t)
    print result
    print calculate_false_rate(X, Y, result)

# calculate centralized and local dp false rates over various bootstrapped datasets
def false_rates():
    # read in and scale data, calculate true result
    X, Y = read_data('CaPUMS5full.csv')
    p = true_data(X, Y)
    scaled_p = (p - 0.5) * 2
    true_result = threshold(true_mechanism(p), 0.00000001)

    # define constants, bootstrap sizes
    epsilon = 1.
    num_trials = 100
    bootstrap_sizes = np.logspace(-5, 0)
    centralized_false_pos_results, centralized_false_neg_results = [], []
    local_false_pos_results, local_false_neg_results = [], []

    for size in bootstrap_sizes:
        print size

        centralized_num_false_pos, centralized_num_false_neg = 0, 0
        for i in range(num_trials):
            # create bootstrap
            indices = np.random.choice(p.shape[0], size=int(size*p.shape[0]), replace=True)
            p_bootstrap = p[indices]
            scaled_p_bootstrap = scaled_p[indices]

            # perform dp analysis
            centralized_t = calculate_centralized_threshold(p_bootstrap.shape[1], p_bootstrap.shape[0], epsilon)
            centralized_result = threshold(centralized_mechanism(p_bootstrap, epsilon), centralized_t)
            local_t = calculate_local_threshold(p_bootstrap.shape[1], p_bootstrap.shape[0], epsilon)
            local_result = threshold(local_mechanism(p_bootstrap, epsilon), local_t)

            # calculate number of false pos and neg
            centralized_trial_pos, centralized_trial_neg = calculate_false_rate(X, Y, centralized_result)
            centralized_num_false_pos += centralized_trial_pos
            centralized_num_false_neg += centralized_trial_neg
            local_trial_pos, local_trial_neg = calculate_false_rate(X, Y, local_result)
            local_num_false_pos += local_trial_pos
            local_num_false_neg += local_trial_neg

        # calculate rates
        centralized_false_pos_rate = float(centralized_num_false_pos) / float(num_trials * p.shape[0])
        centralized_false_neg_rate = float(centralized_num_false_neg) / float(num_trials * p.shape[0])
        local_false_pos_rate = float(local_num_false_pos) / float(num_trials * scaled_p.shape[0])
        local_false_neg_rate = float(local_num_false_neg) / float(num_trials * scaled_p.shape[0])

        # append to results
        centralized_false_pos_results.append(centralized_false_pos_rate)
        centralized_false_neg_results.append(centralized_false_neg_rate)
        local_false_pos_results.append(local_false_pos_rate)
        local_false_neg_results.append(local_false_neg_rate)

    # write results to file
    with open('centralized_false_pos_results.pkl', 'wb') as f:
        pickle.dump(centralized_false_pos_results, f)
    with open('centralized_false_neg_results.pkl', 'wb') as f:
        pickle.dump(centralized_false_neg_results, f)
    with open('local_false_pos_results.pkl', 'wb') as f:
        pickle.dump(local_false_pos_results, f)
    with open('local_false_neg_results.pkl', 'wb') as f:
        pickle.dump(local_false_neg_results, f)

# graph the results
def graph_results():
    # load results from file
    with open('centralized_false_pos_results.pkl', 'rb') as f:
        centralized_false_pos_results = pickle.load(f)
    with open('centralized_false_neg_results.pkl', 'rb') as f:
        centralized_false_neg_results = pickle.load(f)
    with open('local_false_pos_results.pkl', 'rb') as f:
        local_false_pos_results = pickle.load(f)
    with open('local_false_neg_results.pkl', 'rb') as f:
        local_false_neg_results = pickle.load(f)

    # create x axis
    x = np.logspace(-5, 0)

    # plot false positive results
    plt.xscale('log')
    plt.scatter(x, centralized_false_pos_results, marker='o', edgecolors='blue', facecolors='none', label='centralized')
    plt.scatter(x, local_false_pos_results, marker='x', color='red', label='local')
    plt.xlim(8e-6, 1.2)
    plt.ylim(-0.001, 0.010)
    plt.legend(loc='upper left')
    plt.title('False Positive Rates for Statistical Query DP')
    plt.ylabel('False Positive Rate (# classifications / total)')
    plt.xlabel('Size of dataset (relative to full)')
    plt.savefig('false_positive_rate.png')

    # plot false negative results
    plt.clf()
    plt.xscale('log')
    plt.scatter(x, centralized_false_neg_results, marker='o', edgecolors='blue', facecolors='none', label='centralized')
    plt.scatter(x, local_false_neg_results, marker='x', color='red', label='local')
    plt.xlim(8e-6, 1.2)
    plt.ylim(-0.01, 0.26)
    plt.legend(loc='lower left')
    plt.title('False Negative Rates for Statistical Query DP')
    plt.ylabel('False Negative Rate (# classifications / total)')
    plt.xlabel('Size of dataset (relative to full)')
    plt.savefig('false_negative_rate.png')

def main():
    test_success()
    false_rates()
    graph_results()

if __name__ == "__main__":
    main()