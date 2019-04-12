"""
Andrew Shackelford
ashackelford@college.harvard.edu

CS 208 - Spring 2019
Homework 4a, Problem 1
"""

import numpy as np
import csv

def read_test_data(file):
    with open(file, 'rb') as f:
        next(f)
        X, Y = [], []
        reader = csv.reader(f)
        for row in reader:
            X.append(map(int, row[:-1]))
            Y.append(int(row[-1]))
        return np.array(X), np.array(Y)

def centralized_mechanism(X, Y, epsilon):
    p = np.zeros(X.shape[1])

    for j in range(X.shape[1]):
        X_j = X[:, j]
        p_j = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            p_j[i] = int(X_j[i] == 0 and Y[i] == 1)
        p[j] = np.mean(p_j)

    sensitivity = 1. / float(X.shape[0])
    scale = float(X.shape[1]) * sensitivity / float(epsilon) # divide epsilon by d
    for j in range(X.shape[1]):
        p[j] = p[j] + np.random.laplace(scale=scale)

    return p

def local_mechanism(X, Y, epsilon):
    true_prob = np.exp(epsilon) / (np.exp(epsilon) + 1.)
    false_prob = 1. / (np.exp(epsilon) + 1.)
    c = (np.exp(epsilon) + 1.) / (np.exp(epsilon) - 1)
    probs = [true_prob, false_prob]
    unscaled_p = np.zeros(X.shape)

    for j in range(X.shape[1]):
        X_j = X[:, j]
        for i in range(X.shape[0]):
            p_j = 1 if X_j[i] == 0 and Y[i] == 1 else -1 # scale to {-1, 1}
            unscaled_p[i][j] = np.random.choice([p_j, -p_j], p=probs)

    p = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        p_j = np.mean(unscaled_p, axis=0)[j]
        p[j] = ((c * p_j) + 1.) / 2. # scale back to [0, 1] (+ noise)

    return p

def threshold(p, t):
    return np.argwhere(p <= t).flatten()

def main():
    X, Y = read_test_data('hw4testdata.csv')

    print threshold(centralized_mechanism(X, Y, 0.5), 0.01)
    print threshold(local_mechanism(X, Y, 0.5), 0.01)


if __name__ == "__main__":
    main()