"""
Andrew Shackelford
ashackelford@college.harvard.edu

CS 208 - Spring 2019
Homework 1, Problem 3
"""

import numpy as np
from scipy import stats
import math
import csv
import progressbar

PUB = ['sex',
       'age',
       'educ',
       'married',
       'divorced',
       'latino',
       'black',
       'asian',
       'children',
       'employed',
       'militaryservice',
       'disability',
       'englishability']

P = 773

def rescale(x):
    if float(x) == 1.:
        return 1.
    else:
        return -1.

def read_csv(file):
    with open(file, 'rU') as csv_file:
        data = []
        reader = csv.reader(csv_file)
        for row in reader:
            data.append(map(rescale, row))
        return np.array(data)

def test_statistic(alice, pop_mean, sample_mean):
    alice_diff = np.dot(alice, sample_mean)
    pop_diff = np.dot(pop_mean, sample_mean)
    statistic = alice_diff - pop_diff
    variance = 0.
    for i in range(len(pop_mean)):
        variance += (sample_mean[i] ** 2.) * (1 - (pop_mean[i] ** 2.))
    return statistic, variance

def main():
    data = read_csv('problem_3_attributes.csv')
    population = data
    sample_without = data[50:]
    sample_with = data[:50]

    row = data[0]
    pop_mean = population.mean(0)
    sample_with_mean = sample_with.mean(0)
    sample_without_mean = sample_without.mean(0)

    print("with")
    statistic, variance = test_statistic(row, pop_mean, sample_with_mean)
    null_dist = stats.norm(0, math.sqrt(variance))
    res = null_dist.cdf(statistic)
    print(statistic, variance, res)

    print("without")
    statistic, variance = test_statistic(row, pop_mean, sample_without_mean)
    null_dist = stats.norm(0, math.sqrt(variance))
    res = null_dist.cdf(statistic)
    print(statistic, variance, res)

if __name__ == "__main__":
    main()
