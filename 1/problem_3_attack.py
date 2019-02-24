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

# public keys
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

# defense strategies
ROUNDING = 1
NOISE_ADDITION = 2
SUBSAMPLING = 3

# fixed defense parameters
ROUNDING_PARAM = 90
NOISE_ADDITION_PARAM = 40
SUBSAMPLING_PARAM = 10

# number of attributes to use
NUM_ATTRIBUTES = 1000

# prime number
P = 773

# sample and population lengths
SAMPLE_LEN = 100
POPULATION_LEN = 25766
POPULATION_SUBSET_LEN = 150

# false positive rate of 1/10n
P_VALUE = 1. / (10. * float(SAMPLE_LEN))

# read a csv file with just the means of the data
def read_mean_csv(file):
    with open(file, 'rU') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            return np.array(map(float, row[:NUM_ATTRIBUTES]))

# read a csv file with all rows of the data
def read_csv(file):
    with open(file, 'rU') as csv_file:
        data = []
        reader = csv.reader(csv_file)
        for row in reader:
            data.append(map(float, row[:NUM_ATTRIBUTES]))
        return np.array(data)

# perform a subsampled query
def get_subsampled_query(data, subsample_indices):
    result = np.zeros(len(data[0]))
    for idx, row in enumerate(data):
        if idx in subsample_indices:
            result += row
    return result

# perform a query with rounding defense
def rounding(data, R):
    result = np.zeros(len(data))
    for idx, val in enumerate(data):
        modulo = val % R
        if (modulo < R/2):
            result[idx] = val - modulo
        else:
            result[idx] = val + (R - modulo)
    return result

# perform a query with noise addition defense
def noise_addition(data, sigma):
    result = np.zeros(len(data))
    for idx, val in enumerate(data):
        result[idx] = val + np.random.normal(scale=sigma)
    return result

# perform a query with subsampling defense
def subsampling(data, t):
    subsample_indices = set(np.random.choice(len(data), size=t, replace=False))
    subsample = get_subsampled_query(data, subsample_indices)
    scale_factor = float(SAMPLE_LEN) / float(t)
    return subsample * scale_factor

# perform a query with various defense types and parameters
def query(data, defense_type=0):
    if (defense_type == SUBSAMPLING):
        return subsampling(data, SUBSAMPLING_PARAM) / SAMPLE_LEN
    elif (defense_type == NOISE_ADDITION):
        return noise_addition(data, NOISE_ADDITION_PARAM) / SAMPLE_LEN
    elif (defense_type == ROUNDING):
        return rounding(data, ROUNDING_PARAM) / SAMPLE_LEN
    else:
        return np.array(data) / SAMPLE_LEN

# test statistic as described in 2/8 lecture notes
def test_statistic(y, p, a):
    y_diff = y - p
    a_diff = a - p
    statistic = np.dot(y_diff, a_diff)
    variance = 0.
    for j in range(len(p)):
        variance += np.square(a[j] - p[j]) * p[j] * (1 - p[j])
    return statistic, variance

# perform experiment to determine true positive rate
def tpp_experiment(defense_type, num_queries, num_trials=20):
    global sample
    global sample_mean
    global sample_counts
    global population_mean
    global population_counts

    # count number of successes
    successes = 0

    # progress bar
    experiment_bar = progressbar.ProgressBar(maxval=num_trials,
                                         widgets=[progressbar.Bar('=', '[', ']'),
                                         ' ',
                                         progressbar.Percentage()])
    experiment_bar.start()

    # execute multiple trials to average
    for i in range(num_trials):
        experiment_bar.update(i)
        result_mean = np.zeros(len(sample_mean))

        # execute number of queries
        for _ in range(num_queries):
            if defense_type == SUBSAMPLING:
                result_mean += query(sample, defense_type)
            else:
                result_mean += query(sample_counts, defense_type)

        result_mean /= float(num_queries)

        # calculate test statistic and tail distance
        statistic, variance = test_statistic(
                              sample[np.random.randint(SAMPLE_LEN)],
                              population_mean,
                              result_mean)
        null_dst = stats.norm(0, math.sqrt(variance))
        percentile = null_dst.cdf(statistic)
        tail_distance = 0.5 - abs(percentile - 0.5)

        # determine if successful
        if (tail_distance < P_VALUE / 2.):
            successes += 1

    experiment_bar.finish()

    # calculate true positive rate
    true_positive_rate = float(successes) / float(num_trials)
    output = [defense_type, num_queries, true_positive_rate]

    return output

def true_positive_rate():
    global sample
    global sample_mean
    global sample_counts
    global population_mean
    global population_counts

    # read in data
    sample = read_csv('problem_3_sample.csv')

    # read in means
    sample_mean = read_mean_csv('problem_3_sample_mean.csv')
    sample_counts = sample_mean * SAMPLE_LEN
    population_mean = read_mean_csv('problem_3_population_mean.csv')
    population_counts = population_mean * POPULATION_LEN

    # write out the results to a csv file
    with open('problem_3_results.csv', 'wb') as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(['defense_type',
                         'num_queries',
                         'true_positive_rate'])

        # iterate through each defense type and number of queries
        for defense_type in range(1, 4):
            for num_queries in range(1, 2*SAMPLE_LEN+1):
                print("defense_type: " + str(defense_type) + " with num_queries " + str(num_queries))
                writer.writerow(tpp_experiment(defense_type, num_queries))

def false_positive_rate():
    global sample
    global population_subset
    global sample_mean
    global sample_counts
    global population_mean
    global population_counts

    print("performing false positive attack")

    # read in data
    sample = read_csv('problem_3_sample.csv')
    population_subset = read_csv('problem_3_population_subset.csv')

    # read in means and counts
    sample_mean = read_mean_csv('problem_3_sample_mean.csv')
    sample_counts = sample_mean * SAMPLE_LEN
    population_mean = read_mean_csv('problem_3_population_mean.csv')
    population_counts = population_mean * POPULATION_LEN

    # fix number of queries to 2n, and use noise addition defense
    false_positives = 0
    num_queries = 2 * SAMPLE_LEN
    defense_type = NOISE_ADDITION

    # display progress bar
    experiment_bar = progressbar.ProgressBar(maxval=10*SAMPLE_LEN,
                                         widgets=[progressbar.Bar('=', '[', ']'),
                                         ' ',
                                         progressbar.Percentage()])
    experiment_bar.start()

    # perform 10n queries on members of a previously generated random subset of population
    for i in range(10*SAMPLE_LEN):
        experiment_bar.update(i)
        result_mean = np.zeros(len(sample_mean))

        # execute number of queries
        for _ in range(num_queries):
            if defense_type == SUBSAMPLING:
                result_mean += query(sample, defense_type)
            else:
                result_mean += query(sample_counts, defense_type)
        result_mean /= float(num_queries)

        # calculate test static and tail distance
        statistic, variance = test_statistic(
                              population_subset[np.random.randint(POPULATION_SUBSET_LEN)],
                              population_mean,
                              result_mean)
        null_dst = stats.norm(0, math.sqrt(variance))
        percentile = null_dst.cdf(statistic)
        tail_distance = 0.5 - abs(percentile - 0.5)

        # count false positives
        if (tail_distance < P_VALUE / 2.):
            false_positives += 1

    experiment_bar.finish()

    # ideally, should be <= 1
    print("Over 10n queries, had " + str(false_positives) + " false positives.")

def main():
    true_positive_rate()
    false_positive_rate()

if __name__ == "__main__":
    main()
