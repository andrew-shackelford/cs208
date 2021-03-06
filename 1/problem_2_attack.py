"""
Andrew Shackelford
ashackelford@college.harvard.edu

CS 208 - Spring 2019
Homework 1, Problem 2
"""

import numpy as np
from scipy import stats
import math
import csv
import progressbar

# public attributes
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

# prime number
P = 773

# load and clean sample csv
def read_csv(file):
    with open(file, 'rU') as csv_file:
        reader = csv.DictReader(csv_file)
        data = []
        for row in reader:
            if row['englishability'] == 'NA':
                row['englishability'] = 2
            if row['income'] == '1e+05':
                row['income'] = 100000
            for key in row.keys():
                row[key] = int(row[key])
            data.append(row)
        return data

# perform a query with no defense
def get_query(data, predicate):
    result = 0.
    x = np.zeros(len(data))
    for idx, row in enumerate(data):
        if predicate(row):
            result += row['uscitizen']
            x[idx] = 1.
    return result, x

# perform a subsampled query
def get_subsampled_query(data, predicate, subsample_indices):
    result = 0.
    x = np.zeros(len(data))
    for idx, row in enumerate(data):
        if predicate(row):
            x[idx] = 1.
            if idx in subsample_indices:
                result += row['uscitizen']
    return result, x

# perform a query with rounding defense
def rounding(data, predicate, R):
    result, x = get_query(data, predicate)
    modulo = result % R
    if (modulo < R/2):
        return result - modulo, x
    else:
        return result + (R - modulo), x

# perform a query with noise addition defense
def noise_addition(data, predicate, sigma):
    result, x = get_query(data, predicate)
    result = result + np.random.normal(scale=sigma)
    return result, x

# perform a query with subsampling defense
def subsampling(data, predicate, t):
    subsample_indices = set(np.random.choice(len(data), size=t, replace=False))
    result, x = get_subsampled_query(data, predicate, subsample_indices)
    scale_factor = float(len(data)) / float(t)
    return result * scale_factor, x

# perform a query with various defense types and parameters
def query(data, predicate, defense_type=0, defense_factor=0.):
    if (defense_type == SUBSAMPLING):
        return subsampling(data, predicate, defense_factor)
    elif (defense_type == NOISE_ADDITION):
        return noise_addition(data, predicate, defense_factor)
    elif (defense_type == ROUNDING):
        return rounding(data, predicate, defense_factor)
    else:
        return get_query(data, predicate)

# generate a random vector for use in creating random subsets
def gen_random_vector(length):
    global random_vector
    random_vector = []
    for i in range(length):
        random_vector.append(np.random.randint(P))

# a random predicate to generate random subsets
def random_predicate(row):
    global random_vector
    sum = 0
    for idx, pub_key in enumerate(PUB):
        sum += random_vector[idx] * row[pub_key]
    return (sum % P) % 2 == 1

# run an experiment given data, n, a defense type and parameter
def experiment(data, n, defense_type, defense_factor):
    experiment_bar = progressbar.ProgressBar(maxval=2*n,
                                             widgets=[progressbar.Bar('=', '[', ']'),
                                             ' ',
                                             progressbar.Percentage()])
    experiment_bar.start()

    # reset total squared error
    total_squared_error = 0.

    # perform 2n queries
    for i in range(2*n):
        experiment_bar.update(i)

        gen_random_vector(len(PUB)) # generate a new subset for each query

        y, x = query(data, random_predicate, defense_type, defense_factor)
        truth, _ = query(data, random_predicate)

        total_squared_error += np.square(y - truth) # calculate error

        if i == 0:
            Ys, Xs = y, x
        else:
            Ys, Xs = np.vstack((Ys, y)), np.vstack((Xs, x))

    experiment_bar.finish()

    # perform least squares regression
    Betas, residuals, ranks, s = np.linalg.lstsq(Xs, Ys)

    # calculate number of successes
    successes = 0
    false_positives = 0
    false_negatives = 0
    for index, estimate in enumerate(Betas):
        if estimate >= 0.5:
            if data[index]['uscitizen']:
                successes += 1
            else:
                false_positives += 1
        else:
            if data[index]['uscitizen']:
                false_negatives += 1
            else:
                successes += 1

    # calculate success rate
    success_rate = float(successes) / float(len(data))

    # calculate root mean squared error
    mse = total_squared_error / float(n)
    root_mse = math.sqrt(mse)

    # create an output row for the csv
    output = [defense_type,
              defense_factor,
              successes,
              false_positives,
              false_negatives,
              success_rate,
              root_mse]

    return output

def main():
    # read in the 100-person sample
    data = read_csv('FultonPUMS5sample100.csv')
    n = len(data)

    # write out the results to a csv file
    with open('problem_2_results.csv', 'wb') as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(['defense_type',
                         'defense_factor',
                         'successes',
                         'false_positives',
                         'false_negatives',
                         'success_rate',
                         'root_mse'])

        # iterate through each defense type and parameter
        for defense_type in range(1, 4):
            for defense_factor in range(1, n+1):
                defense_factor = int(defense_factor)
                print("defense_type: " + str(defense_type) + " with defense_factor " + str(defense_factor))
                average = np.zeros(7)
                for i in range(10): # run 10 trials
                    print("Trial " + str(i+1) + " of 10")
                    average += experiment(data, n, defense_type, defense_factor)
                average /= 10
                writer.writerow(average)

if __name__ == "__main__":
    main()
