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

ROUNDING = 1
NOISE_ADDITION = 2
SUBSAMPLING = 3

P = 961845637

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

def get_query(data, predicate):
    result = 0.
    x = np.zeros(len(data))
    for idx, row in enumerate(data):
        if predicate(row):
            result += row['uscitizen']
            x[idx] = 1.
    return result, x

def rounding(data, predicate, R):
    result, x = get_query(data, predicate)
    modulo = result % R
    if (modulo < R/2):
        return result - modulo, x
    else:
        return result + (R - modulo), x

def noise_addition(data, predicate, sigma):
    result, x = get_query(data, predicate)
    result = result + np.random.normal(scale=sigma)
    return result, x

def subsampling(data, predicate, t):
    subsample_indices = np.random.choice(len(data), size=t, replace=False)
    subsample = []
    x = np.zeros(len(data))
    for index in subsample_indices:
        subsample.append(data[index])
        x[index] = 1.
    result, _ = get_query(subsample, predicate)
    scale_factor = float(len(data)) / float(t)
    return result * scale_factor, x

def query(data, predicate, defense_type=0, defense_factor=0.):
    if (defense_type == SUBSAMPLING):
        return subsampling(data, predicate, defense_factor)
    elif (defense_type == NOISE_ADDITION):
        return noise_addition(data, predicate, defense_factor)
    elif (defense_type == ROUNDING):
        return rounding(data, predicate, defense_factor)
    else:
        return get_query(data, predicate)

def random_predicate(row):
    sum = 0
    for pub_key in PUB:
        sum += np.random.randint(P) * row[pub_key]
    return (sum % P) % 2 == 1

def experiment(data, n, defense_type, defense_factor):
    true_predicate = lambda row: True

    experiment_bar = progressbar.ProgressBar(maxval=10*n, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    experiment_bar.start()
    for i in range(10*n):
        experiment_bar.update(i)
        y, x = query(data, true_predicate, defense_type, defense_factor)
        if i == 0:
            Ys, Xs = y, x
        else:
            Ys, Xs = np.vstack((Ys, y)), np.vstack((Xs, x))
    experiment_bar.finish()

    Betas, residuals, ranks, s = np.linalg.lstsq(Xs, Ys)

    successes = 0
    false_positives = 0
    false_negatives = 0
    total_squared_error = 0.
    for index, estimate in enumerate(Betas):
        total_squared_error += (estimate - float(data[index]['uscitizen'])) ** 2.
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

    success_rate = float(successes) / float(len(data))
    mse = total_squared_error / float(n)
    root_mse = math.sqrt(mse)

    output = [defense_type,
              defense_factor,
              successes,
              false_positives,
              false_negatives,
              success_rate,
              root_mse]

    return output

def main():
    data = read_csv('FultonPUMS5.csv')
    n = len(data)

    with open('testing.csv', 'wb') as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(['defense_type',
                         'defense_factor',
                         'successes',
                         'false_positives',
                         'false_negatives',
                         'success_rate',
                         'root_mse'])

        for defense_type in range(1, 4):
            for defense_factor in np.linspace(1, n, 50):
                defense_factor = int(defense_factor)
                print("defense_type: " + str(defense_type) + " with defense_factor " + str(defense_factor))
                average = np.zeros(7)
                for i in range(10):
                    print("Trial " + str(i+1) + " of 10")
                    average += experiment(data, n, defense_type, defense_factor)
                average /= 10
                writer.writerow(average)

if __name__ == "__main__":
    main()
