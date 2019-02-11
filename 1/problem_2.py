"""
Andrew Shackelford
ashackelford@college.harvard.edu

CS 208 - Spring 2019
Homework 1, Problem 2
"""

import numpy as np
import math
import csv

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
    for row in data:
        if predicate(row):
            result += row['uscitizen']
    return result

def rounding(data, predicate, R):
    result = get_query(data, predicate)
    modulo = result % R
    if (modulo < R/2):
        return result - modulo
    else:
        return result + (R - modulo)

def noise_addition(data, predicate, sigma):
    return get_query(data, predicate) + np.random.normal(scale=sigma)

def subsampling(data, predicate, t):
    subsample_indices = np.random.choice(len(data), size=t, replace=False)
    subsample = []
    for index in subsample_indices:
        subsample.append(data[index])
    result = get_query(subsample, predicate)
    scale_factor = float(len(data)) / float(t)
    return result * scale_factor

def query(data, predicate, defense=0, defense_factor=0.):
    if (defense == SUBSAMPLING):
        return subsampling(data, predicate, defense_factor)
    elif (defense == NOISE_ADDITION):
        return noise_addition(data, predicate, defense_factor)
    elif (defense == ROUNDING):
        return rounding(data, predicate, defense_factor)
    else:
        return get_query(data, predicate)

def random_predicate(row):
    sum = 0
    for pub_key in PUB:
        sum += np.random.randint(P) * row[pub_key]
    return (sum % P) % 2 == 1

def main():
    data = read_csv('FultonPUMS5.csv')
    total_predicate = lambda row: True

    print(query(data, total_predicate), query(data, total_predicate, NOISE_ADDITION, 1))
    print(query(data, random_predicate), query(data, random_predicate, NOISE_ADDITION, 1))

if __name__ == "__main__":
    main()