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

def random_predicate(row):
    sum = 0
    for idx, pub_key in enumerate(PUB):
        sum += np.random.randint(P) * row[pub_key]
    return (sum % P) % 2 == 1

def read_csv(file, m):
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
            data.append(gen_attributes(row, m))
        return data

def gen_attributes(row, m):
    result = []
    for i in range(m):
        if random_predicate(row):
            result.append(1.)
        else:
            result.append(0.)
    return result

def main():
    data = read_csv('FultonPUMS5sample100.csv', 500)
    with open('problem_3_attributes.csv', 'wb') as out_csv:
        writer = csv.writer(out_csv)
        for row in data:
            writer.writerow(row)

if __name__ == "__main__":
    main()
