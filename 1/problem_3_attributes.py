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

# prime number
P = 773

# random predicate generator
def random_predicate(row):
    sum = 0
    for idx, pub_key in enumerate(PUB):
        sum += np.random.randint(P) * row[pub_key]
    return (sum % P) % 2 == 1

# read and clean csv file
def read_csv(file):
    with open(file, 'rU') as csv_file:
        reader = csv.DictReader(csv_file)
        data = []
        for row in reader:
            if row['englishability'] == 'NA':
                row['englishability'] = 2
            if row['income'][-4:] == 'e+05':
                row['income'] = int(row['income'][:-4]) * 100000
            for key in row.keys():
                row[key] = int(row[key])
            data.append(row)
        return data

# generate m random attributes
def gen_attributes(row, m):
    result = []
    for i in range(m):
        if random_predicate(row):
            result.append(1.)
        else:
            result.append(0.)
    return result

# write derived attributes to csv file
def write_derived_attributes(sample, population):
    with open('problem_3_population.csv', 'wb') as pop_file:
        with open('problem_3_sample.csv', 'wb') as sam_file:
            attribute_bar = progressbar.ProgressBar(maxval=len(population),
                                                    widgets=[progressbar.Bar('=', '[', ']'),
                                                    ' ',
                                                    progressbar.Percentage(),
                                                    ' ',
                                                    progressbar.ETA()])
            attribute_bar.start()

            # create csv writers
            pop_writer = csv.writer(pop_file)
            sam_writer = csv.writer(sam_file)
            m = int(len(sample) ** 2.)

            # for each member of population, generate n^2 new attributes
            for idx, pop in enumerate(population):
                attribute_bar.update(idx)
                res = gen_attributes(pop, m)
                pop_writer.writerow(res)
                if pop in sample:
                    sam_writer.writerow(res)

            attribute_bar.finish()

def main():
    sample = read_csv('FultonPUMS5sample100.csv')
    population = read_csv('FultonPUMS5full.csv')
    write_derived_attributes(sample, population)

if __name__ == "__main__":
    main()
