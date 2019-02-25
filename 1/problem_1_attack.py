"""
Andrew Shackelford
ashackelford@college.harvard.edu

CS 208 - Spring 2019
Homework 1, Problem 1
"""

import csv
import numpy as np
from collections import Counter

TOTAL = 25766 * 20 # number of rows in 5% sample * 20 to get total

# load and clean sample csv
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

# count number of entries with each possible value
def get_counts(data):
    ret = {}
    for row in data:
        for key, value in row.iteritems():
            entry = ret.get(key, {})
            entry[value] = entry.get(value, 0) + 20 # use 20 instead of 1, since 5% sample
            ret[key] = entry
    return ret

# convert counts to proportions
def get_proportions(counts):
    for attr, breakdown in counts.iteritems():
        for value, number in breakdown.iteritems():
            breakdown[value] = float(number) / float(TOTAL)
    return counts

# generate new random individuals with properties of dataset
def generate_random_individuals(proportions, desired_attributes):
    generated = ["" for i in range(TOTAL)]

    # for each attribute
    for attr, breakdown in proportions.iteritems():
        # that we plan to use in our reconstruction attack
        if attr not in desired_attributes:
            continue

        # get the possible options as well as the proportions for each option
        keys = breakdown.keys()
        values = breakdown.values()

        # represent each individual as a concatenated string of each attribute
        for idx, individual in enumerate(generated):
            generated[idx] = individual + str(np.random.choice(keys, p=values))

    return generated

# count the number of individuals that are unique
def count_unique_individuals(generated):
    counter = Counter(generated)
    num_unique = 0
    for element, number in counter.most_common():
        if number == 1:
            num_unique += 1
    return num_unique

def main():
    data = read_csv('FultonPUMS5full.csv')
    counts = get_counts(data)
    proportions = get_proportions(counts)

    # attributes based on only publicly available information
    desired_attributes = ['sex', 'latino', 'black', 'asian', 'married', 'divorced']
    generated = generate_random_individuals(proportions, desired_attributes)
    print(count_unique_individuals(generated))

    # attributes based on tax returns
    desired_attributes = ['sex', 'latino', 'black', 'asian', 'married', 'divorced', 'puma', 'children', 'employed', 'income']
    generated = generate_random_individuals(proportions, desired_attributes)
    print(count_unique_individuals(generated))

    # attributes based on voter registration database
    desired_attributes = ['sex', 'latino', 'black', 'asian', 'married', 'divorced', 'puma', 'age']
    generated = generate_random_individuals(proportions, desired_attributes)
    print(count_unique_individuals(generated))

    # attributes based on tax returns + voter registration database
    desired_attributes = ['sex', 'latino', 'black', 'asian', 'married', 'divorced', 'puma', 'children', 'employed', 'income', 'age']
    generated = generate_random_individuals(proportions, desired_attributes)
    print(count_unique_individuals(generated))

if __name__ == "__main__":
    main()
