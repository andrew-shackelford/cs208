"""
Andrew Shackelford
ashackelford@college.harvard.edu

CS 208 - Spring 2019
Homework 1, Problem 3
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

# defense strategies
ROUNDING = 1
NOISE_ADDITION = 2
SUBSAMPLING = 3

# read in result csv
def read_csv(file):
    with open(file, 'rU') as csv_file:
        reader = csv.DictReader(csv_file)
        rounding = []
        noise_addition = []
        subsampling = []
        for row in reader:
            if float(row['defense_type']) == ROUNDING:
                rounding.append(row)
            elif float(row['defense_type']) == NOISE_ADDITION:
                noise_addition.append(row)
            elif float(row['defense_type']) == SUBSAMPLING:
                subsampling.append(row)
        return rounding, noise_addition, subsampling

# plot a result graph
def plot_graph(data, attributes):
    # load data
    x, y = [], []
    for row in data:
        x.append(float(row['num_queries']))
        y.append(float(row[attributes['y_value']]))

    # convolve to smooth out noise
    x_prime = x[4:-4]
    y_prime = np.convolve(y, np.ones((9,))/9, mode='valid')

    # plot data
    plt.plot(x_prime, y_prime, '-o') 

    # add titles and labels
    plt.title('True Positive Rate for ' + attributes['title'] + ' Attack')
    plt.xlabel('Number of Queries')
    plt.ylabel('True Positive Rate (fraction of total)')

    # save to file and clear figure
    plt.savefig(attributes['output_file'])
    plt.clf()

def main():
    # read in results
    rounding, noise_addition, subsampling = read_csv('problem_3_results.csv')

    # plot graphs for different defense types
    plot_graph(rounding, {
    'output_file' : 'figures/problem_3_rounding_true_positive_rate.png',
    'title': 'Rounding',
    'y_value' : 'true_positive_rate'
                         })
    plot_graph(noise_addition, {
    'output_file' : 'figures/problem_3_noise_addition_true_positive_rate.png',
    'title': 'Noise Addition',
    'y_value' : 'true_positive_rate'
                               })
    plot_graph(subsampling, {
   'output_file' : 'figures/problem_3_subsampling_true_positive_rate.png',
   'title': 'Subsampling',
   'y_value' : 'true_positive_rate'
                            })

if __name__ == "__main__":
    main()