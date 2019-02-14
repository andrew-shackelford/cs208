"""
Andrew Shackelford
ashackelford@college.harvard.edu

CS 208 - Spring 2019
Homework 1, Problem 2
"""

import csv
import matplotlib.pyplot as plt

ROUNDING = 1
NOISE_ADDITION = 2
SUBSAMPLING = 3

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


def plot_graph(data, attributes):
    x, y = [], []
    for row in data:
        x.append(float(row['defense_factor']))
        y.append(float(row[attributes['y_value']]))

    plt.plot(x, y, '-o')

    if attributes['y_value'] == 'success_rate':     
        plt.title('Success Rate for ' + attributes['title'] + ' Attack')
        plt.xlabel('Parameter (' + attributes['parameter'] + ')')
        plt.ylabel('Success Rate (fraction of total)')
    else:
        plt.title('Root Mean Squared Error for ' + attributes['title'] + ' Attack')
        plt.xlabel('Parameter (' + attributes['parameter'] + ')')
        plt.ylabel('Root Mean Squared Error')

    plt.savefig(attributes['output_file'])
    plt.clf()


def main():
    rounding, noise_addition, subsampling = read_csv('problem_2_results.csv')
    plot_graph(rounding, {
                          'output_file' : 'figures/problem_2_rounding_success_rate.png',
                          'title': 'Rounding',
                          'parameter' : 'R',
                          'y_value' : 'success_rate'
                         })
    plot_graph(rounding, {
                          'output_file' : 'figures/problem_2_rounding_error.png',
                          'title': 'Rounding',
                          'parameter' : 'R',
                          'y_value' : 'root_mse'
                         })
    plot_graph(noise_addition, {
                                'output_file' : 'figures/problem_2_noise_addition_success_rate.png',
                                'title': 'Noise Addition',
                                'parameter' : r'$\sigma$',
                                'y_value' : 'success_rate'
                               })
    plot_graph(noise_addition, {
                                'output_file' : 'figures/problem_2_noise_addition_error.png',
                                'title': 'Noise Addition',
                                'parameter' : r'$\sigma$',
                                'y_value' : 'root_mse'
                               })
    plot_graph(subsampling, {
                             'output_file' : 'figures/problem_2_subsampling_success_rate.png',
                             'title': 'Subsampling',
                             'parameter' : 't',
                             'y_value' : 'success_rate'
                            })
    plot_graph(subsampling, {
                             'output_file' : 'figures/problem_2_subsampling_error.png',
                             'title': 'Subsampling',
                             'parameter' : 't',
                             'y_value' : 'root_mse'
                            })


if __name__ == "__main__":
    main()