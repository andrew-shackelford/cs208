import csv
import numpy as np
import matplotlib.pyplot as plt

# read in the csv of psi epsilon_0 values
def read_csv(infile):
    with open(infile, 'rb') as f:
        next(f) # skip first line
        reader = csv.reader(f)
        x, y = [], []
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return x, y

# return the epsilon_0 under basic composition
def basic_composition(epsilon, k):
    return float(epsilon) / float(k)

# return the epsilon_0 under advanced composition
def advanced_composition(epsilon, delta, k):
    return float(epsilon) / np.sqrt(2 * float(k) * np.log(1./float(delta)))

# take an epsilon value and convert it to the appropriate laplace
# standard deviation for a count statistic, that is, invert it
def epsilon_to_std_dev(x):
    return 1. / float(x)

def main():
    # load and generate data
    psi_x, psi_y = read_csv('psi_epsilon.csv')
    basic_x, basic_y = [], []
    advanced_x, advanced_y = [], []
    for k in range(1, 501):
        basic_x.append(k)
        advanced_x.append(k)
        basic_y.append(basic_composition(1, k))
        advanced_y.append(advanced_composition(1, np.power(10., -9.), k))

    # convert epsilons to standard deviations
    basic_y = map(epsilon_to_std_dev, basic_y)
    advanced_y = map(epsilon_to_std_dev, advanced_y)
    psi_y = map(epsilon_to_std_dev, psi_y)

    # plot wide scale graph
    plt.plot(basic_x, basic_y, label='Basic')
    plt.plot(advanced_x, advanced_y, label='Advanced')
    plt.plot(psi_x, psi_y, label='PSI')
    plt.legend(loc='upper left')
    plt.title('Basic vs. Advanced vs. Optimal Composition Theorem (PSI)')
    plt.xlabel(r'$k$')
    plt.ylabel('Standard Deviation of Laplace Noise Added')
    plt.savefig('problem_2_wide.png')
    plt.clf()

    # plot close scale graph
    plt.plot(basic_x, basic_y, label='Basic')
    plt.plot(advanced_x, advanced_y, label='Advanced')
    plt.plot(psi_x, psi_y, label='PSI')
    plt.legend(loc='upper left')
    plt.title('Basic vs. Advanced vs. Optimal Composition Theorem (PSI)')
    plt.xlabel(r'$k$')
    plt.ylabel('Standard Deviation of Laplace Noise Added')
    plt.xlim((0, 50))
    plt.ylim((0, 50))
    plt.savefig('problem_2_close.png')

    # print out points where psi and advanced beat basic 
    advanced_found = False
    psi_found = False
    for k in range(1, 501):
        # account for floating point imprecision by using 0.0000001

        if basic_y[k-1] - advanced_y[k-1] > 0.0000001 and not advanced_found:
            print("advanced surpasses basic at k = " + str(k))
            advanced_found = True

        if basic_y[k-1] - psi_y[k-1] > 0.0000001 and not psi_found:
            print("psi surpasses basic at k = " + str(k))
            psi_found = True

if __name__ == "__main__":
    main()