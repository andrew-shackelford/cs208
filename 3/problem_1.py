import numpy as np
from scipy.special import softmax as softmax
import csv
import pickle
import matplotlib.pyplot as plt
import operator

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

# get the data split by a variable
def get_variable_data_by(data, variable, by):
    ret = {}
    for row in data:
        lst = ret.get(row[by], [])
        lst.append(row[variable])
        ret[row[by]] = lst
    return ret

# get the data split by a variable, clipped to a lower and upper bound
def get_variable_data_by_clip(data, variable, by, clip_l, clip_u):
    ret = get_variable_data_by(data, variable, by)
    for key in ret.keys():
        ret[key] = np.clip(ret[key], clip_l, clip_u)
    return ret

# calculate the sum of a dataset's points that are within certain bounds
def sum_within_bounds(x, lower, upper):
    sum_x = 0.
    for x_i in x:
        if x_i >= lower and x_i <= upper:
            sum_x += x_i
    return sum_x 

# utility function for any percentile
def utility(x, y, t):
    left_index = np.searchsorted(x, y, side='left')
    right_index = np.searchsorted(x, y, side='right')

    left_dist = np.abs(t * x.shape[0] - left_index)
    right_dist = np.abs(t * x.shape[0] - right_index)

    dist = min(left_dist, right_dist)
    # account for bias of left_index due to the way numpy insertion works
    if left_index <= t * x.shape[0]:
        dist += 1
    return x.shape[0] - dist

# mechanism for part a
def part_a(x, D, epsilon):
    p_05, p_95 = np.percentile(x, [5., 95.])

    # calculate mean for points within true percentiles
    n = float(len(x))
    sum_x = sum_within_bounds(x, p_05, p_95)
    mean_x = sum_x * (1. / (0.9 * n))

    # add sufficient noise
    scale = float(D) / (float(epsilon) * 0.9 * n)
    noise = np.random.laplace(scale=scale)

    return mean_x + noise

# mechanism for part c
def part_c(x, D, epsilon, t):
    global global_probabilities
    utilities = {}

    # create versions of x
    x_unique = np.unique(x)
    x_hash = hash(x.tostring())
    x_sorted = np.sort(x)

    # create variables for use in mechanism
    cur_x_idx = 0
    unique_count = x_unique.shape[0]
    unique_max = x_unique[-1]

    # only recalculate probabilities if needed
    if (x_hash, D, epsilon, t) not in global_probabilities:
        x = np.array(x)
        probabilities = []

        # use mechanism for sorted x that allows us to reduce number of weights calculated
        for d in range(D+1):
            if cur_x_idx >= unique_count:
                break
            if d >= x_unique[cur_x_idx]:
                utilities[d] = utility(x_sorted, d, t) * float(epsilon) / 2.
                cur_x_idx += 1

            probabilities.append(utilities[x_unique[cur_x_idx-1]])

        probabilities = softmax(probabilities)
        global_probabilities[(x_hash, D, epsilon, t)] = probabilities

    # calculate release
    p = global_probabilities[(x_hash, D, epsilon, t)]
    release = np.random.choice([d for d in range(unique_max+1)], p=p)
    return release

# mechanism for part d
def part_d(x, D, epsilon,):
    # calculate differentially private 5th and 95th percentiles
    local_ep = float(epsilon) / 3.
    p_05 = part_c(x, D, local_ep, 0.05)
    p_95 = part_c(x, D, local_ep, 0.95)

    # calculate sum within bounds for differentially private estimates
    n = float(len(x))
    sum_x = sum_within_bounds(x, p_05, p_95)
    mean_x = sum_x * (1. / (0.9 * n))

    # add noise
    scale = (3. * (p_95 - p_05)) / (0.9 * float(epsilon) * n)
    noise = np.random.laplace(scale=scale)

    return mean_x + noise

# calculate a differentially private mean using the standard Laplace mechanism
def standard_laplace_mean(x, D, epsilon):
    scale = float(epsilon) * float(D) / float(len(x))
    return np.mean(x) + np.random.laplace(scale=scale)

# function to perform the experiment for part f
def part_f_experiment():
    D = 1000000
    epsilon = 1.
    num_trials = 100

    # read in data, get both income and clipped income
    data = read_csv('MaPUMS5full.csv')
    income = get_variable_data_by(data, 'income', 'puma')
    clipped_income = get_variable_data_by_clip(data, 'income', 'puma', 0, D)

    true_means, trimmed_means, laplace_means = [], [], []
    true_mean = {}

    # calculate the true mean for each PUMA
    for key, array in income.iteritems():
        true_mean[key] = np.mean(array)
    true_means.append(true_mean)

    # for each trial
    for i in range(num_trials):
        trimmed_mean, laplace_mean = {}, {}

        # calculate the Trimmed and Laplace mean for each PUMA
        for key, array in clipped_income.iteritems():
            trimmed_mean[key] = part_d(array, D, epsilon)
            laplace_mean[key] = standard_laplace_mean(array, D, epsilon)

        trimmed_means.append(trimmed_mean)
        laplace_means.append(laplace_mean)

    # write the results out to file
    with open('true_means.pkl', 'wb') as f:
        pickle.dump(true_means, f)
    with open('trimmed_means.pkl', 'wb') as f:
        pickle.dump(trimmed_means, f)
    with open('laplace_means.pkl', 'wb') as f:
        pickle.dump(laplace_means, f)

def part_f_analysis():
    # read in the results
    with open('true_means.pkl', 'rb') as f:
        true_means = pickle.load(f)
    with open('trimmed_means.pkl', 'rb') as f:
        trimmed_means = pickle.load(f)
    with open('laplace_means.pkl', 'rb') as f:
        laplace_means = pickle.load(f)

    # create dictionaries for RMSE calculations
    trimmed_rmse = {puma:0. for puma in sorted(true_means[0].keys())}
    laplace_rmse = {puma:0. for puma in sorted(true_means[0].keys())}

    # sum the squared error for trimmed means
    for trimmed_mean in trimmed_means:
        for puma, mean in trimmed_mean.iteritems():
            trimmed_rmse[puma] += np.square(true_means[0][puma] - mean)

    # sum the squared error for Laplace means
    for laplace_mean in laplace_means:
        for puma, mean in laplace_mean.iteritems():
            laplace_rmse[puma] += np.square(true_means[0][puma] - mean)

    # take the mean and sqrt of trimmed SE
    for puma, rmse in trimmed_rmse.iteritems():
        trimmed_rmse[puma] = np.sqrt(rmse / float(len(trimmed_means)))

    # take the mean and sqrt of Laplace SE
    for puma, rmse in laplace_rmse.iteritems():
        laplace_rmse[puma] = np.sqrt(rmse / float(len(laplace_means)))

    # sort the PUMAs by their true means
    sorted_true_means = sorted(true_means[0].items(), key=operator.itemgetter(1))

    # print out LaTeX formatted results table
    print("\\hline")
    print("\\textbf{PUMA} & \\textbf{True Mean} \\textbf{Laplace RMSE} & \\textbf{Trimmed RMSE} \\\\ \\hline")
    for puma, _ in sorted_true_means:
        print(str(puma) + " & " +
              str(int(round(true_means[0][puma]))) + " & " +
              str(int(round(laplace_rmse[puma]))) + " & " +
              str(int(round(trimmed_rmse[puma]))) + " \\\\ \\hline")

# function to graph a boxplot
def graph(true_means, dp_means, puma_keys, title, file, xlim_l, xlim_u):
    fig, ax = plt.subplots(figsize=(17, 22))

    # create the boxplot data
    dp_boxplots = [[] for i in range(len(puma_keys))]
    for i in range(len(dp_means)):
        for j in range(len(puma_keys)):
            dp_boxplots[j].append(dp_means[i][puma_keys[j]])

    # draw the true mean data
    for j, puma in enumerate(puma_keys):
        ax.plot(true_means[0][puma], j+1, marker='o', color='blue')

    # draw the boxplot
    ax.boxplot(dp_boxplots, labels=puma_keys, vert=False)

    # add labels and save
    plt.ylabel('PUMA')
    plt.xlabel('Income')
    plt.title(title)
    plt.xlim(xlim_l, xlim_u)
    plt.savefig(file, bbox_inches='tight', dpi=300)

def part_f_graph():
    # load the data
    with open('true_means.pkl', 'rb') as f:
        true_means = pickle.load(f)
    with open('trimmed_means.pkl', 'rb') as f:
        trimmed_means = pickle.load(f)
    with open('laplace_means.pkl', 'rb') as f:
        laplace_means = pickle.load(f)

    # create variables
    sorted_true_means = sorted(true_means[0].items(), key=operator.itemgetter(1))
    puma_keys = [puma for puma, _ in sorted_true_means]

    # graph trimmed means
    graph(true_means=true_means,
          dp_means=trimmed_means,
          puma_keys=puma_keys,
          title="Boxplots of Trimmed Mean DP Releases",
          file="trimmed_means_graph.png",
          xlim_l=15000,
          xlim_u=65000)

    # graph Laplace means
    graph(true_means=true_means,
          dp_means=laplace_means,
          puma_keys=puma_keys,
          title="Boxplots of Laplace Mean DP Releases",
          file="laplace_means_graph.png",
          xlim_l=15000,
          xlim_u=65000)

def main():
    # create global probabilities dictionary
    global global_probabilities
    global_probabilities = {}

    # perform experiment, analysis, and graph
    part_f_experiment()
    part_f_analysis()
    part_f_graph()

if __name__ == "__main__":
    main()