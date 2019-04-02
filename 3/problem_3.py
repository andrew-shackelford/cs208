import numpy as np
import math
import csv

# load and clean sample csv
def read_csv(file, size=-1):
    with open(file, 'rU') as f:
        reader = csv.DictReader(f)
        data = []
        i = 0
        for row in reader:
            if i == size:
                break
            if row['englishability'] == 'NA':
                row['englishability'] = 2
            if row['income'][-4:] == 'e+05':
                row['income'] = int(row['income'][:-4]) * 100000
            for key in row.keys():
                row[key] = int(row[key])

            # change income to log(income)
            if row['income'] < 1:
                row['income'] = 1
            row['income'] = np.log(row['income'])

            data.append(row)
            i += 1
        return data

# write out histogram csv
def write_histogram_csv(results, file):
    with open(file, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['income_bound_l', 'income_bound_r', 'educ_bound_l', 'educ_bound_r', 'age_bound_l', 'age_bound_r', 'value'])
        for row in results:
            writer.writerow(row)

# read in histogram csv
def read_histogram_csv(file):
    probabilities, bounds = [], []
    with open(file, 'rb') as f:
        reader = csv.DictReader(f)
        for row in reader:
            probabilities.append(float(row['value']))
            bound = [row['income_bound_l'], row['income_bound_r'], row['educ_bound_l'], row['educ_bound_r'], row['age_bound_l'], row['age_bound_r']]
            bounds.append(map(float, bound))
    return probabilities, bounds

# write out synthetic data csv
def write_data_csv(results, file):
    with open(file, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['income', 'educ', 'age'])
        for row in results:
            writer.writerow(row)

# read in synthetic data csv as Xs and Ys
def read_data_csv(file):
    Xs, Ys = [], []
    with open(file, 'rb') as f:
        reader = csv.DictReader(f)
        for row in reader:
            Xs.append([1., float(row['educ']), float(row['age'])])
            Ys.append(float(row['income']))
    return Xs, Ys

# write out beta coefficient csv
def write_coefficient_csv(results, file):
    with open(file, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['beta_0', 'beta_1', 'beta_2'])
        for row in results:
            writer.writerow(row)

# write out mse (bias and variance) csv
def write_mse_csv(results, file):
    with open(file, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'bias_squared_0', 'bias_squared_1', 'bias_squared_2', 'variance_0', 'variance_1', 'variance_2'])
        for row in results:
            writer.writerow(row)

# clip a certain variable
def clip_data(data, variable, l, u):
    for row in data:
        if row[variable] < l:
            row[variable] = l
        if row[variable] > u:
            row[variable] = u

# get the bins for a given lower bound, upper bound, and number of bins
def get_bins(lower, upper, nbins):
    bins = np.linspace(lower, upper, num=nbins+1, dtype=int)
    bins = bins.astype(float)
    bins[-1] = bins[-1] + 0.0000001 # granularity
    return bins

# get the three-way histogram for a dataset
def get_histogram(data, income, educ, age, epsilon):
    # break down income, educ, and age into bounds and number of bins
    income_l, income_u, num_income_bins = income
    educ_l, educ_u, num_educ_bins = educ
    age_l, age_u, num_age_bins = age

    # generate the individual bins
    income_bins = get_bins(income_l, income_u, num_income_bins)
    educ_bins = get_bins(educ_l, educ_u, num_educ_bins)
    age_bins = get_bins(age_l, age_u, num_age_bins)

    # create the three-way bins
    bins = []
    for income_bin in income_bins:
        for educ_bin in educ_bins:
            for age_bin in age_bins:
                bins.append((income_bin, educ_bin, age_bin))
    num_bins = len(bins)

    # clip the data
    clip_data(data, 'income', income_l, income_u)
    clip_data(data, 'educ', educ_l, educ_u)
    clip_data(data, 'age', age_l, age_u)

    # calculate scale
    sensitivity = 2
    scale = sensitivity / epsilon

    true_dct = {}

    for row in data:
        # calculate which three-way bin a row should be in
        income_idx = int(np.floor((row['income']-income_l) * num_income_bins / (income_u - income_l)))
        educ_idx = row['educ'] - 1
        age_idx = int(np.floor((row['age']-age_l) * num_age_bins / (age_u - age_l)))

        # fix the granularity issues caused by upper bounds
        if row['income'] == income_u:
            income_idx = num_income_bins - 1
        if row['age'] == age_u:
            age_idx = num_age_bins - 1

        # calculate the bin bounds
        income_bound_l, income_bound_r = income_bins[income_idx], income_bins[income_idx+1]
        educ_bound_l, educ_bound_r = educ_bins[educ_idx], educ_bins[educ_idx+1]
        age_bound_l, age_bound_r = age_bins[age_idx], age_bins[age_idx+1]

        # enter the true results into the dictionary
        true_dct[(income_bound_l, income_bound_r, educ_bound_l, educ_bound_r, age_bound_l, age_bound_r)] = \
            true_dct.get((income_bound_l, income_bound_r, educ_bound_l, educ_bound_r, age_bound_l, age_bound_r), 0) + 1

    true_results, dp_results = [], []

    # calculate the true and DP results as rows
    for key, val in true_dct.iteritems():
        true_results.append((key[0], key[1], key[2], key[3], key[4], key[5], val))
        dp_results.append((key[0], key[1], key[2], key[3], key[4], key[5], val + np.random.laplace(scale=scale)))

    return dp_results, true_results

# Generate and write out the histograms
def generate_histograms():
    print("Generating histogram")
    data = read_csv('MaPUMS5full.csv')

    epsilon = 1.
    #format: lower, upper, num_bins
    income = 0, 12, 12 # lower bound of exp(0) = 1, upper bound of exp(12) = 162754
    educ = 1, 17, 16
    age = 0, 100, 20

    dp_results, true_results = get_histogram(data, income, educ, age, epsilon)

    write_histogram_csv(dp_results, 'dp_results.csv')
    write_histogram_csv(true_results, 'true_results.csv')

def generate_synthetic_data():
    print("Generating synthetic data")
    N = 241830

    # read in histogram results, clip the negative probabilities created by the DP noise, and normalize them
    probabilities, bounds = read_histogram_csv('dp_results.csv')
    probabilities = np.clip(probabilities, 0, None)
    probabilities = probabilities / np.linalg.norm(probabilities, ord=1)

    # generate the histogram indices to be used in the bootstrapped synthetic data
    indices = np.random.choice(len(bounds), N, replace=True, p=probabilities)

    synthetic_bounds = np.array(bounds)[indices, :]

    # generate the synthetic data
    synthetic_data = []
    for bounds in synthetic_bounds:
        income_bound_l, income_bound_r, educ_bound_l, educ_bound_r, age_bound_l, age_bound_r = bounds
        income = np.random.uniform(income_bound_l, income_bound_r)
        educ = educ_bound_l
        age = np.random.uniform(age_bound_l, age_bound_r)
        synthetic_data.append((income, educ, age))

    # write the results out to csv
    write_data_csv(synthetic_data, 'synthetic_data.csv')

# write out a csv of the true data in our desired format
def generate_true_data():
    print("Generating true data")
    data = read_csv('MaPUMS5full.csv')
    true_data = []
    for row in data:
        true_data.append((row['income'], row['educ'], row['age']))

    write_data_csv(true_data, 'true_data.csv')

# write out a csv of a bootstrap of the true data in our desired format
def generate_bootstrap_data():
    print("Generating bootstrap data")
    data = read_csv('MaPUMS5full.csv')
    indices = np.random.choice(len(data), len(data), replace=True)
    bootstrap_data = []
    for idx in indices:
        bootstrap_data.append((data[idx]['income'], data[idx]['educ'], data[idx]['age']))

    write_data_csv(bootstrap_data, 'bootstrap_data.csv')    

# calculate the linear regression coefficients for a given csv data file
def predict_coefficients(file):
    print ("Calculating coefficients for " + file)
    Xs, Ys = read_data_csv(file)
    Betas, residuals, ranks, s = np.linalg.lstsq(Xs, Ys)
    return Betas

# calculate the bias squared between two sets of betas
def calculate_bias_squared(betas_hat, betas):
    all_values = []
    for i in range(len(betas_hat)):
        all_values.append(betas_hat[i] - betas[i])
    return np.square(np.mean(all_values, axis=0))

# calculate the variance between two sets of betas
def calculate_variance(betas_hat, betas):
    all_values = []
    for i in range(len(betas_hat)):
        all_values.append(np.square(np.mean(betas_hat, axis=0) - betas_hat[i]))
    return np.mean(all_values, axis=0)

def main():
    num_trials = 100
    synthetic_betas, true_betas, bootstrap_betas = [], [], []

    # for each trial:
    # generate the histograms, data files, and predict coefficents
    for i in range(num_trials):
        print ("Running Trial " + str(i+1) + " of " + str(num_trials))

        generate_histograms()
        generate_synthetic_data()
        generate_true_data()
        generate_bootstrap_data()

        synthetic_betas.append(predict_coefficients('synthetic_data.csv'))
        true_betas.append(predict_coefficients('true_data.csv'))
        bootstrap_betas.append(predict_coefficients('bootstrap_data.csv'))

    # write out the coefficients to csvs
    write_coefficient_csv(synthetic_betas, 'synthetic_betas.csv')
    write_coefficient_csv(true_betas, 'true_betas.csv')
    write_coefficient_csv(bootstrap_betas, 'bootstrap_betas.csv')

    # calculate the bias squared and variance
    bootstrap_bias = calculate_bias_squared(bootstrap_betas, true_betas)
    bootstrap_variance = calculate_variance(bootstrap_betas, true_betas)
    synthetic_bias = calculate_bias_squared(synthetic_betas, true_betas)
    synthetic_variance = calculate_variance(synthetic_betas, true_betas)

    # write out the results to file
    results = [['bootstrap', bootstrap_bias[0], bootstrap_bias[1], bootstrap_bias[2], bootstrap_variance[0], bootstrap_variance[1], bootstrap_variance[2]],
               ['synthetic', synthetic_bias[0], synthetic_bias[1], synthetic_bias[2], synthetic_variance[0], synthetic_variance[1], synthetic_variance[2]]]
    write_mse_csv(results, 'problem_3_results.csv')

if __name__ == "__main__":
    main()