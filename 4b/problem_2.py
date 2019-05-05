import numpy as np
import csv
import matplotlib.pyplot as plt

# read in csv data
def read_data(file):
    with open(file, 'rb') as f:
        reader = csv.reader(f)
        next(f)
        X, Y = [], []
        for row in reader:
            try:
                X.append(int(row[4]))
                Y.append(int(row[9]))
            except:
                pass
        return np.array(X), np.array(Y)

# read in non-private coefficients
def read_non_private_coefs():
    with open('non_private_results.csv', 'rb') as f:
        reader = csv.reader(f)
        next(f)
        betas = []
        for row in reader:
            betas.append(float(row[1]))
        return np.array(betas)

# read in private coefficients
def read_private_coefs():
    with open('local_results.csv', 'rb') as f:
        reader = csv.reader(f)
        next(f)
        results = {}
        for row in reader:
            try:
                epsilon = float(row[1])
                results[epsilon] = np.vstack((results.get(epsilon, np.array([]).reshape(0, 2)), [float(row[2]), float(row[3])]))
            except:
                pass
        return results

# calculate classification error for given coefficient pair
def classification_error(X, Y, coef):
    num_correct = 0.
    for i in range(X.shape[0]):
        pred = coef[0] + coef[1] * X[i]
        num_correct += round(pred) == Y[i]
    return 1. - (num_correct / float(X.shape[0]))

# calculate average classification error for list of coefficient pairs
def calculate_avg_class_error(X, Y, coefs):
    avg_errors = {}
    for epsilon in sorted(coefs.keys()):
        errors = np.array([])
        for coef in coefs[epsilon]:
            errors = np.concatenate((errors, [classification_error(X, Y, coef)]))
        avg_errors[epsilon] = np.mean(errors, axis=0)
    return avg_errors

# plot the average classification error
def plot_avg_class_error(errors):
    X, Y = [], []
    for epsilon, error in errors.iteritems():
        X.append(epsilon)
        Y.append(error)

    plt.xscale('log')
    plt.scatter(X, Y)
    plt.xlabel(r'$\epsilon$')
    plt.ylabel('Classification Error (averaged over 10 trials)')
    plt.title('Classification Error of Local DP SGD Model')
    plt.xlim(min(X)/1.25, max(X)*1.25)
    plt.savefig('classification_error.png')
    plt.clf()

# calculate the rmse for two pairs of coefficients
def calculate_rmse(non_private_coef, private_coefs):
    avg_errors = {}
    for epsilon in sorted(private_coefs.keys()):
        errors = np.array([]).reshape(0, 2)
        for private_coef in private_coefs[epsilon]:
            errors = np.vstack((errors, [np.square(private_coef[0] - non_private_coef[0]), np.square(private_coef[1] - non_private_coef[1])]))
        avg_errors[epsilon] = np.sqrt(np.mean(errors, axis=0))
    return avg_errors

# plot the rmse
def plot_rmse(errors):
    X, Y_0, Y_1 = [], [], []
    for epsilon, error in errors.iteritems():
        X.append(epsilon)
        Y_0.append(error[0])
        Y_1.append(error[1])

    plt.xscale('log')
    plt.scatter(X, Y_0, label=r'$\beta_0$', edgecolors='blue', facecolors='none', marker='o')
    plt.scatter(X, Y_1, label=r'$\beta_1$', facecolors='red', marker='x')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel('RMSE (over 10 trials)')
    plt.title('Root Mean Squared Error of Local DP Coefficients')
    plt.xlim(min(X)/1.25, max(X)*1.25)
    plt.legend(loc='upper right')
    plt.savefig('rmse.png')
    plt.clf()

def main():
    X, Y = read_data('MaPUMS5full.csv')

    non_private_coef = read_non_private_coefs()
    private_coefs = read_private_coefs()

    error = calculate_avg_class_error(X, Y, private_coefs)
    plot_avg_class_error(error)

    rmse = calculate_rmse(non_private_coef, private_coefs)
    plot_rmse(rmse)

if __name__ == "__main__":
    main()