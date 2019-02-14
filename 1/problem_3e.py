import csv
import numpy as np

def read_csv(file):
    with open(file, 'rU') as csv_file:
        reader = csv.DictReader(csv_file)
        data = []
        for row in reader:
            if row['englishability'] == 'NA':
                row['englishability'] = 2
            if row['income'][-4:] == 'e+05':
                row['income'] = 100000 * int(row['income'][0])
            for key in row.keys():
                row[key] = int(row[key])
            del row['']
            data.append(row)
        return data

def get_population_means(data):
    means = {}
    for row in data:
        for key, value in row.iteritems():
            means[key] = means.get(key, 0.) + float(value)
    for key, value in means.iteritems():
        means[key] = value / float(len(data))
    return means

def main():
    data = read_csv('FultonPUMS5full.csv')
    means = get_population_means(data)
    with open('FultonPUMS5full_means.csv', 'wb') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=means.keys())
        writer.writeheader()
        writer.writerow(means)

if __name__ == "__main__":
    main()