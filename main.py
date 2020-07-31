import re
import scipy.stats as s
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import random
import csv


def extract(file_name):
    with open(file_name + '.csv') as f:
        data = list(csv.reader(f))
    headings = data[0]
    data = np.array(data[1:])
    d = {headings[i] : list(data[:, i]) for i in range(len(headings))}
    return d


def proper_data(data, category):
    data = data.lower()
    if category == "Size":
        if "m" in data:
            return float(data.strip("m"))
        if "k" in data:
            return float(data.strip("k")) / 1024
    if category == "Installs":
        return float(data.strip("+").replace(",", ""))
    if category == "Price":
        return float(data.strip("$"))
    else:
        return float(data)


def z(data):
    return s.zscore(data)


def clean(a, categorical, numerical):
    for category in categorical:
        for j in range(len(a[category])):
            if a[category][j] == "" or a[category][j] == "NaN":
                if j == 0:
                    a[category][j].replace("", a[category][j + 1])
                else:
                    a[category][j] = a[category][j - 1]

    for category in numerical:
        mean = count = 0
        for j in range(len(a[category])):
            if not re.search("\d", a[category][j]):
                a[category][j] = "Nan"
            else:
                a[category][j] = proper_data(a[category][j], category)
                mean += a[category][j]
                count += 1
        mean /= count
        for j in range(len(a[category])):
            if a[category][j] == "Nan":
                a[category][j] = mean


def normalize(x):
    obj = col.Normalize()
    norm = obj.__call__(x)
    return norm

def bar(data, xaxis="", yaxis=""):
    keys = []
    values = []
    labels = []
    for i in list(set(data)):
        keys.append(i)
        values.append(list(data).count(i) / len(data) * 100)
        labels.append(i)
    lis = list(zip(keys, values, labels))
    lis = sorted(lis, key=lambda x:x[1])
    # plt.figure(figsize=[10, 10])
    plt.xticks(range(0, len(keys)), np.array(lis)[:, 2], rotation=85)
    for i in range(len(lis)):
        print(i, lis[i][0], lis[i][1])
        plt.bar(i, lis[i][1], color=[0] + [i / len(keys)] * 2, align="center")
        #plt.bar(i, lis[i][1], color=[random.random() for i in range(3)], align="center")
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.show()


def scatterplot(x, y, xlabel="", ylabel="", norm=0):
    x1 = x.copy()
    y1 = y.copy()
    if norm:
        x1 = normalize(x1)
        y1 = normalize(y1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.plot(x1, y1, 'ro')
    plt.show()


def bar2(data, labels):
    lis = list(zip(labels, data))
    print(lis)
    lis = sorted(lis, key=lambda x: x[1])
    print(lis)
    for i in range(len(lis)):
        print(i, lis[i][0], lis[i][1])
        plt.bar(i, lis[i][1], color=[0] + [i / len(labels)] * 2, align="center")
    plt.xticks(range(len(lis)), np.array(lis)[:, 0])
    plt.xlabel("Factors")
    plt.ylabel("Correlation of factors vs Installs")
    plt.show()


def correlation(x, y):
    return np.dot(np.array(z(x)), np.array(z(y))) / (len(x))


def left_tail(sample_mean, sample_std, null_mean, alpha=0.05):          # left tailed test
    z = (sample_mean - null_mean) / (sample_std)
    p = s.norm.cdf(z)
    print("z:", z)
    print("p:", p)
    if p < alpha:
        print("Reject null hypothesis")
    else:
        print("Both hypotheses are plausible")


def right_tail(sample_mean, sample_std, null_mean, alpha=0.05):          # right tailed test
    z = (sample_mean - null_mean) / (sample_std)
    p = 1 - s.norm.cdf(z)
    print("z:", z)
    print("p:", p)
    if p < alpha:
        print("Reject null hypothesis")
    else:
        print("Both hypotheses are plausible")


def two_sided(sample_mean, sample_std, null_mean, alpha=0.05):          # right tailed test
    z = (sample_mean - null_mean) / (sample_std)
    p =  2 * s.norm.cdf(z)
    print("z:", z)
    print("p:", p)
    if p < alpha:
        print("Reject null hypothesis")
    else:
        print("Both the hypotheses are plausible")


def hypothesis_test(data, samples, null_mean, null_sign):
    sample = []
    while samples != 0:
        l = []
        for i in range(100):
            r = random.randint(0, len(data)-1)
            if r not in l:
                l.append(r)
        l = [data[i] for i in l]
        sample.append(np.mean(l))
        samples -= 1
    mean = np.mean(sample)
    std = np.std(sample)
    print("Sample mean:", mean)
    print("Sample standard deviation:", std)
    print("Number of samples:", len(sample))
    print("Size of each sample:", 100)

    if(null_sign == ">"):
        left_tail(mean, std, null_mean)
    elif(null_sign == "<"):
        right_tail(mean, std, null_mean)
    elif(null_sign == "="):
        two_sided(mean, std, null_mean)


categorical = ["Category", "Type", "Content Rating", "Last Updated", "Current Ver", "Android Ver", "Genres"]
numerical = ["Installs", "Rating", "Reviews", "Size", "Price"]

a = extract("googleplaystore")
clean(a, categorical, numerical)
'''ans = two_sided(1000.6, 2, 60, 1000)
print(ans)'''
# hypothesis_test(a["Size"], 100, 21, ">")
# plt.pie([a["Category"].count(i) for i in set(a["Category"])], labels=set(a["Category"]))
# corr = [abs(correlation(a["Installs"], a[i])) for i in numerical[1:]]
# bar2(corr, numerical[1:])
'''plt.pie(corr, labels=numerical[1:])
plt.show()'''
'''plt.hist(a["Size"])
plt.xticks(range(0, 100, 10))
plt.xlabel("Size")
plt.ylabel("Number of apps")
plt.show()
'''
'''for i in numerical:
    print(i)
    plt.ylabel(i)
    plt.boxplot(a[i])
    plt.show()
'''

#bar(a["Category"])
#bar(a["Android Ver"], "Android Version", "Number of apps")

#plt.show()
# two_sided(871, 21, 50, 880)
#left_tail(a['Size'], , 64, 250000)
hypothesis_test(a["Size"], 100, 20, ">")
