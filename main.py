import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import sys


def make_word_list_from_single_document(file):  # Takes file object and returns list of words from document
    s = file.read() # read file
    s = s.strip().split("\n") # split it about '\n'
    final = []
    for each in s:
        temp = each.strip().split(" ")
        final.append(temp)

    for each in final: # remove spaces
        while '' in each:
            each.remove('')

    veryFinal = []

    for eachlist in final: # loop over each list and make final word list
        if len(eachlist) == 0:
            continue
        for word in eachlist:
            word = word.strip()

            if "/" in word:
                tosplit = word.split("/")
                for w in tosplit:
                    eachlist.append(w)
                continue

            if word.isnumeric():
                continue

            if len(word) > 1:
                if word[0].isalpha() == 0 or word[-1].isalpha() == 0:
                    t = ""
                    for ch in word:
                        if ch.isalpha():
                            t += ch
                    word = t

            if len(word) > 1:
                veryFinal.append(word.lower())

    return veryFinal


def give_stopwords_list(): # returns stopwords taken from file stopwords.txt
    stopwords_file = open("stopwords.txt", 'r+')
    stopwords = stopwords_file.read()
    stopwords = stopwords.strip().split('\n')
    stopwords_file.close()
    return stopwords


def build_vocabulary(): # building vocabulary from training data
    stopwords = give_stopwords_list()
    vocabulary = {}
    path = r"Training_data"  # selected subset for training purpose
    i = 1
    for class_name_with_path in glob.glob(os.path.join(path, '*')): # loop over each class folder
        for filename in glob.glob(os.path.join(class_name_with_path, '*')): # loop over each file of current folder
            file = open(filename, 'r+')
            print("(Vocab. building) working on file no.", i)
            i += 1
            wordlist = make_word_list_from_single_document(file)
            for word in wordlist:
                if word in stopwords:
                    continue
                if vocabulary.get(word) is None:  # checking key is present or not
                    vocabulary[word] = 1
                else:
                    vocabulary[word] += 1

            file.close()
    return vocabulary


def write_k_features(K): # writes first 'k' features in file feature_set.txt
    vocabulary = build_vocabulary()
    sorted_vocabulary = sorted(zip(vocabulary.values(), vocabulary.keys()),
                               reverse=True)  # sorting vocabulary in descending order by value
    file = open("vocabulary.txt", 'w')
    for k, v in sorted_vocabulary: # writing sorted vocabulary
        file.write(str(k) + " " + str(v) + "\n")
    file.close()
    file = open("feature_set.txt", 'w')
    i = 1
    for key, value in sorted_vocabulary: # writes first k words as features in feature_set.txt
        file.write(value + '\n')
        i += 1
        if i == K + 1:
            break
    file.close()


def get_feature_list(): # return list of feature_list read from feature_set.txt
    file = open("feature_set.txt", 'r')
    return file.read().strip().split('\n')


def build_dataset(path, m, k, write_file_name): # build .csv file for train and test
    file = open("class_list.txt", 'r+') # read class names from already stored class_list.txt
    class_list = file.read().strip().split('\n')
    file.close()

    feature_list = get_feature_list() # getting feature_list
    X = np.zeros(shape=(m, k))
    Y = np.zeros(m, int)
    i = 0
    for class_name_with_path in glob.glob(os.path.join(path, '*')):  # loop over each class folder
        class_name = class_name_with_path.split('\\')[-1]
        for filename in glob.glob(os.path.join(class_name_with_path, '*')): # loop over each file of current folder
            file = open(filename, 'r+')
            print('(' + write_file_name + ') building: Working on file no.', i + 1)
            wordlist = make_word_list_from_single_document(file)
            for word in wordlist:
                if word in feature_list:
                    X[i, feature_list.index(word)] += 1
            file.close()
            Y[i] = class_list.index(class_name)
            i += 1

    df = pd.DataFrame(X)
    df["Y"] = Y
    data = df.values
    np.savetxt(write_file_name, data, delimiter=",")


def build_train_and_test_csv(k): # build train test .csv files using first k words found earlier as features
    training_file_count = 18997
    testing_file_count = 997
    write_k_features(k)
    training_data_path = r"Training_data"
    testing_data_path = r"Testing_data"

    build_dataset(training_data_path, training_file_count, k, "Train.csv")
    build_dataset(testing_data_path, testing_file_count, k, "Test.csv")


def fit(X_train, Y_train): # fit function for self-implemented naive_bayes
    Count = {}
    class_values = set(Y_train)
    Count["total_data"] = len(Y_train)
    for current_class in class_values:
        Count[current_class] = {}
        current_class_rows = (Y_train == current_class)
        X_train_current = X_train[current_class_rows]
        Y_train_current = Y_train[current_class_rows]
        Count[current_class]["total_count"] = len(Y_train_current)
        for each_feature in range(X_train.shape[1]):
            Count[current_class][each_feature] = X_train_current[:, each_feature].sum()
    print("FITTED")
    return Count


def get_current_class_all_word_freq(dict): # returns word frequency of current class from dictionary
    c = 0
    for k, v in zip(dict.keys(), dict.values()):
        if k == "total_count":
            continue
        c += v
    return c


def probability(Count, x, current_class): # finds probability of each class
    output = np.log(Count[current_class]["total_count"]) - np.log(Count["total_data"])
    num_features = len(Count[current_class].keys()) - 1
    for j in range(num_features):
        xj = x[j]
        count_current_class_with_current_feature = Count[current_class][j] + 1
        count_current_class_data = get_current_class_all_word_freq(Count[current_class]) + 2000
        current_prob = np.log(count_current_class_with_current_feature) - np.log(count_current_class_data)
        if xj != 0:
            current_prob += np.log(xj)
        output += current_prob
    return output


def predictSinglePoint(Count, x): # predicts one particular data point
    first_run = True
    best_prob = None
    best_class = None
    classes = Count.keys()
    for current_class in classes:
        if current_class == "total_data":
            continue
        current_class_prob = probability(Count, x, current_class)
        if first_run or current_class_prob > best_prob:
            best_class = current_class
            best_prob = current_class_prob
        first_run = False
    return best_class


def predict(Count, x_test): # predict classes for all data-points
    y_pred = []
    i = 1
    for x in x_test:
        x_predicted_class = predictSinglePoint(Count, x)
        y_pred.append(x_predicted_class)
        i += 1
    return y_pred


def run_self_multinomial_naive_bayes(): # runs self implemented code
    k = 2000
    build_train_and_test_csv(k)
    training_file_count = 18997
    testing_file_count = 997
    df = pd.read_csv("Train.csv", header=None)
    data = df.values
    print(data.shape)
    X_train = data[:, 0:len(data[0]) - 1]
    Y_train = data[:, len(data[0]) - 1]

    df = pd.read_csv("Test.csv", header=None)
    data = df.values
    print(data.shape)
    X_test = data[:, 0:len(data[0]) - 1]
    Y_test = data[:, len(data[0]) - 1]

    Count = fit(X_train, Y_train)
    Y_pred = predict(Count, X_test)

    print(accuracy_score(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))


def run_inbuilt_multinomial_naive_bayes(): # runs using in-built (sklearn) naive_bayes
    k = 2000
    build_train_and_test_csv(k)

    df = pd.read_csv("Train.csv", header=None)
    data = df.values
    print(data.shape)
    X_train = data[:, 0:len(data[0]) - 1]
    Y_train = data[:, len(data[0]) - 1]

    df = pd.read_csv("Test.csv", header=None)
    data = df.values
    print(data.shape)
    X_test = data[:, 0:len(data[0]) - 1]
    Y_test = data[:, len(data[0]) - 1]

    clf = MultinomialNB()
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    print(accuracy_score(Y_test, Y_pred))
    print(classification_report(Y_test, Y_pred))


run_inbuilt_multinomial_naive_bayes()

run_self_multinomial_naive_bayes()
