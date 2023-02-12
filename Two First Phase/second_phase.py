import os
import pickle
import random
import numpy
import pandas as pd
from scipy.spatial import distance
from sklearn import tree, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
from numpy.fft import fft
from scipy.signal import butter, lfilter
from scipy.stats import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids

seed = 57

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

x = pickle.load(open('x.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))

x_normal = np.concatenate((x[:300], x[400:]), axis=0)
x_seizure = x[300:400]
# print(x_normal.shape)
# print(x_seizure.shape)
sampling_freq = 173.6  # based on info from website

b, a = butter(3, [0.5, 40], btype='bandpass', fs=sampling_freq)

x_normal_filtered = np.array([lfilter(b, a, x_normal[ind, :]) for ind in range(x_normal.shape[0])])
x_seizure_filtered = np.array([lfilter(b, a, x_seizure[ind, :]) for ind in range(x_seizure.shape[0])])
# print(x_normal.shape)
# print(x_seizure.shape)

x_normal = x_normal_filtered
x_seizure = x_seizure_filtered

x = np.concatenate((x_normal, x_seizure))
y = np.concatenate((np.zeros((400, 1)), np.ones((100, 1))))

# print(x.shape)  # (500, 4097)
# print(y.shape)  # (500, 1)

FEATURES = ['MIN', 'MAX', 'MEAN', 'RMS', 'VAR', 'STD', 'POWER', 'PEAK', 'P2P', 'CREST FACTOR',
            'MAX_f', 'SUM_f', 'MEAN_f', 'VAR_f', 'PEAK_f']


def features_extraction(df):
    Min = []
    Max = []
    Mean = []
    Rms = []
    Var = []
    Std = []
    Power = []
    Peak = []
    Skew = []
    Kurtosis = []
    P2p = []
    CrestFactor = []
    FormFactor = []
    PulseIndicator = []
    Max_f = []
    Sum_f = []
    Mean_f = []
    Var_f = []
    Peak_f = []

    X = df
    ## TIME DOMAIN ##

    Min.append(np.min(X))
    Max.append(np.max(X))
    Mean.append(np.mean(X))
    Rms.append(np.sqrt(np.mean(X ** 2)))
    Var.append(np.var(X))
    Std.append(np.std(X))
    Power.append(np.mean(X ** 2))
    Peak.append(np.max(np.abs(X)))
    P2p.append(np.ptp(X))
    CrestFactor.append(np.max(np.abs(X)) / np.sqrt(np.mean(X ** 2)))
    Skew.append(stats.skew(X))
    Kurtosis.append(stats.kurtosis(X))
    FormFactor.append(np.sqrt(np.mean(X ** 2)) / np.mean(X))
    PulseIndicator.append(np.max(np.abs(X)) / np.mean(X))
    ## FREQ DOMAIN ##
    ft = fft(X)
    S = np.abs(ft ** 2) / len(df)
    Max_f.append(np.max(S))
    Sum_f.append(np.sum(S))
    Mean_f.append(np.mean(S))
    Var_f.append(np.var(S))

    Peak_f.append(np.max(np.abs(S)))
    # Create dataframe from features
    df_features = pd.DataFrame(index=[FEATURES],
                               data=[Min, Max, Mean, Rms, Var, Std, Power, Peak, P2p, CrestFactor,
                                     Max_f, Sum_f, Mean_f, Var_f, Peak_f])
    return df_features[0].to_list()


def train_test_splitter(feature_x_):
    kfold = KFold(n_splits=5, random_state=seed, shuffle=True)
    for train_index, test_index in kfold.split(feature_x_):
        X_train, X_test = feature_x_[train_index], feature_x_[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test


def select_separate_features(features_array, index):
    final_array = []
    final_array_2 = []
    list_ = [0]
    for item in features_array:
        list_[0] = item[index]
        final_array.append(list_)
        final_array_2.append(item[index])
    return final_array, final_array_2


def calculate_correlation(a, b):
    return numpy.corrcoef(a, b)[0][1]


def combination_of_two_number(a, b):
    return (2 * abs(a) * abs(b)) / (abs(a) + abs(b))


def find_accuracy_per_feature(feature_1_D):
    feature_1_D = numpy.array(feature_1_D)
    X_train, X_test, y_train, y_test = train_test_splitter(feature_1_D)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy


def filter_feature_vector(train, given_index):
    new_feature = []
    for item in train:
        temp_array = []
        for i in given_index:
            temp_array.append(item[i])
        new_feature.append(temp_array)
    return new_feature


def cluster_indices_comp(cluster_num, labels_array):  # list comprehension
    # print(np.array([i for i, x in enumerate(labels_array) if x == cluster_num]).shape)
    return [i for i, x in enumerate(labels_array) if x == cluster_num]


def build_classification_per_cluster(x_train, y_tran, indexes):
    x_final = []
    y_final = []
    for i in indexes:
        x_final.append(x_train[i])
        y_final.append(y_tran[i])
    clf = RandomForestClassifier(n_estimators=200)
    x_final = numpy.array(x_final)
    y_final = numpy.array(y_final)
    clf.fit(x_final, y_final)
    return clf


def find_nearest_cluster(test_data, cluster_centers):
    distance_array = []
    for i in range(0, len(cluster_centers)):
        distance_array.append(distance.euclidean(test_data, cluster_centers[i]))
    return distance_array.index(np.min(distance_array))


def calculate_accuracy(x_test, y_test, cluster_centers):
    x_test_labels = []
    for i in range(0, len(x_test)):
        nearest_cluster = find_nearest_cluster(x_test[i], cluster_centers)
        test_data = []
        test_data.append(x_test[i])
        # print(np.array(test_data),np.array(test_data).shape)
        # print("i + nearest cluster:",i , nearest_cluster)
        if nearest_cluster == 0:
            x_test_labels.append(first_cluster_classification.predict(test_data))
        elif nearest_cluster == 1:
            x_test_labels.append(second_cluster_classification.predict(test_data))
        elif nearest_cluster == 2:
            x_test_labels.append(third_cluster_classification.predict(test_data))
    print(accuracy_score(x_test_labels, y_test))


features_x = []
normalizedData = (x - np.min(x)) / (np.max(x) - np.min(x))

for item in normalizedData:
    feature_vector = features_extraction(item)
    features_x.append(feature_vector)


def find_important_features(all_features):
    previous_features = []
    saved_index = []
    for i in range(0, 15):
        feature_1_D, feature_1_D_flat = select_separate_features(all_features, i)
        previous_features.append(feature_1_D_flat)
        accuracy = find_accuracy_per_feature(feature_1_D)
        correlation_numbers = []
        for j in range(0, i):
            correlation_numbers.append(calculate_correlation(feature_1_D_flat, previous_features[j]))
        if i > 0:
            correlation = np.max(correlation_numbers)
            f1_measure = combination_of_two_number(accuracy, correlation)
            if f1_measure >= 0.88:
                saved_index.append(i)
    return saved_index


saved_index = find_important_features(features_x)
features_x = numpy.array(features_x)
kfold = KFold(n_splits=5, random_state=seed, shuffle=True)
for train_index, test_index in kfold.split(features_x):
    X_train, X_test = features_x[train_index], features_x[test_index]
    y_train, y_test = y[train_index], y[test_index]

new_feature_x_train = numpy.array(filter_feature_vector(X_train, saved_index))
# data = list(zip(new_feature_x_train, y_train))
# print(np.unique(list(map(len, y_train))))
k_means = KMeans(n_clusters=3)
k_means.fit(new_feature_x_train)
# kmedoids = KMedoids(n_clusters=3, random_state=0).fit(new_feature_x_train)

first_cluster_classification = build_classification_per_cluster(new_feature_x_train, y_train,
                                                                cluster_indices_comp(0, k_means.labels_))
second_cluster_classification = build_classification_per_cluster(new_feature_x_train, y_train,
                                                                 cluster_indices_comp(1, k_means.labels_))

third_cluster_classification = build_classification_per_cluster(new_feature_x_train, y_train,
                                                                cluster_indices_comp(2, k_means.labels_))

new_feature_x_test = np.array(filter_feature_vector(X_test, given_index=saved_index))
calculate_accuracy(new_feature_x_test, y_test, k_means.cluster_centers_)
