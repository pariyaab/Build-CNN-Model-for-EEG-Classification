import os
import pickle
import random

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import numpy
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from numpy.fft import fft
from scipy.signal import butter, lfilter
from scipy.stats import stats
from sklearn import svm

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


features_x = []
normalizedData = (x - np.min(x)) / (np.max(x) - np.min(x))

for item in normalizedData:
    feature_vector = features_extraction(item)
    features_x.append(feature_vector)

features_x = numpy.array(features_x)
kfold = KFold(n_splits=5, random_state=seed, shuffle=True)
for train_index, test_index in kfold.split(features_x):
    X_train, X_test = features_x[train_index], features_x[test_index]
    y_train, y_test = y[train_index], y[test_index]

print(X_test.shape)
# SVM
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred_svm = clf.predict(X_test)
print(type(y_pred_svm))
# print("Accuracy SVM:", accuracy_score(y_test, y_pred_svm))
# print("Precision SVM:", precision_score(y_test, y_pred_svm))
# print("Recall SVM:", recall_score(y_test, y_pred_svm))
# confusion_matrix = metrics.confusion_matrix(y_test, y_pred_svm)
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0., 1.])
# cm_display.plot()
# plt.show()

# Random Forest
# clf = RandomForestClassifier(n_estimators=200)
# clf.fit(X_train, y_train)
# y_pred_random_forest = clf.predict(X_test)
# print("Accuracy Random Forest:", accuracy_score(y_test, y_pred_random_forest))
# print("Precision Random Forest:", precision_score(y_test, y_pred_random_forest))
# print("Recall Random Forest:", recall_score(y_test, y_pred_random_forest))
# confusion_matrix = metrics.confusion_matrix(y_test, y_pred_random_forest)
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0., 1.])
# cm_display.plot()
# plt.show()

# y_score1 =clf.predict_proba(X_test)[:, 1]
# false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_score1)
# plt.title('Receiver Operating Characteristic - Random Forest')
# plt.plot(false_positive_rate1, true_positive_rate1)
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
# KNN
# knn_model = KNeighborsClassifier(n_neighbors=10)
# knn_model.fit(X_train, y_train)
# y_pred_knn = knn_model.predict(X_test)
# print("Accuracy KNN:", accuracy_score(y_test, y_pred_knn))
# print("Precision KNN:", precision_score(y_test, y_pred_knn))
# print("Recall KNN:", recall_score(y_test, y_pred_knn))
# confusion_matrix = metrics.confusion_matrix(y_test, y_pred_knn)
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0., 1.])
# cm_display.plot()
# plt.show()
# y_score1 = knn_model.predict_proba(X_test)[:, 1]
# false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_score1)
# plt.title('Receiver Operating Characteristic - Random Forest')
# plt.plot(false_positive_rate1, true_positive_rate1)
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
