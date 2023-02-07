import pickle
import numpy as np
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold

# load dataset
X = pickle.load(open('x_1.pkl', 'rb'))
y = pickle.load(open('y_1.pkl', 'rb'))
seed = 57
kfold = KFold(n_splits=5, random_state=seed, shuffle=True)
for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

print(X_train.shape)
print(X_test.shape)