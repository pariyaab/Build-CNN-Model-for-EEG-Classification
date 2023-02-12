import numpy
import pyedflib
import numpy as np
import pickle

# augmentation methods
from sklearn.model_selection import KFold


def add_noise(data, mean=0, std=0.1):
    noise = np.random.normal(mean, std, data.shape)
    augmented_data = data + noise
    return augmented_data


def scale(data, factor=0.9):
    augmented_data = data * factor
    return augmented_data


def time_shift(data, shift=10):
    augmented_data = np.roll(data, shift, axis=1)
    return augmented_data


def amplitude_scale(data, factor=0.5):
    augmented_data = data * factor
    return augmented_data


def augment_vector(vector, X, y, label):

    augmented_data = add_noise(vector)
    reshaped = augmented_data.reshape(1, 1280, 2)
    X = np.vstack([X, reshaped])
    y = np.vstack([y, label])

    augmented_data = scale(vector)
    reshaped = augmented_data.reshape(1, 1280, 2)
    X = np.vstack([X, reshaped])
    y = np.vstack([y, label])

    augmented_data = time_shift(vector)
    reshaped = augmented_data.reshape(1, 1280, 2)
    X = np.vstack([X, reshaped])
    y = np.vstack([y, label])

    augmented_data = amplitude_scale(vector)
    reshaped = augmented_data.reshape(1, 1280, 2)
    X = np.vstack([X, reshaped])
    y = np.vstack([y, label])
    return X, y


# load dataset
X = pickle.load(open('new_2_x.pkl', 'rb'))
y = pickle.load(open('new_2_y.pkl', 'rb'))
# generate a random permutation index
permutation = np.random.permutation(X.shape[0])

# apply the permutation to both x and y
X = X[permutation, :, :]
y = y[permutation, :]
seed = 57
kfold = KFold(n_splits=5, random_state=seed, shuffle=True)
for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

print(X_train.shape , X_test.shape)

for i in range(0, len(X_train)):
    X_train, y_train = augment_vector(X_train[i], X_train, y_train, y_train[i])

for i in range(0, len(X_test)):
    X_test, y_test = augment_vector(X_test[i], X_test, y_test, y_test[i])

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
pickle.dump(X_train, open('X_train.pkl', 'wb'))
pickle.dump(y_train, open('y_train.pkl', 'wb'))
pickle.dump(X_test, open('X_test.pkl', 'wb'))
pickle.dump(y_test, open('y_test.pkl', 'wb'))
