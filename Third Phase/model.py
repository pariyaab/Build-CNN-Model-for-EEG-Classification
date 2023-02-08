import pickle

import keras
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd
from numpy.fft import fft
from scipy.signal import butter, lfilter
from scipy.stats import stats
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
# load dataset
X = pickle.load(open('new_x.pkl', 'rb'))
y = pickle.load(open('new_y.pkl', 'rb'))
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

print(X_train.shape)
print(X_test.shape)
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


class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();


plot_losses = PlotLosses()
features_x_train = np.empty(shape=(0, 15, 2))
for item in X_train:
    channel_17 = item[..., 0]
    feature_vector_channel_17 = features_extraction(channel_17)
    feature_vector_channel_17 = np.array(feature_vector_channel_17).reshape(1, 15)
    # channel_18
    channel_18 = item[..., 1]
    feature_vector_channel_18 = features_extraction(channel_18)
    feature_vector_channel_18 = np.array(feature_vector_channel_18).reshape(1, 15)
    normal_vector = np.stack([feature_vector_channel_17, feature_vector_channel_18], axis=-1)
    features_x_train = np.vstack([features_x_train, normal_vector])

features_x_test = np.empty(shape=(0, 15, 2))
for item in X_test:
    channel_17 = item[..., 0]
    feature_vector_channel_17 = features_extraction(channel_17)
    feature_vector_channel_17 = np.array(feature_vector_channel_17).reshape(1, 15)
    # channel_18
    channel_18 = item[..., 1]
    feature_vector_channel_18 = features_extraction(channel_18)
    feature_vector_channel_18 = np.array(feature_vector_channel_18).reshape(1, 15)
    normal_vector = np.stack([feature_vector_channel_17, feature_vector_channel_18], axis=-1)
    features_x_test = np.vstack([features_x_test, normal_vector])

# build the model
inputs = keras.layers.Input(shape=(1280, 2))
x = keras.layers.Conv1D(32, kernel_size=5, activation='relu')(inputs)
x = keras.layers.MaxPooling1D(2)(x)
x = keras.layers.Conv1D(64, kernel_size=5, activation='relu')(x)
x = keras.layers.MaxPooling1D(2)(x)
x = keras.layers.Conv1D(128, kernel_size=5, activation='relu')(x)
x = keras.layers.Flatten()(x)
# concatenate the extracted features
features_inputs = keras.layers.Input(shape=(15, 2))
features = keras.layers.Flatten()(features_inputs)
x = keras.layers.concatenate([x, features])
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)
# compile the model
model = keras.models.Model(inputs=[inputs, features_inputs], outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit the model
history = model.fit([X_train, features_x_train], y_train, epochs=10,
                    validation_data=([X_test, features_x_test], y_test), callbacks=[early_stopping, plot_losses])
y_pred = model.predict([X_test, features_x_test])
print("Recall:", recall_score(y_test, y_pred.round()))
print("Precision:", precision_score(y_test, y_pred.round()))

