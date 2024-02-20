from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.initializers import glorot_normal, glorot_uniform, orthogonal
from keras.optimizers import Adam
from keras import models
from keras.layers import Dense, Layer, Dropout, LSTM, GRU, SimpleRNN, RNN
from keras.models import Sequential
from keras.models import load_model
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend




def SimpleRNN_(n_units=10, l1_reg=0, seed=0):
    model = Sequential()
    model.add(SimpleRNN(n_units, activation='tanh', kernel_initializer=glorot_uniform(seed),
                        bias_initializer=glorot_uniform(seed), recurrent_initializer=orthogonal(seed),
                        kernel_regularizer=l1(l1_reg), input_shape=(x_train.shape[1], x_train.shape[-1]),
                        unroll=True, stateful=False))
    model.add(Dense(1, kernel_initializer=glorot_uniform(seed), bias_initializer=glorot_uniform(seed),
                    kernel_regularizer=l1(l1_reg)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def GRU_(n_units=10, l1_reg=0, seed=0):
    model = Sequential()
    model.add(GRU(n_units, activation='tanh', kernel_initializer=glorot_uniform(seed),
                  bias_initializer=glorot_uniform(seed),
                  recurrent_initializer=orthogonal(seed),
                  kernel_regularizer=l1(l1_reg), input_shape=(x_train.shape[1], x_train.shape[-1]),
                  unroll=True, stateful=False))
    model.add(Dense(1, kernel_initializer=glorot_uniform(seed),
                    bias_initializer=glorot_uniform(seed), kernel_regularizer=l1(l1_reg)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def LSTM_(n_units=10, l1_reg=0, seed=0):
    model = Sequential()
    model.add(LSTM(n_units, activation='tanh', kernel_initializer=glorot_uniform(seed),
                   bias_initializer=glorot_uniform(seed),
                   recurrent_initializer=orthogonal(seed),
                   kernel_regularizer=l1(l1_reg), input_shape=(x_train.shape[1], x_train.shape[-1]),
                   unroll=True, stateful=False))
    model.add(Dense(1, kernel_initializer=glorot_uniform(seed),
                    bias_initializer=glorot_uniform(seed), kernel_regularizer=l1(l1_reg)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]
