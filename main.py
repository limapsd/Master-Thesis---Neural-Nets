import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import tensorflow as tf
import datetime as dt


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



max_epochs = 1000
batch_size = 1000

es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=100, min_delta=1e-7, restore_best_weights=True)

params = {
    'rnn': {
        'model': None, 'function': SimpleRNN_, 'l1_reg': 0.0, 'H': 20, 
        'color': 'blue', 'label':'RNN'}, 
    'gru': {
        'model': None, 'function':GRU_,'l1_reg': 0.0, 'H': 10, 
        'color': 'orange', 'label': 'GRU'},
    'lstm': {
        'model': None, 'function': LSTM_,'l1_reg': 0.0, 'H': 10, 
        'color':'red', 'label': 'LSTM'}
}

n_units = [5, 10, 20]
# l1_reg = [0, 0.001, 0.01, 0.1]
l1_reg = [0, 1e-3, 1e-2, 1e-1, 1e-4, 1e-5]

# A dictionary containing a list of values to be iterated through
# for each parameter of the model included in the search
param_grid = {'n_units': n_units, 'l1_reg': l1_reg}

# In the kth split, TimeSeriesSplit returns first k folds 
# as training set and the (k+1)th fold as test set.

tscv = BlockingTimeSeriesSplit(n_splits = 5)

# A grid search is performed for each of the models, and the parameter set which
# performs best over all the cross-validation splits is saved in the `params` dictionary
for key in params.keys():
    print('Performing cross-validation. Model:', key)
    model = KerasRegressor(build_fn=params[key]['function'], epochs=max_epochs, 
                           batch_size=batch_size, verbose=0)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, 
                        cv=tscv, n_jobs=-1, verbose=2)
    grid_result = grid.fit(x_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params_ = grid_result.cv_results_['params']
    for mean, stdev, param_ in zip(means, stds, params_):
        print("%f (%f) with %r" % (mean, stdev, param_))
        
    params[key]['H'] = grid_result.best_params_['n_units']
    params[key]['l1_reg']= grid_result.best_params_['l1_reg']


for key in params.keys():
    tf.random.set_seed(0)
    print('Training', key, 'model')
    model = params[key]['function'](params[key]['H'], params[key]['l1_reg'])
    model.fit(x_train, y_train, epochs=max_epochs, 
              batch_size=batch_size, callbacks=[es], shuffle=False, verbose = 0)
    params[key]['model'] = model



for key in params.keys():
    model = params[key]['model']
    model.summary()
    
    params[key]['pred_train'] = model.predict(x_train, verbose=1)
    params[key]['MSE_train'] = mean_squared_error(y_train, params[key]['pred_train'])
    params[key]['MAE_train'] = mean_absolute_error(y_train, params[key]['pred_train'])
    
    params[key]['pred_test'] = model.predict(x_test, verbose=1) 
    params[key]['MSE_test'] = mean_squared_error(y_test, params[key]['pred_test'])
    params[key]['MAE_test'] = mean_absolute_error(y_test, params[key]['pred_test']) 
    params[key]['R2oo'] = (1 - (params[key]['MSE_test']/np.var(y_test)))    