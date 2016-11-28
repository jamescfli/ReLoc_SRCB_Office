__author__ = 'bsl'

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import GridSearchCV

import numpy as np
from utils.timer import Timer

train_data = np.load(open('bottleneck_data/bottleneck_feature_vggrr_smallset.npy'))
train_label = np.loadtxt('datasets/train_test_split_480x1920/test_label.csv', dtype='float32', delimiter=',')

assert train_data.shape[0] == train_label.shape[0], 'nb of data samples != nb of labels'
print('training data shape: {}'.format(train_data.shape))
print('training label shape: {}'.format(train_label.shape))

def create_model(dropout_ratio=0.0, weight_constraint=0, nb_hidden_node=4096):
    model = Sequential()
    model.add(Dense(nb_hidden_node,
                    activation='relu',
                    W_constraint=maxnorm(weight_constraint),
                    input_shape=train_data.shape[1:]))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(nb_hidden_node,
                    activation='relu',
                    W_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(2, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=[])
    return model

nb_epoch = 100
batch_size = 16

seed = 7
np.random.seed(seed)

model = KerasRegressor(build_fn=create_model,
                       nb_epoch=nb_epoch,
                       batch_size=batch_size,
                       verbose=2)   # valid if n_jobs=1, more positive more detail
search_grid_dropout_ratio = [0.2, 0.5, 0.8]
search_grid_weight_constraint = [1, 3, 5]
search_grid_nb_hidden_node = [256, 512, 1024, 2048, 4096]   # limited by GPU mem
# # measure unit time consumption
# search_grid_dropout_ratio = [0.5]
# search_grid_nb_hidden_node = [4096]
param_grid = dict(dropout_ratio=search_grid_dropout_ratio,
                  nb_hidden_node = search_grid_nb_hidden_node)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
with Timer('train one grid'):   # 50% DO + fc4096: 1177sec => 36 test in ~<11.77 hours
    grid_result = grid.fit(train_data, train_label)

# results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# # time estimation through single model test
# model = create_model(dropout_ratio=0.9, nb_hidden_node=4096)   # options are 256, 512, 1024, 2048, 4096
# model.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=1)
