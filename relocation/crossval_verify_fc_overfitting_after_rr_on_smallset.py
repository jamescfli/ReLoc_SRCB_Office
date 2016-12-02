__author__ = 'bsl'

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import GridSearchCV

import numpy as np
from utils.loss_acc_history_rtplot import LossRTPlot
from utils.timer import Timer

img_height = 448  # options: 448, 224, original 450*1920
img_width = img_height * 4

train_data = np.load(open('bottleneck_data/bottleneck_feature_vggrr_smallset_{}x{}.npy'
                          .format(img_height, img_width)))
# apply x100 label values
label_scaling_factor = 100
train_label = np.loadtxt('datasets/train_test_split_480x1920/test_label_x{}.csv'
                         .format(label_scaling_factor),
                         dtype='float32', delimiter=',')

assert train_data.shape[0] == train_label.shape[0], 'nb of data samples != nb of labels'
print('training data shape: {}'.format(train_data.shape))
print('training label shape: {}'.format(train_label.shape))


def create_model(dropout_ratio=0.5, weight_constraint=2, nb_hidden_node=256):
    model = Sequential()
    model.add(Dense(nb_hidden_node,
                    name='FC_Dense_1',  # for further weight loading
                    activation='relu',
                    W_constraint=maxnorm(weight_constraint),
                    input_shape=train_data.shape[1:]))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(nb_hidden_node,
                    name='FC_Dense_2',
                    activation='relu',
                    W_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(2, name='FC_Dense_3', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=[])
    return model

nb_epoch = 500
batch_size = 16

seed = 7
np.random.seed(seed)

# model = KerasRegressor(build_fn=create_model,
#                        nb_epoch=nb_epoch,
#                        batch_size=batch_size,
#                        verbose=1)   # valid if n_jobs=1, info 1 > 2 > 3 > 0
# search_grid_dropout_ratio = [0.2, 0.8]   # option 0.0~0.9
# search_grid_weight_constraint = [2]  # option 1~5
# search_grid_nb_hidden_node = [256, 512, 1024, 2048]   # limited by GPU mem
# # # measure unit time consumption
# # search_grid_dropout_ratio = [0.5]
# # search_grid_nb_hidden_node = [4096]
# param_grid = dict(dropout_ratio=search_grid_dropout_ratio,
#                   weight_constraint=search_grid_weight_constraint,
#                   nb_hidden_node = search_grid_nb_hidden_node)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)  # -1 has GPU mem issue
# with Timer('train one grid'):
#     grid_result = grid.fit(train_data, train_label)
#
# # results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))


# derive the following hyper parameter through cross validation
nb_hidden_node = 1024
dropout_ratio = 0.5
weight_constraint = 2
model = create_model(dropout_ratio=dropout_ratio,
                     weight_constraint=weight_constraint,
                     nb_hidden_node=nb_hidden_node)
lossRTplot = LossRTPlot()
model.fit(train_data, train_label,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          shuffle=True,
          verbose=1,
          callbacks=[lossRTplot])
model.save_weights('models/train_input{}_topfc{}_smallset_ls{}_{}epoch_DO{}_WC{}_reloc_model.h5'
                   .format(img_height,
                           nb_hidden_node,
                           label_scaling_factor,    # ls - label scalar
                           nb_epoch,
                           dropout_ratio,
                           weight_constraint))
