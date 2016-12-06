__author__ = 'bsl'

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

import numpy as np
from utils.loss_acc_history_rtplot import LossAccRTPlot
from utils.timer import Timer

img_height = 224
img_width = 224

# load small dataset
train_data = np.load(open('bottleneck_data/bottleneck_feature_vgg_places_smallset.npy'
                          .format(img_height, img_width)))
train_label_index = np.array([0]*3000+[1]*3000)
train_label = np_utils.to_categorical(train_label_index, nb_classes=2)

assert train_data.shape[0] == train_label.shape[0], 'nb of data samples != nb of labels'
print('training data shape: {}'.format(train_data.shape))
print('training label shape: {}'.format(train_label.shape))

# # load large data set
# train_data = np.load(open('bottleneck_data/bottleneck_feature_vgg_places_largeset.npy'))
# train_label_index = np.array([0]*18344+[1]*29055)
# train_label = np_utils.to_categorical(train_label_index, nb_classes=2)
#
# assert train_data.shape[0] == train_label.shape[0], 'nb of data samples != nb of labels'
# print('training data shape: {}'.format(train_data.shape))
# print('training label shape: {}'.format(train_label.shape))


def create_model(dropout_ratio=0.5, weight_constraint=2, nb_hidden_node=256):
    model = Sequential()
    model.add(Dense(nb_hidden_node,
                    name='FC_Dense_1',  # for further weight loading
                    activation='relu',
                    W_constraint=maxnorm(weight_constraint),
                    input_shape=train_data.shape[1:]))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(2, name='FC_Dense_2', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

nb_epoch = 100
batch_size = 32

seed = 7
np.random.seed(seed)

model = KerasClassifier(build_fn=create_model,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        verbose=1)   # valid if n_jobs=1, info 1 > 2 > 3 > 0
search_grid_dropout_ratio = [0.2, 0.5, 0.8]   # option 0.0~0.9
search_grid_weight_constraint = [2]  # option 1~5
search_grid_nb_hidden_node = [256, 512, 1024, 2048]   # limited by GPU mem
param_grid = dict(dropout_ratio=search_grid_dropout_ratio,
                  weight_constraint=search_grid_weight_constraint,
                  nb_hidden_node=search_grid_nb_hidden_node)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)  # -1 has GPU mem issue
with Timer('train one grid'):
    grid_result = grid.fit(train_data, train_label)

# results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# # derive the following hyper parameter through cross validation
# nb_hidden_node = 1024
# dropout_ratio = 0.5
# weight_constraint = 2
# model = create_model(dropout_ratio=dropout_ratio,
#                      weight_constraint=weight_constraint,
#                      nb_hidden_node=nb_hidden_node)
#
# lossaccRTplot = LossAccRTPlot()
# # TODO revise the training by conducting validation,
# # i.e. check overfitting at bottleneck feature learning
# model.fit(train_data, train_label,
#           batch_size=batch_size,
#           nb_epoch=nb_epoch,
#           shuffle=True,
#           verbose=1,
#           # callbacks=[])
#           callbacks=[lossaccRTplot])
# model.save_weights('models/train_input{}_top2fc{}_largeset_{}epoch_DO{}_WC{}_HomeOrOff_model.h5'
#                    .format(img_height,
#                            nb_hidden_node,
#                            nb_epoch,
#                            dropout_ratio,
#                            weight_constraint))
