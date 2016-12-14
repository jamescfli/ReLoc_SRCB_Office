__author__ = 'bsl'

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import maxnorm
from keras.optimizers import SGD

import numpy as np
from utils.loss_acc_history_rtplot import LossRTPlot

img_height = 448
img_width = img_height * 4

train_data = np.load(open('bottleneck_data/bottleneck_feature_vggrr_20image_{}x{}.npy'
                          .format(img_height, img_width)))
# apply x100 label values
label_scaling_factor = 100
train_label = np.loadtxt('datasets/train_test_split_480x1920/20imageset_label_x{}.csv'
                         .format(label_scaling_factor),
                         dtype='float32', delimiter=',')
# # try delete x100 scaling
# train_label /= 100.0

assert train_data.shape[0] == train_label.shape[0], 'nb of data samples != nb of labels'
print('training data shape: {}'.format(train_data.shape))
print('training label shape: {}'.format(train_label.shape))


def create_model(dropout_ratio=0.5, weight_constraint=2, nb_hidden_node=256):
    model = Sequential()
    model.add(Dense(nb_hidden_node,
                    name='FC_Dense_Regress_1',  # for further weight loading
                    activation='relu',
                    W_constraint=maxnorm(weight_constraint),
                    input_shape=train_data.shape[1:]))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(nb_hidden_node,
                    name='FC_Dense_Regress_2',  # for further weight loading
                    activation='relu',
                    W_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(2, name='FC_Dense_Regress_3', activation='linear'))
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(lr=1e-5, momentum=0.9),
                  # optimizer='adadelta',
                  metrics=['mean_squared_error'])
    return model

nb_epoch = 3000
batch_size = 20  # total 20 samples

seed = 7
np.random.seed(seed)

nb_hidden_node = 4096
dropout_ratio = 0.0
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

# Result after 3000 epochs (batch_size=20) + x100 label + no DO layers (ratio=0.0) + 4096 hidden nodes
#   SGD(lr=1e-5, momentum=0.9) - 1.3175e-08 (first success)
#                                1.1925e-08 (on MBP)
#   adadelta - 8.2241 (failed)
# 2048 hidden nodes
#   SGD(lr=1e-5, momentum=0.9) - 1.0444e-08
# 1024 hidden nodes
#   SGD(lr=1e-5, momentum=0.9) - 8.6365e-09 (loss getting smaller)
# 512 hidden nodes
#   SGD(lr=1e-5, momentum=0.9) - 4.4657e-09
# 256 hidden nodes
#   SGD(lr=1e-5, momentum=0.9) - 4.6972e-09
# 128 hidden nodes
#   SGD(lr=1e-5, momentum=0.9) - 3.8389e-09
# add dropout 0.5
#   128         4096 (DO0.5)    4096 (DO0.1)
#   144.8600    9.1036          1.3941

# model.save_weights('models/train_input{}_topfc{}_smallset_ls{}_{}epoch_DO{}_WC{}_reloc_model.h5'
#                    .format(img_height,
#                            nb_hidden_node,
#                            label_scaling_factor,    # ls - label scalar
#                            nb_epoch,
#                            dropout_ratio,
#                            weight_constraint))
