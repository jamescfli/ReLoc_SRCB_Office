__author__ = 'bsl'

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.regularizers import l1l2

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


def create_model(dropout_ratio=0.5,
                 weight_decay=(0.0, 0.0),
                 nb_hidden_node=256):
    model = Sequential()
    model.add(Dense(nb_hidden_node,
                    name='FC_Dense_Regress_1',  # for further weight loading
                    activation='relu',
                    W_regularizer=l1l2(l1=weight_decay[0], l2=weight_decay[1]),
                    b_regularizer=l1l2(l1=weight_decay[0], l2=weight_decay[1]),
                    input_shape=train_data.shape[1:]))
    model.add(Dropout(dropout_ratio, name='Dropout_Regress_1'))
    # model.add(Dense(nb_hidden_node,
    #                 name='FC_Dense_Regress_2',
    #                 activation='relu',
    #                 W_regularizer=l1l2(l1=weight_decay[0], l2=weight_decay[1]),
    #                 b_regularizer=l1l2(l1=weight_decay[0], l2=weight_decay[1])))
    # model.add(Dropout(dropout_ratio, name='Dropout_Regress_2'))
    model.add(Dense(2, name='FC_Dense_Regress_3',
                    activation='linear',
                    W_regularizer=l1l2(l1=weight_decay[0], l2=weight_decay[1]),
                    b_regularizer=l1l2(l1=weight_decay[0], l2=weight_decay[1])))
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(lr=1e-5, momentum=0.9),
                  # optimizer='adadelta',
                  metrics=['mean_squared_error'])
    return model

nb_epoch = 3000
batch_size = 20  # total 20 samples

seed = 7
np.random.seed(seed)

nb_hnode = 2048
do_ratio = 0.0
L1_regular = 0.0
L2_regular = 0.0
model = create_model(dropout_ratio=do_ratio,
                     weight_decay=(L1_regular, L2_regular),
                     nb_hidden_node=nb_hnode)
# lossRTplot = LossRTPlot()
model.fit(train_data, train_label,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          shuffle=True,
          # callbacks=[lossRTplot],
          verbose=1)
model.summary()     # confirmation

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
# 64 hidden nodes
#   SGD(lr=1e-5, momentum=0.9) - 3.3980e-09 (approaching numerical limit)
# 32 hidden nodes
#   SGD(lr=1e-5, momentum=0.9) - 6.5721e-09 (less representation capability)
# 16 hidden nodes
#   SGD(lr=1e-5, momentum=0.9) - 4.5110e-09
# 8 hidden nodes (can be improved by different random seed)
#   SGD(lr=1e-5, momentum=0.9) - 218.9302 (incapable of reaching 0 w/ seed 7), 4.6471e-04 (seed 8)

# add dropout 0.5   i.e. increase regularization -> increase loss
#   128         4096 (DO0.5)    4096 (DO0.1)
#   144.8600    9.1036          1.3941

# substract one Dense (+DO) i.e. one Dense + one DO + Dense + linear activation
# optimizer SGD(lr=1e-5, momentum=0.9) for all
# 512           256         128         64          32          16          8
# 4.0327e-09    3.8266e-09  3.3826e-09  1.5830e-09  1.9569e-09  1.2538e-09  1.9231e-09

# rethink 7168 for 448* images try 2048 (comparable and has room for tuning regularizer)
# test diff DO ratio, 2048 hidden nodes
# 0.0           0.1     0.2     0.3     0.4     0.5 (default)
# 7.0598e-09    1.0005  2.4779  4.6259  5.9055  11.2066
# test diff L2 regularizer / weight decay. Note loss != MSE value if weight decay > 0
# 0.0           1e-6        1e-4        1e-2
# 6.9062e-09    0.0032      0.3200      31.6238
# test diff L1 regularizer / weight decay
# 0.0           1e-6        1e-4        1e-2
# 6.9062e-09    0.1875      18.7115     1462.4402

# if delete weight_constraint maxnorm(2)
#   2048 hidden node loss = 5.1211e-10 even further down
#   256  hidden node loss = 3.5416e-10

# model.save_weights('models/train_input{}_topfc{}_smallset_ls{}_{}epoch_DO{}_WC{}_reloc_model.h5'
#                    .format(img_height,
#                            nb_hidden_node,
#                            label_scaling_factor,    # ls - label scalar
#                            nb_epoch,
#                            dropout_ratio,
#                            weight_constraint))
