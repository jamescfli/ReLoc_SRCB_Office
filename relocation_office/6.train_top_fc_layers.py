__author__ = 'bsl'


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1l2
from keras.optimizers import SGD

import numpy as np
from utils.loss_acc_mse_history_rtplot import LossMseRTPlot


img_height = 448  # options: 448, 224, original 450*1920
img_width = img_height * 4

# load train data
train_data = np.load(open('bottleneck_data/bottleneck_feature_vggrr_trainset_{}x{}.npy'
                          .format(img_height, img_width)))
label_scaling_factor = 100
train_label = np.loadtxt('datasets/train_test_split_480x1920_20161125/train_label_x{}.csv'
                         .format(label_scaling_factor),
                         dtype='float32', delimiter=',')

assert train_data.shape[0] == train_label.shape[0], 'train: nb of data samples != nb of labels'
print('training data shape: {}'.format(train_data.shape))
print('training label shape: {}'.format(train_label.shape))

# load test data
test_data = np.load(open('bottleneck_data/bottleneck_feature_vggrr_testset_{}x{}.npy'
                         .format(img_height, img_width)))
test_label = np.loadtxt('datasets/train_test_split_480x1920_20161125/test_label_x{}.csv'
                        .format(label_scaling_factor),
                        dtype='float32', delimiter=',')

assert test_data.shape[0] == test_label.shape[0], 'test: nb of data samples != nb of labels'
print('testing data shape: {}'.format(test_data.shape))
print('testing label shape: {}'.format(test_label.shape))


def create_model(nb_hidden_node=2048,
                 dropout_ratio=0.5,
                 weight_decay=(0.0, 0.0)):
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

seed = 7
np.random.seed(seed)

nb_epoch = 100
batch_size = 32
nb_hnode = 2048
do_ratio = 0.5
L1_regular = 0.0
L2_regular = 0.0
model = create_model(dropout_ratio=do_ratio,
                     weight_decay=(L1_regular, L2_regular),
                     nb_hidden_node=nb_hnode)
loss_mse_RTplot = LossMseRTPlot()
model.fit(train_data, train_label,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          shuffle=True,
          validation_data=(test_data, test_label),
          callbacks=[loss_mse_RTplot],
          verbose=1)
model.summary()     # verify
model.save_weights('models/train_input{}_top2fc{}_ls{}_{}epoch_DO{}_L1nm{}L2nm{}_reloc_model.h5'
                   .format(img_height,
                           nb_hnode,
                           label_scaling_factor,    # ls - label scalar
                           nb_epoch,
                           do_ratio,
                           L1_regular,
                           L2_regular))
