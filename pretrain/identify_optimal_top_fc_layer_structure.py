__author__ = 'bsl'

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.optimizers import SGD

import numpy as np
from utils.loss_acc_history_rtplot import LossAccRTPlot


# load large data set as training set
train_data = np.load(open('bottleneck_data/bottleneck_feature_vgg_places_largeset.npy'))
train_label_index = np.array([0]*18344+[1]*29055)
train_label = np_utils.to_categorical(train_label_index, nb_classes=2)

assert train_data.shape[0] == train_label.shape[0], 'nb of data samples != nb of labels'
print('training data shape: {}'.format(train_data.shape))
print('training label shape: {}'.format(train_label.shape))

# load small dataset as testing set
test_data = np.load(open('bottleneck_data/bottleneck_feature_vgg_places_smallset.npy'))
test_label_index = np.array([0]*3000+[1]*3000)
test_label = np_utils.to_categorical(test_label_index, nb_classes=2)

assert test_data.shape[0] == test_label.shape[0], 'nb of data samples != nb of labels'
print('testing data shape: {}'.format(test_data.shape))
print('testing label shape: {}'.format(test_label.shape))

assert train_data.shape[1:] == test_data.shape[1:], 'shape of training and testing data do not match!'


def create_model(dropout_ratio=0.5, weight_constraint=2, nb_hidden_node=512):
    model = Sequential()
    model.add(Dense(nb_hidden_node,
                    name='FC_Dense_1',  # for further weight loading
                    activation='relu',
                    W_constraint=maxnorm(weight_constraint),
                    input_shape=train_data.shape[1:]))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(2, name='FC_Dense_2', activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=1e-5, momentum=0.9),
                  # optimizer='adadelta',
                  # optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

nb_epoch = 100
batch_size = 32

seed = 7
np.random.seed(seed)

# derive the following hyper parameter through cross validation
nb_hn = 256
do_ratio = 0.5
weight_con = 2
model = create_model(dropout_ratio=do_ratio,
                     weight_constraint=weight_con,
                     nb_hidden_node=nb_hn)

loss_acc_RTplot = LossAccRTPlot()
# check overfitting at bottleneck feature learning
model.fit(train_data, train_label,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          shuffle=True,
          validation_data=(test_data, test_label),
          verbose=1,
          # callbacks=[])
          callbacks=[loss_acc_RTplot])
img_height = 224
model.save_weights('models/train_input{}_top2fc{}_largeset_{}epoch_DO{}_WC{}_sgd1e-5_HomeOrOff_model.h5'
                   .format(img_height,
                           nb_hn,
                           nb_epoch,
                           do_ratio,
                           weight_con))