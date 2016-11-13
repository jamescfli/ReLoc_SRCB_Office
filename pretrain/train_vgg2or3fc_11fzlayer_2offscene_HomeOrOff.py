__author__ = 'bsl'

from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.layers import Input
from keras.models import Model
# from utils.timer import Timer
from keras.callbacks import EarlyStopping

# original size for train challenge is 256x256
img_width = 224
img_height = 224
img_size = (3, img_width, img_height)

# with Timer('model & data preparation'):   # Timer started here
input_tensor = Input(batch_shape=(None,) + img_size)
model_vgg = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
base_model_output = model_vgg.output
base_model_output = Flatten()(base_model_output)
nb_fc_nodes = 512
base_model_output = Dense(nb_fc_nodes, activation='relu')(base_model_output)
base_model_output = Dropout(0.5)(base_model_output)
# # add one more fc layer with 256 hidden nodes
# base_model_output = Dense(nb_fc_nodes, activation='relu')(base_model_output)
# base_model_output = Dropout(0.5)(base_model_output)
preds = Dense(2, activation='softmax')(base_model_output)   # 2 scenes: home_office and office
model_stacked = Model(model_vgg.input, preds)   # fc layers are randomly initiated

# reset trainable layers in VGG16 from keras.applications
nb_frozen_layers = 11
for layer in model_stacked.layers[:nb_frozen_layers]:
    layer.trainable = False
# 19 - train top fc layers only
# 15 - train block5 and fc layers
# 11 - train block5,4 and fc layers

# use 'SGD' with low learning rate
# learning_rate = 1e-5
model_stacked.compile(loss='categorical_crossentropy',
                      # optimizer=SGD(lr=learning_rate, momentum=0.9),   # for fine tuning
                      optimizer='adadelta',                      # train from imagenet
                      metrics=['accuracy'])

# train data
batch_size = 64    # used to be 32
datagen_train = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
generator_train = datagen_train.flow_from_directory('datasets/data_256_HomeOrOff/train',
                                                    target_size=(img_height,img_width),
                                                    batch_size=batch_size,
                                                    shuffle=True,   # default is True
                                                    class_mode='categorical')
# test
datagen_test = ImageDataGenerator(rescale=1./255)
generator_test = datagen_test.flow_from_directory('datasets/data_256_HomeOrOff/test',
                                                  target_size=(img_height,img_width),
                                                  batch_size=batch_size,
                                                  shuffle=True,   # default is True
                                                  class_mode='categorical')
# Timer ended here

nb_epoch = 1       # 1 epoch in ~890 sec
nb_train_samples = 51399    # 2016/11/03 20344+31055 = 51399
nb_test_samples = 2000      # 2016/11/03 1000*2
early_stopping = EarlyStopping(monitor='loss', patience=2)  # number of epochs with no improvement
history_callback = model_stacked.fit_generator(generator_train,
                                               samples_per_epoch=nb_train_samples,
                                               nb_epoch=nb_epoch,
                                               validation_data=generator_test,
                                               nb_val_samples=nb_test_samples,
                                               callbacks=[early_stopping])

import numpy as np
record = np.column_stack((np.array(history_callback.epoch) + 1,
                          history_callback.history['loss'],
                          history_callback.history['acc'],
                          history_callback.history['val_loss'],
                          history_callback.history['val_acc']))
# np.savetxt('training_procedure/convergence_vgg2fc{}_{}fzlayer_{}epoch_lr{}_2class_HomeOrOff_model.csv'
#            .format(nb_fc_nodes, nb_frozen_layers, nb_epoch, learning_rate), record, delimiter=',')
np.savetxt('training_procedure/convergence_vgg2fc{}_{}fzlayer_{}epoch_adadelta_2class_HomeOrOff_model.csv'
           .format(nb_fc_nodes, nb_frozen_layers, nb_epoch), record, delimiter=',')

# save the pretrained parameter into models folder
# model_stacked.save_weights('models/train_vgg2fc{}_{}fzlayer_{}epoch_lr{}_2class_HomeOrOff_model.h5'
#                            .format(nb_fc_nodes, nb_frozen_layers, nb_epoch, learning_rate))
model_stacked.save_weights('models/train_vgg2fc{}_{}fzlayer_{}epoch_adadelta_2class_HomeOrOff_model.h5'
                           .format(nb_fc_nodes, nb_frozen_layers, nb_epoch))
