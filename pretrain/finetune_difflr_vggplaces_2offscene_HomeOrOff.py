__author__ = 'bsl'

from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.layers import Input
from keras.models import Model
from utils.loss_acc_history_rtplot import LossAccRTPlot

# original size for train challenge is 256x256
img_width = 224
img_height = 224
img_size = (3, img_width, img_height)

input_tensor = Input(batch_shape=(None,) + img_size)

model_vgg_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)

base_model_output = model_vgg_notop.output
base_model_output = Flatten()(base_model_output)
nb_fc_nodes = 512
base_model_output = Dense(nb_fc_nodes,
                          W_learning_rate_multiplier=50.0,
                          b_learning_rate_multiplier=50.0,
                          activation='relu')(base_model_output)
base_model_output = Dropout(0.5)(base_model_output)
# # add one more fc layer with 'nb_fc_nodes' hidden nodes
# base_model_output = Dense(nb_fc_nodes, activation='relu')(base_model_output)
# base_model_output = Dropout(0.5)(base_model_output)
preds = Dense(2,
              W_learning_rate_multiplier=50.0,
              b_learning_rate_multiplier=50.0,
              activation='softmax')(base_model_output)   # 2 scenes: home_office and office
model_stacked_pretrained = Model(model_vgg_notop.input, preds)   # fc layers are randomly initiated
model_file = 'train_vgg2fc512_places_19fzlayer_20epoch_adadelta_2class_HomeOrOff_model.h5'
# keep fine tuning with 1e-5 for more epochs
# model_file = 'train_vgg2fc512_places_fineT_11fzlayer_50epoch_lr1e-5_2class_HomeOrOff_model.h5'
model_stacked_pretrained.load_weights('models/'+model_file)

# reset trainable layers in VGG16 from keras.applications
nb_frozen_layers = 0
for layer in model_stacked_pretrained.layers[:nb_frozen_layers]:
    layer.trainable = False
# 19 - train top fc layers only, ~400s/epoch
# 15 - train block5 and fc layers, s/epoch
# 11 - train block5,4 and fc layers, ~800s/epoch

# use 'SGD' with low learning rate
learning_rate = 1e-5
model_stacked_pretrained.compile(loss='categorical_crossentropy',
                                 optimizer=SGD(lr=learning_rate, momentum=0.9),   # for fine tuning
                                 metrics=['accuracy'])

# train data
batch_size = 32    # used to be 32
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

# train 50 first to check the consistence btw loss and val_loss, try different lr's
nb_epoch = 100       # 1 epoch in ~890 sec TitanX ~530 sec GTX1080, 17043 sec CPU without interference
nb_train_samples = 51399    # 2016/11/03 20344+31055 = 51399
nb_test_samples = 2000      # 2016/11/03 1000*2
loss_acc_rtplot = LossAccRTPlot()
history_callback = model_stacked_pretrained.fit_generator(generator_train,
                                                          samples_per_epoch=nb_train_samples,
                                                          nb_epoch=nb_epoch,
                                                          validation_data=generator_test,
                                                          nb_val_samples=nb_test_samples,
                                                          callbacks=[loss_acc_rtplot])

import numpy as np
record = np.column_stack((np.array(history_callback.epoch) + 1,
                          history_callback.history['loss'],
                          history_callback.history['acc'],
                          history_callback.history['val_loss'],
                          history_callback.history['val_acc']))


np.savetxt('training_procedure/convergence_vgg2fc{}_places_fineTdifflr_{}epoch_sgdlr{}_2class_HomeOrOff_model.csv'
           .format(nb_fc_nodes, (history_callback.epoch[-1]+1), learning_rate), record, delimiter=',')
model_stacked_pretrained\
    .save_weights('models/train_vgg2fc{}_places_fineTdifflr_{}epoch_lr{}_2class_HomeOrOff_model.h5'
                  .format(nb_fc_nodes, (history_callback.epoch[-1]+1), learning_rate))
