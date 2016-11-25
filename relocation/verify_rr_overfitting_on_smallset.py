__author__ = 'bsl'

from keras.layers import Input
from keras.applications import vgg16
from keras.layers import Dense, Flatten, Dropout, MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from utils.custom_image import ImageDataGenerator

import numpy as np
from utils.loss_acc_history_rtplot import LossRTPlot


# note rr = rotation robustness
img_height = 480    # 480 image size is > doubled to 224
img_width = 480*4
img_size = (3, img_height, img_width)   # expected: shape (nb_sample, 3, 480, 1920)
input_tensor = Input(batch_shape=(None,) + img_size)

model_vgg_places_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)
model_vgg_places_notop.load_weights('models/vgg16_places365_notop.h5')
base_model_output = model_vgg_places_notop.output

# add max pooling along horizontal direction, (vertical, horizontal), strides = pool_size if None
# Input 480 -> Pool1 240 -> Pool2 120 -> Pool3 60 -> Pool4 30 -> Pool5 15
base_model_output = MaxPooling2D(pool_size=(1, 15*4), strides=None)(base_model_output)

base_model_output = Flatten()(base_model_output)    # 512*15 = 7680
nb_fc_nodes = 1024      # model expension does not affact GPU mem usage
learning_rate_multiplier = 50.0
base_model_output = Dense(nb_fc_nodes,
                          W_learning_rate_multiplier=learning_rate_multiplier,
                          b_learning_rate_multiplier=learning_rate_multiplier,
                          activation='relu')(base_model_output)
base_model_output = Dropout(0.5)(base_model_output)
base_model_output = Dense(nb_fc_nodes,
                          W_learning_rate_multiplier=learning_rate_multiplier,
                          b_learning_rate_multiplier=learning_rate_multiplier,
                          activation='relu')(base_model_output)
base_model_output = Dropout(0.5)(base_model_output)
preds = Dense(2,
              W_learning_rate_multiplier=learning_rate_multiplier,
              b_learning_rate_multiplier=learning_rate_multiplier,
              activation='softmax')(base_model_output)
model_stacked = Model(model_vgg_places_notop.input, preds)

# no frozen layer

# use 'SGD' with low learning rate
learning_rate = 1e-5
model_stacked.compile(loss='mean_squared_error',
                      optimizer=SGD(lr=learning_rate, momentum=0.9),
                      metrics=[])

# train data
batch_size = 3     # determine the generator batch size
nb_epoch = 100
nb_train_sample = 5000

# # setting input mean to 0 over dataset, not applicable to fit_generator due to difficulty to get the whole set
# datagen_train = ImageDataGenerator(rescale=1./255, featurewise_center=True)
datagen_train = ImageDataGenerator(rescale=1./255)
generator_train = datagen_train.flow_from_directory('datasets/train_test_split_480x1920/test/',
                                                    target_size=(img_height, img_width),    # order checked
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    class_mode='xy_pos',    # possible name: 'xy_pos'
                                                    label_file="../test_label.csv")


loss_rtplot = LossRTPlot()
history_callback = model_stacked.fit_generator(generator_train,
                                               samples_per_epoch=nb_train_sample,
                                               nb_epoch=nb_epoch,
                                               validation_data=[],
                                               # callbacks=[])
                                               callbacks=[loss_rtplot])

# record the loss
record = np.column_stack((np.array(history_callback.epoch) + 1, history_callback.history['loss']))

np.savetxt('training_procedure/convergence_smallset_vggrr3fc{}_places_{}epoch_sgdlr{}m{}_reloc_model.csv'
           .format(nb_fc_nodes, (history_callback.epoch[-1]+1), learning_rate, int(learning_rate_multiplier)),
           record, delimiter=',')
model_stacked.save_weights('models/train_smallset_vggrr3fc{}_places_{}epoch_sgdlr{}m{}_reloc_model.h5'
                           .format(nb_fc_nodes,
                                   (history_callback.epoch[-1]+1),
                                   learning_rate,
                                   learning_rate_multiplier))
