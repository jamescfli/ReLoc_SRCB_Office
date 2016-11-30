__author__ = 'bsl'

from keras.models import Model, Sequential
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Dropout
from keras.constraints import maxnorm
from keras.applications import vgg16
from keras.optimizers import SGD

from utils.custom_image import ImageDataGenerator
from utils.loss_acc_history_rtplot import LossRTPlot

import numpy as np


img_height = 448
img_width = img_height*4


def build_vggrrfc_model(vgg_initial_weights='places',
                        nb_fc_hidden_node=1024,
                        dropout_ratio=0.5,
                        weight_constraint=2,
                        nb_fzlayer=0,
                        global_learning_rate=1e-5,
                        learning_rate_multiplier=100.0):
    img_size = (3, img_height, img_width)  # expected: shape (nb_sample, 3, 480, 1920)
    input_tensor = Input(batch_shape=(None,) + img_size)

    vgg_places_model_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)
    if vgg_initial_weights == 'places':
        print 'loading places weights ..'
        vgg_places_model_notop.load_weights('models/vgg16_places365_notop.h5')
    else:   # o.w leave it as ImageNet
        print 'keep using imagenet weights ..'
    vggrr_model_output = vgg_places_model_notop.output
    vggrr_model_output = MaxPooling2D(pool_size=(1, (img_height/32) * 4), strides=None)(vggrr_model_output)
    vggrr_model_output = Flatten()(vggrr_model_output)

    vggrr_model_output = Dense(nb_fc_hidden_node,
                               name='FC_Dense_1',
                               W_constraint=maxnorm(weight_constraint),
                               W_learning_rate_multiplier=learning_rate_multiplier,
                               b_learning_rate_multiplier=learning_rate_multiplier,
                               activation='relu')(vggrr_model_output)
    # .. vggrr_model.layers[21].get_weights()[0].shape = (7680, 256)
    vggrr_model_output = Dropout(dropout_ratio)(vggrr_model_output)
    vggrr_model_output = Dense(nb_fc_hidden_node,
                               name='FC_Dense_2',
                               W_constraint=maxnorm(weight_constraint),
                               W_learning_rate_multiplier=learning_rate_multiplier,
                               b_learning_rate_multiplier=learning_rate_multiplier,
                               activation='relu')(vggrr_model_output)
    vggrr_model_output = Dropout(dropout_ratio)(vggrr_model_output)
    vggrr_model_output = Dense(2,
                               name='FC_Dense_3',
                               W_learning_rate_multiplier=learning_rate_multiplier,
                               b_learning_rate_multiplier=learning_rate_multiplier,
                               activation='linear')(vggrr_model_output)
    # .. vggrr_model.layers[25].get_weights()[0].shape = (256,2)
    vggrr_model = Model(vgg_places_model_notop.input, vggrr_model_output)
    # # debug: verify loading weights
    # w_init_weights = vggrr_model.layers[21].get_weights()[0]
    # b_init_weights = vggrr_model.layers[21].get_weights()[1]
    vggrr_model.load_weights('models/train_input448_topfc1024_smallset_100epoch_DO0.5_WC2_reloc_model.h5',
                             by_name=True)
    # w_load_weights = vggrr_model.layers[21].get_weights()[0]
    # b_load_weights = vggrr_model.layers[21].get_weights()[1]
    # # debug: verify loading weights
    # print w_init_weights.shape
    # print w_init_weights.shape == w_load_weights.shape
    # print np.array_equal(w_init_weights, w_load_weights)
    # print b_init_weights.shape == b_load_weights.shape
    # print np.array_equal(b_init_weights, b_load_weights)

    # set frozen layers
    for layer in vggrr_model.layers[:nb_fzlayer]:
        layer.trainable = False

    vggrr_model.compile(loss='mean_squared_error',
                        optimizer=SGD(lr=global_learning_rate, momentum=0.9),
                        metrics=[])
    return vggrr_model      # total 26 layers

# build model
nb_fc_hidden_node = 1024
dropout_ratio = 0.5
weight_constraint = 2
nb_fzlayer = 11         # 11 block4, 15 block5, 19 top fc
learning_rate = 1e-4            # to conv layers
learning_rate_multiplier = 1.0    # to top fc layers
model_stacked = build_vggrrfc_model(nb_fc_hidden_node=nb_fc_hidden_node,
                                    dropout_ratio=dropout_ratio,
                                    weight_constraint=weight_constraint,
                                    nb_fzlayer=nb_fzlayer,
                                    global_learning_rate=learning_rate,
                                    learning_rate_multiplier=learning_rate_multiplier)

batch_size = 16
nb_epoch = 30
# prepare training data
nb_train_sample = 13182

datagen_train = ImageDataGenerator(rescale=1./255)
generator_train = datagen_train.flow_from_directory('datasets/train_test_split_480x1920/train/train_subdir/',
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    class_mode='xy_pos',
                                                    label_file="../../train_label.csv")

# prepare test data
nb_test_sample = 2000
datagen_test = ImageDataGenerator(rescale=1./255)
# no shuffle
generator_test = datagen_test.flow_from_directory('datasets/train_test_split_480x1920/test/test_subdir/',
                                                  target_size=(img_height, img_width),
                                                  batch_size=batch_size,
                                                  class_mode='xy_pos',
                                                  label_file="../../test_label.csv")

# fit model
loss_rtplot = LossRTPlot()
history_callback = model_stacked.fit_generator(generator_train,
                                               samples_per_epoch=nb_train_sample,
                                               nb_epoch=nb_epoch,
                                               validation_data=generator_test,
                                               nb_val_samples=nb_test_sample,
                                               # callbacks=[])
                                               callbacks=[loss_rtplot])

# record
record = np.column_stack((np.array(history_callback.epoch) + 1,
                          history_callback.history['loss'],
                          history_callback.history['val_loss']))

np.savetxt('training_procedure/convergence_vggrr3fc{}_largeset_{}fzlayer_{}epoch_sgdlr{}m{}_reloc_model.csv'
           .format(nb_fc_hidden_node,
                   nb_fzlayer,
                   (history_callback.epoch[-1]+1),
                   learning_rate,
                   int(learning_rate_multiplier)),
           record, delimiter=',')
model_stacked.save_weights('models/train_vggrr3fc{}_largeset_{}fzlayer_{}epoch_sgdlr{}m{}_reloc_model.h5'
                           .format(nb_fc_hidden_node,
                                   nb_fzlayer,
                                   (history_callback.epoch[-1]+1),
                                   learning_rate,
                                   learning_rate_multiplier))
