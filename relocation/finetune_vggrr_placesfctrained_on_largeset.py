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
                        nb_frozen_layer=0,
                        global_learning_rate=1e-5,
                        learning_rate_multiplier=1.0,
                        label_scaling_factor=1):
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
    vggrr_model = Model(vgg_places_model_notop.input, vggrr_model_output)
    vggrr_model.load_weights('models/train_input{}_topfc{}_smallset_ls{}_500epoch_DO{}_WC{}_reloc_model.h5'
                             .format(img_height,
                                     nb_fc_hidden_node,
                                     label_scaling_factor,
                                     dropout_ratio,
                                     weight_constraint),
                             by_name=True)

    # set frozen layers
    for layer in vggrr_model.layers[:nb_frozen_layer]:
        layer.trainable = False

    vggrr_model.compile(loss='mean_squared_error',
                        # optimizer=SGD(lr=global_learning_rate, momentum=0.9),
                        optimizer='adadelta',   # keep apply 'adadelta'
                        metrics=[])
    return vggrr_model      # total 26 layers

def load_vggrrfc_model(nb_fc_hidden_node=1024,
                       dropout_ratio=0.5,
                       weight_constraint=2,
                       nb_frozen_layer=0,
                       global_learning_rate=1e-5,
                       learning_rate_multiplier=1.0,
                       label_scaling_factor=1,
                       model_weight_path=None):
    if model_weight_path == None:
        print 'please provide path to the model weights'
        return None
    img_size = (3, img_height, img_width)
    input_tensor = Input(batch_shape=(None,) + img_size)

    vgg_places_model_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)
    vggrr_model_output = vgg_places_model_notop.output
    vggrr_model_output = MaxPooling2D(pool_size=(1, (img_height / 32) * 4), strides=None)(vggrr_model_output)
    vggrr_model_output = Flatten()(vggrr_model_output)

    vggrr_model_output = Dense(nb_fc_hidden_node,
                               name='FC_Dense_1',
                               W_constraint=maxnorm(weight_constraint),
                               W_learning_rate_multiplier=learning_rate_multiplier,
                               b_learning_rate_multiplier=learning_rate_multiplier,
                               activation='relu')(vggrr_model_output)
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
    vggrr_model = Model(vgg_places_model_notop.input, vggrr_model_output)
    vggrr_model.load_weights(model_weight_path)

    # set frozen layers
    for layer in vggrr_model.layers[:nb_frozen_layer]:
        layer.trainable = False

    vggrr_model.compile(loss='mean_squared_error',
                        optimizer=SGD(lr=global_learning_rate, momentum=0.9),
                        # optimizer='adadelta',  # keep apply 'adadelta'
                        metrics=[])
    return vggrr_model


# build model from scratch
nb_hidden_node = 1024
do_ratio = 0.5
weight_con = 2
nb_fzlayer = 15         # 11 block4, 15 block5, 19 top fc
learning_rate = 1e-5            # to conv layers
lr_multiplier = 1.0    # to top fc layers
label_scalar = 100      # expend from [0, 1]
model_stacked = build_vggrrfc_model(nb_fc_hidden_node=nb_hidden_node,
                                    dropout_ratio=do_ratio,
                                    weight_constraint=weight_con,
                                    nb_frozen_layer=nb_fzlayer,
                                    global_learning_rate=learning_rate,
                                    learning_rate_multiplier=lr_multiplier,
                                    label_scaling_factor=label_scalar)
# # build model from trained one, TODO better use json to save model structure
# model_wt_path = 'models/train_vggrr3fc1024_largeset_15fzlayer_ls100_8epoch_sgdlr1e-05m1.0_reloc_model.h5'
# model_stacked = load_vggrrfc_model(nb_fc_hidden_node=nb_hidden_node,
#                                    dropout_ratio=do_ratio,
#                                    weight_constraint=weight_con,
#                                    nb_frozen_layer=nb_fzlayer,
#                                    global_learning_rate=learning_rate,
#                                    learning_rate_multiplier=lr_multiplier,
#                                    label_scaling_factor=label_scalar,
#                                    model_weight_path=model_wt_path)


batch_size = 16     # higher size, e.g. 16, due to less tranable layers
nb_epoch = 32
# prepare training data
nb_train_sample = 13182

datagen_train = ImageDataGenerator(rescale=1./255)
generator_train = datagen_train.flow_from_directory('datasets/train_test_split_480x1920/train/train_subdir/',
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    class_mode='xy_pos',
                                                    label_file="../../train_label_x{}.csv".format(label_scalar))

# prepare test data
nb_test_sample = 2000
datagen_test = ImageDataGenerator(rescale=1./255)
# no shuffle
generator_test = datagen_test.flow_from_directory('datasets/train_test_split_480x1920/test/test_subdir/',
                                                  target_size=(img_height, img_width),
                                                  batch_size=batch_size,
                                                  class_mode='xy_pos',
                                                  label_file="../../test_label_x{}.csv".format(label_scalar))

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

np.savetxt('training_procedure/convergence_vggrr3fc{}_largeset_{}fzlayer_ls{}_{}epoch_sgdlr{}m{}_reloc_model.csv'
           .format(nb_hidden_node,
                   nb_fzlayer,
                   label_scalar,
                   (history_callback.epoch[-1]+1),
                   learning_rate,
                   int(lr_multiplier)),
           record, delimiter=',')
model_stacked.save_weights('models/train_vggrr3fc{}_largeset_{}fzlayer_ls{}_{}epoch_sgdlr{}m{}_reloc_model.h5'
                           .format(nb_hidden_node,
                                   nb_fzlayer,
                                   label_scalar,
                                   (history_callback.epoch[-1]+1),
                                   learning_rate,
                                   lr_multiplier))
