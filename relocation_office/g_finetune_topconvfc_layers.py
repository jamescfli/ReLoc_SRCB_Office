__author__ = 'bsl'

from keras.models import Model
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Dropout
from keras.applications import vgg16
from keras.optimizers import SGD

from utils.custom_image import ImageDataGenerator
from utils.loss_acc_mse_history_rtplot import LossMseRTPlot

import numpy as np


img_height = 448
img_width = img_height*4


def build_vggrrfc_model(nb_fc_hidden_node=2048,
                        dropout_ratio=0.5,
                        nb_frozen_layer=0,
                        global_learning_rate=1e-5,
                        learning_rate_multiplier=1.0):
    img_size = (3, img_height, img_width)
    input_tensor = Input(batch_shape=(None,) + img_size)

    vgg_office_model_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)
    print 'loading office weights ..'
    office_weight_file = 'weights_vgg2fc256_largeset_11fzlayer_60epoch_sgdlr5e-5m10anneal20epoch_HomeOrOff_model.h5'
    vgg_office_model_notop.load_weights('models/'+office_weight_file, by_name=True)
    vggrr_model_output = vgg_office_model_notop.output
    vggrr_model_output = MaxPooling2D(pool_size=(1, (img_height/32) * 4), strides=None)(vggrr_model_output)
    vggrr_model_output = Flatten()(vggrr_model_output)

    vggrr_model_output = Dense(nb_fc_hidden_node,
                               name='FC_Dense_Regress_1',
                               W_learning_rate_multiplier=learning_rate_multiplier,
                               b_learning_rate_multiplier=learning_rate_multiplier,
                               activation='relu')(vggrr_model_output)
    vggrr_model_output = Dropout(dropout_ratio, name='Dropout_Regress_1')(vggrr_model_output)
    vggrr_model_output = Dense(2,
                               name='FC_Dense_Regress_3',
                               W_learning_rate_multiplier=learning_rate_multiplier,
                               b_learning_rate_multiplier=learning_rate_multiplier,
                               activation='linear')(vggrr_model_output)
    vggrr_model = Model(vgg_office_model_notop.input, vggrr_model_output)
    toplayer_weight_file = 'train_input448_top2fc2048_ls100_100epoch_DO0.5_L1nm0.0L2nm0.0_reloc_model.h5'
    vggrr_model.load_weights('models/' + toplayer_weight_file, by_name=True)

    for layer in vggrr_model.layers[:nb_frozen_layer]:
        layer.trainable = False

    vggrr_model.compile(loss='mean_squared_error',
                        optimizer=SGD(lr=global_learning_rate, momentum=0.9),
                        # optimizer='rmsprop',
                        metrics=['mean_squared_error'])
    return vggrr_model

if __name__ == '__main__':
    # build model from scratch
    nb_hidden_node = 2048
    do_ratio = 0.5
    nb_fzlayer = 11         # 11 block4, 15 block5, 19 top fc
    learning_rate = 1e-5    # to conv layers
    lr_multiplier = 1.0     # to top fc layers
    label_scalar = 100      # expend from [0, 1]
    model_stacked = build_vggrrfc_model(nb_fc_hidden_node=nb_hidden_node,
                                        dropout_ratio=do_ratio,
                                        nb_frozen_layer=nb_fzlayer,
                                        global_learning_rate=learning_rate,
                                        learning_rate_multiplier=lr_multiplier)

    batch_size = 16
    nb_epoch = 30
    # prepare training data
    nb_train_sample = 13182

    datagen_train = ImageDataGenerator(rescale=1./255)
    generator_train = datagen_train.flow_from_directory('datasets/train_test_split_480x1920_20161125/train/train_subdir/',
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        class_mode='xy_pos',
                                                        label_file="../../train_label_x{}.csv".format(label_scalar))

    # prepare test data
    nb_test_sample = 2000
    datagen_test = ImageDataGenerator(rescale=1./255)
    # no shuffle
    generator_test = datagen_test.flow_from_directory('datasets/train_test_split_480x1920_20161125/test/test_subdir/',
                                                      target_size=(img_height, img_width),
                                                      batch_size=batch_size,
                                                      class_mode='xy_pos',
                                                      label_file="../../test_label_x{}.csv".format(label_scalar))

    # fit model
    loss_mse_rtplot = LossMseRTPlot()
    history_callback = model_stacked.fit_generator(generator_train,
                                                   samples_per_epoch=nb_train_sample,
                                                   nb_epoch=nb_epoch,
                                                   validation_data=generator_test,
                                                   nb_val_samples=nb_test_sample,
                                                   # callbacks=[])
                                                   callbacks=[loss_mse_rtplot],
                                                   verbose=1)
    model_stacked.summary()

    # record
    record = np.column_stack((np.array(history_callback.epoch) + 1,
                              history_callback.history['loss'],
                              history_callback.history['val_loss'],
                              history_callback.history['mean_squared_error'],
                              history_callback.history['val_mean_squared_error']))

    np.savetxt('training_procedure/convergence_vggrr2fc{}_20161125img_{}fzlayer_ls{}_{}epoch_sgdlr{}m{}_reloc_model.csv'
               .format(nb_hidden_node,
                       nb_fzlayer,
                       label_scalar,
                       nb_epoch,
                       learning_rate,
                       int(lr_multiplier)),
               record, delimiter=',')
    model_stacked_json = model_stacked.to_json()
    with open('models/structure_vggrr2fc{}_20161125img_{}fzlayer_ls{}_{}epoch_sgdlr{}m{}_reloc_model.h5'
                      .format(nb_hidden_node,
                              nb_fzlayer,
                              label_scalar,
                              nb_epoch,
                              learning_rate,
                              int(lr_multiplier)), "w") \
            as json_file_model_stacked:
        json_file_model_stacked.write(model_stacked_json)
    model_stacked.save_weights('models/weights_vggrr2fc{}_20161125img_{}fzlayer_ls{}_{}epoch_sgdlr{}m{}_reloc_model.h5'
                               .format(nb_hidden_node,
                                       nb_fzlayer,
                                       label_scalar,
                                       nb_epoch,
                                       learning_rate,
                                       int(lr_multiplier)), overwrite=False)
