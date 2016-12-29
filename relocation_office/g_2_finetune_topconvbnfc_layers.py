__author__ = 'bsl'

from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import BatchNormalization, Activation
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras.optimizers import SGD
from keras.regularizers import l1, l2

from utils.custom_image import ImageDataGenerator
from utils.loss_acc_mse_history_rtplot import LossMseRTPlot

import numpy as np

TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

img_height = 224
img_width = img_height*4

def build_vggrrfc_bn_model(weights='places',
                           nb_fc_hidden_node=2048,
                           dropout_ratio=0.5,
                           global_learning_rate=1e-5,
                           learning_rate_multiplier=1.0,
                           l1_regularization=None,  # TODO
                           l2_regularization=None,
                           is_BN_enabled=False,
                           is_DO_enabled=False):

    if weights not in {'imagenet', 'places', 'office', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet`, `places`, `office` '
                         '(pre-training on ImageNet, Places or Office scenario).')
    # Determine proper input shape
    img_size = (3, img_height, img_width)
    img_input = Input(batch_shape=(None,) + img_size)   # i.e. input_tensor

    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1', trainable=False)(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1', trainable=False)(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1', trainable=False)(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2', trainable=False)(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1', trainable=False)(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2', trainable=False)(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    if is_BN_enabled:
        x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv1')(x)
        x = BatchNormalization(name='block5_bn1')(x)
        x = Activation('relu', name='block5_act1')(x)
        x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv2')(x)
        x = BatchNormalization(name='block5_bn2')(x)
        x = Activation('relu', name='block5_act2')(x)
        x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv3')(x)
        x = BatchNormalization(name='block5_bn3')(x)
        x = Activation('relu', name='block5_act3')(x)
    else:
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
        x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
        # # substract this layer and substitute by
        # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # rr pooling
    x = MaxPooling2D(pool_size=(1, (img_height / (2**4)) * 4), strides=None, name='rr_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(nb_fc_hidden_node,
              name='fc_dense_regress_1',
              W_learning_rate_multiplier=learning_rate_multiplier,
              b_learning_rate_multiplier=learning_rate_multiplier*2,    # *2 from LHY's practice
              W_regularizer=l2(l2_regularization) if l2_regularization else None,
              b_regularizer=l2(l2_regularization) if l2_regularization else None)(x)
    if is_BN_enabled:
        x = BatchNormalization(name='fc_bn1')(x)
    x = Activation('relu', name='fc_act1')(x)
    if is_DO_enabled:
        x = Dropout(dropout_ratio, name='fc_do_1')(x)     # can possibly be dropped if BN
    x = Dense(2,
              name='fc_dense_regress_2',
              W_learning_rate_multiplier=learning_rate_multiplier,
              b_learning_rate_multiplier=learning_rate_multiplier*2,
              W_regularizer=l2(l2_regularization) if l2_regularization else None,
              b_regularizer=l2(l2_regularization) if l2_regularization else None,
              activation='linear')(x)
    inputs = get_source_inputs(img_input)
    model = Model(inputs, x, name='vgg_blk5rrfc_bn_reg2out')

    # log_2_finetune_topconvbnfc_layers.pyad weights
    if weights == 'imagenet':
        weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5',
                                TH_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models')
        print 'loading imagenet weights ..'
        model.load_weights(weights_path, by_name=True)
    elif weights == 'places':
        weights_path = 'models/vgg16_places365_notop_weights.h5'
        print 'loading places weights ..'
        model.load_weights(weights_path, by_name=True)  # rest layer will be randomly initialized
    elif weights == 'office':
        weights_path = 'models/weights_vgg2fc256_largeset_11fzlayer_60epoch_sgdlr5e-5m10anneal20epoch_HomeOrOff_model.h5'
        print 'loading office weights ..'
        model.load_weights(weights_path, by_name=True)
    else:
        print 'NOTE: no weights loaded to the model ..'

    # frozen layers has been annotated when building
    # compile
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(lr=global_learning_rate, momentum=0.9),
                  metrics=['mean_squared_error'])
    return model

if __name__ == '__main__':
    # build model from scratch
    initial_weights = 'imagenet'
    nb_hidden_node = 2048
    learning_rate = 1e-5    # to conv layers
    lr_multiplier = 1.0     # to top fc layers
    l2_regular = 1e-3       # weight decay in L2 norm
    label_scalar = 100      # expend from [0, 1]
    flag_add_bn = True
    flag_add_do = True
    do_ratio = 0.5
    batch_size = 32         # tried 32
    nb_epoch = 30
    model_stacked = build_vggrrfc_bn_model(weights=initial_weights,
                                           nb_fc_hidden_node=nb_hidden_node,
                                           dropout_ratio=do_ratio,
                                           global_learning_rate=learning_rate,
                                           learning_rate_multiplier=lr_multiplier,
                                           l1_regularization=None,
                                           l2_regularization=l2_regular,
                                           is_BN_enabled=flag_add_bn,
                                           is_DO_enabled=flag_add_do)
    model_stacked.summary()

    # prepare training data, augmented by 5 times
    nb_train_sample = 13182 * 5

    datagen_train = ImageDataGenerator(rescale=1. / 255)
    generator_train = datagen_train.flow_from_directory(
        'datasets/train_test_split_480x1920_20161125/train_augmented/train_augmented_subdir/',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True,
        class_mode='xy_pos',
        label_file="../../train_augmented_label_x{}.csv".format(label_scalar))

    # prepare test data, apply img20161215 for generalization
    nb_test_sample = 2000
    datagen_test = ImageDataGenerator(rescale=1. / 255)
    # no shuffle
    generator_test = datagen_test.flow_from_directory(
        'datasets/test_image_20161215/image_480x1920_2000_for_test/image_480x1920_2000/',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='xy_pos',
        label_file="../../label_list_480x1920_2000_x{}.csv".format(label_scalar))

    # fit model
    loss_mse_rtplot = LossMseRTPlot()
    history_callback = model_stacked.fit_generator(generator_train,
                                                   samples_per_epoch=nb_train_sample,
                                                   nb_epoch=nb_epoch,
                                                   validation_data=generator_test,
                                                   nb_val_samples=nb_test_sample,
                                                   callbacks=[loss_mse_rtplot],
                                                   verbose=1)

    # record
    record = np.column_stack((np.array(history_callback.epoch) + 1,
                              history_callback.history['loss'],
                              history_callback.history['val_loss'],
                              history_callback.history['mean_squared_error'],
                              history_callback.history['val_mean_squared_error']))

    np.savetxt(
        'training_procedure/convergence_vggrr2fc{}bn_{}_1125imgaug_ls{}_{}epoch_sgdlr{}m{}_l2reg{}_reloc_model.csv'
        .format(nb_hidden_node,
                initial_weights,
                label_scalar,
                nb_epoch,
                learning_rate,
                int(lr_multiplier),
                l2_regular),
        record, delimiter=',')
    model_stacked_json = model_stacked.to_json()
    with open('models/structure_vggrr2fc{}bn_{}_1125imgaug_ls{}_{}epoch_sgdlr{}m{}_l2reg{}_reloc_model.h5'
                      .format(nb_hidden_node,
                              initial_weights,
                              label_scalar,
                              nb_epoch,
                              learning_rate,
                              int(lr_multiplier),
                              l2_regular), "w") \
            as json_file_model_stacked:
        json_file_model_stacked.write(model_stacked_json)
    model_stacked.save_weights(
        'models/weights_vggrr2fc{}bn_{}_1125imgaug_ls{}_{}epoch_sgdlr{}m{}_l2reg{}_reloc_model.h5'
        .format(nb_hidden_node,
                initial_weights,
                label_scalar,
                nb_epoch,
                learning_rate,
                int(lr_multiplier),
                l2_regular), overwrite=False)
