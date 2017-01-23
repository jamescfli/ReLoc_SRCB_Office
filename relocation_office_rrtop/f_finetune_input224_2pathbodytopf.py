__author__ = 'bsl'

from keras.layers import (
    Input,
    Cropping2D,
    Convolution2D,
    MaxPooling2D,
    BatchNormalization,
    Activation,
    Flatten,
    Dense,
    Dropout,
    merge
)
from keras.regularizers import l1l2
from keras.engine.topology import get_source_inputs
from keras.models import Model
from keras.optimizers import SGD
from keras.applications import vgg16

from utils.custom_image import ImageDataGenerator
from utils.loss_acc_mse_history_rtplot import LossMseRTPlot
from utils.lr_annealing import LearningRateAnnealing
from keras.callbacks import ModelCheckpoint

import numpy as np
import os


def build_2path_vgg_bodytopf_model(img_height=224,
                                   weights='imagenet',
                                   nb_fc_hidden_node=2048,
                                   dropout_ratio=0.5,
                                   global_learning_rate=1e-5,
                                   learning_rate_multiplier=1.0,
                                   l1_regularization=0.0,
                                   l2_regularization=0.0,
                                   is_bn_enabled=False,
                                   is_do_enabled=False):

    if weights not in {'imagenet', 'places', 'office', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet`, `places`, `office` '
                         '(pre-training on ImageNet, Places or Office scenario).')
    # Determine proper input shape
    img_width = img_height*5    # where x4 for the body part, and x1 for the top face
    img_size = (3, img_height, img_width)
    # Note Input is not a class, but a method
    img_input = Input(batch_shape=(None,) + img_size, name='body_topf_concat_input')

    # fork to img body and img top face
    body_path_x = Cropping2D(cropping=((0,0), (0,img_height)),  name='cut_body_input')(img_input) # cut right hxh patch
    topf_path_x = Cropping2D(cropping=((0,0), (img_height*4,0)), name='cut_topf_input')(img_input) # cut left hx4h patch

    # freeze Block 1~4 in VGG
    # body
    #   block 1
    body_path_x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1_body', trainable=False)(body_path_x)
    body_path_x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2_body', trainable=False)(body_path_x)
    body_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool_body', trainable=False)(body_path_x)
    #   block 2
    body_path_x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1_body', trainable=False)(body_path_x)
    body_path_x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2_body', trainable=False)(body_path_x)
    body_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool_body', trainable=False)(body_path_x)
    #   block 3
    body_path_x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1_body', trainable=False)(body_path_x)
    body_path_x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2_body', trainable=False)(body_path_x)
    body_path_x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3_body', trainable=False)(body_path_x)
    body_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_body', trainable=False)(body_path_x)
    #   block 4
    body_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1_body', trainable=False)(body_path_x)
    body_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2_body', trainable=False)(body_path_x)
    body_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3_body', trainable=False)(body_path_x)
    body_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_body', trainable=False)(body_path_x)

    # top face
    topf_path_x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1_topf', trainable=False)(topf_path_x)
    topf_path_x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2_topf', trainable=False)(topf_path_x)
    topf_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool_topf', trainable=False)(topf_path_x)
    #   block 2
    topf_path_x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1_topf', trainable=False)(topf_path_x)
    topf_path_x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2_topf', trainable=False)(topf_path_x)
    topf_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool_topf', trainable=False)(topf_path_x)
    #   block 3
    topf_path_x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1_topf', trainable=False)(topf_path_x)
    topf_path_x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2_topf', trainable=False)(topf_path_x)
    topf_path_x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3_topf', trainable=False)(topf_path_x)
    topf_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_topf', trainable=False)(topf_path_x)
    #   block 4
    topf_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1_topf', trainable=False)(topf_path_x)
    topf_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2_topf', trainable=False)(topf_path_x)
    topf_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3_topf', trainable=False)(topf_path_x)
    topf_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_topf', trainable=False)(topf_path_x)

    # Block 5
    if is_bn_enabled:
        # body
        body_path_x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv1_body')(body_path_x)
        body_path_x = BatchNormalization(name='block5_bn1_body')(body_path_x)
        body_path_x = Activation('relu', name='block5_act1_body')(body_path_x)
        body_path_x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv2_body')(body_path_x)
        body_path_x = BatchNormalization(name='block5_bn2_body')(body_path_x)
        body_path_x = Activation('relu', name='block5_act2_body')(body_path_x)
        body_path_x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv3_body')(body_path_x)
        body_path_x = BatchNormalization(name='block5_bn3_body')(body_path_x)
        body_path_x = Activation('relu', name='block5_act3_body')(body_path_x)
        # top face
        topf_path_x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv1_topf')(topf_path_x)
        topf_path_x = BatchNormalization(name='block5_bn1_topf')(topf_path_x)
        topf_path_x = Activation('relu', name='block5_act1_topf')(topf_path_x)
        topf_path_x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv2_topf')(topf_path_x)
        topf_path_x = BatchNormalization(name='block5_bn2_topf')(topf_path_x)
        topf_path_x = Activation('relu', name='block5_act2_topf')(topf_path_x)
        topf_path_x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv3_topf')(topf_path_x)
        topf_path_x = BatchNormalization(name='block5_bn3_topf')(topf_path_x)
        topf_path_x = Activation('relu', name='block5_act3_topf')(topf_path_x)
        # Exception: The name "block1_conv1" is used 2 times in the model. All layer names should be unique.
    else:
        # body
        body_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1_body')(body_path_x)
        body_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2_body')(body_path_x)
        body_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3_body')(body_path_x)
        # top face
        topf_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1_topf')(topf_path_x)
        topf_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2_topf')(topf_path_x)
        topf_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3_topf')(topf_path_x)

    # body: add rr pooling
    body_path_x = MaxPooling2D(pool_size=(1, (img_height / (2**4)) * 4), strides=None, name='rr_pool_body')(body_path_x)
    body_path_x = Flatten(name='flatten_body')(body_path_x)     # only one flatten layer so far, no index
    body_path_x = Dense(
            nb_fc_hidden_node,
            name='fc_dense_1_body',
            W_learning_rate_multiplier=learning_rate_multiplier,
            b_learning_rate_multiplier=learning_rate_multiplier*2,    # *2 Caffe practice
            W_regularizer=l1l2(l1=l1_regularization, l2=l2_regularization) if (l1_regularization > 0)
                                                                              or (l2_regularization > 0) else None,
            b_regularizer=l1l2(l1=l1_regularization, l2=l2_regularization) if (l1_regularization > 0)
                                                                              or (l2_regularization > 0) else None)\
        (body_path_x)
    if is_bn_enabled:
        body_path_x = BatchNormalization(name='fc_bn1_body')(body_path_x)
    body_path_x = Activation('relu', name='fc_act1_body')(body_path_x)
    if is_do_enabled:
        body_path_x = Dropout(dropout_ratio, name='fc_do_1_body')(body_path_x)
    # top face: normal max pooling
    topf_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool_topf')(topf_path_x)
    topf_path_x = Flatten(name='flatten_topf')(topf_path_x)
    topf_path_x = Dense(
            nb_fc_hidden_node/4,    # due img size 1x1 rather than 1x4 for top face, reduce dimen of fc by 4 times
            name='fc_dense_1_topf',
            W_learning_rate_multiplier=learning_rate_multiplier,
            b_learning_rate_multiplier=learning_rate_multiplier*2,    # *2 Caffe practice
            W_regularizer=l1l2(l1=l1_regularization, l2=l2_regularization) if (l1_regularization > 0)
                                                                              or (l2_regularization > 0) else None,
            b_regularizer=l1l2(l1=l1_regularization, l2=l2_regularization) if (l1_regularization > 0)
                                                                              or (l2_regularization > 0) else None)\
        (topf_path_x)
    if is_bn_enabled:
        topf_path_x = BatchNormalization(name='fc_bn1_topf')(topf_path_x)
    topf_path_x = Activation('relu', name='fc_act1_topf')(topf_path_x)
    if is_do_enabled:
        topf_path_x = Dropout(dropout_ratio, name='fc_do_1_topf')(topf_path_x)

    body_topf_comb_x = merge([body_path_x, topf_path_x], mode='concat', concat_axis=1, name='concat_body_and_topf')

    x = Dense(2,
              name='fc_dense_2_comb',
              W_learning_rate_multiplier=learning_rate_multiplier,
              b_learning_rate_multiplier=learning_rate_multiplier*2,
              W_regularizer=l1l2(l1=l1_regularization, l2=l2_regularization) if (l1_regularization > 0)
                                                                                or (l2_regularization > 0) else None,
              b_regularizer=l1l2(l1=l1_regularization, l2=l2_regularization) if (l1_regularization > 0)
                                                                                or (l2_regularization > 0) else None,
              activation='linear')(body_topf_comb_x)
    inputs = get_source_inputs(img_input)
    model = Model(inputs, x, name='vgg_body_topf_2path_model')

    if weights == 'imagenet':
        # weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5',
        #                         TH_WEIGHTS_PATH_NO_TOP,
        #                         cache_subdir='models')
        print 'loading imagenet weights ..'
        # model.load_weights(weights_path, by_name=True)
        model = load_imagenet_weights(model)
    elif weights == 'places':
        # weights_path = 'models/vgg16_places365_notop_weights.h5'
        print ("places still under construction ..")
        exit(1)
        # print 'loading places weights ..'
        # model.load_weights(weights_path, by_name=True)  # rest layer will be randomly initialized
    elif weights == 'office':
        # weights_path = 'models/weights_vgg2fc256_largeset_11fzlayer_60epoch_sgdlr5e-5m10anneal20epoch_HomeOrOff_model.h5'
        print ("office still under construction ..")
        exit(1)
        # print 'loading office weights ..'
        # model.load_weights(weights_path, by_name=True)
    else:
        print 'NOTE: no weights loaded to the model ..'
        exit(1)

    # compile after loading weights
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(lr=global_learning_rate, momentum=0.9),
                  metrics=['mean_squared_error'])
    return model

def load_imagenet_weights(model_to_be_loaded):
    img_width = 224
    img_height = 224
    img_size = (3, img_width, img_height)
    input_tensor = Input(batch_shape=(None,) + img_size)
    model_vgg_places365_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)

    for layer in model_vgg_places365_notop.layers:
        if layer.get_weights().__len__() > 0:   # not pooling, activation etc.
            layer_name = layer.get_config()['name']     # get_config() returns a dictionary
            # # verify
            # old_weights_body = model_to_be_loaded.get_layer(layer_name+'_body').get_weights()
            model_to_be_loaded.get_layer(layer_name+'_body')\
                .set_weights(model_vgg_places365_notop.get_layer(layer_name).get_weights())
            # old_weights_topf = model_to_be_loaded.get_layer(layer_name+'_topf').get_weights()
            model_to_be_loaded.get_layer(layer_name+'_topf')\
                .set_weights(model_vgg_places365_notop.get_layer(layer_name).get_weights())
            # print('{} ((body and top) x (W and b)): {} {} | {} {}'.format(layer_name,
            #       np.array_equal(model_to_be_loaded.get_layer(layer_name+'_body').get_weights()[0],
            #                      old_weights_body[0]),
            #       np.array_equal(model_to_be_loaded.get_layer(layer_name+'_body').get_weights()[1],
            #                      old_weights_body[1]),
            #       np.array_equal(model_to_be_loaded.get_layer(layer_name+'_topf').get_weights()[0],
            #                      old_weights_topf[0]),
            #       np.array_equal(model_to_be_loaded.get_layer(layer_name+'_topf').get_weights()[1],
            #                      old_weights_topf[1])))
    print 'finished loading imagenet parameters ..'
    return model_to_be_loaded

if __name__ == '__main__':
    img_height = 224
    initial_weights = 'imagenet'
    nb_hidden_node = 2048
    learning_rate = 1e-3        # to conv layers
    lr_multiplier = 1.0         # to top fc layers
    l1_regular = 1e-3           # weight decay in L1 norm
    l2_regular = 1e-3           # L2 norm
    label_scalar = 10           # expend from [0, 1]
    flag_add_bn = True
    flag_add_do = True
    do_ratio = 0.5
    batch_size = 32              # tried 32 (224), 8(448) half of GPU
    nb_epoch = 10                # due to higher dimension of 448 img @ network bottle-neck
    nb_epoch_annealing = 3       # anneal for every <> epochs
    annealing_factor = 0.1
    np.random.seed(7)           # to repeat results
    model_stacked = build_2path_vgg_bodytopf_model(img_height=img_height,
                                                   weights=initial_weights,
                                                   nb_fc_hidden_node=nb_hidden_node,
                                                   dropout_ratio=do_ratio,
                                                   global_learning_rate=learning_rate,
                                                   learning_rate_multiplier=lr_multiplier,
                                                   l1_regularization=l1_regular,
                                                   l2_regularization=l2_regular,
                                                   is_bn_enabled=flag_add_bn,
                                                   is_do_enabled=flag_add_do)
    model_stacked.summary()

    nb_train_sample = 15182
    aug_factor = 10     # 1ep ~= 10ep before
    img_width = 5*img_height

    datagen_train = ImageDataGenerator(rescale=1. / 255)
    # TODO
    generator_train = datagen_train.flow_from_directory(
        'datasets/train_960x1920_20161125/aug_10_times_body_top_concat/',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True,
        class_mode='xy_pos',
        label_file="../../label_list_train1125_15182_aug{}_x{}.csv".format(aug_factor, label_scalar))

    nb_valid_sample = 2000
    datagen_valid = ImageDataGenerator(rescale=1. / 255)
    generator_test = datagen_valid.flow_from_directory(
        'datasets/valid_480x2400_concat_nb2000_20161215/concat/',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False,
        class_mode='xy_pos',
        label_file="../../label_list_valid1215_2000_x{}.csv".format(label_scalar))

    # model training callbacks
    # 1) plot mse graphs
    loss_mse_rtplot = LossMseRTPlot()
    # 2) lr annealing
    annealing_schedule = LearningRateAnnealing(nb_epoch_annealing, annealing_factor)
    # 3) checkpoint saving in case of outage
    saver_filepath = 'model_checkpoints'
    saver_filename = 'model_latest_{epoch:03d}.h5'  # currently < 1000 epochs
    checkpoint_saver = ModelCheckpoint(saver_filepath + os.path.sep + saver_filename,
                                       monitor='val_mean_squared_error',
                                       verbose=1,
                                       save_best_only=False,
                                       save_weights_only=True)

    history_callback = model_stacked.fit_generator(generator_train,
                                                   samples_per_epoch=nb_train_sample*aug_factor,
                                                   nb_epoch=nb_epoch,
                                                   validation_data=generator_test,
                                                   nb_val_samples=nb_valid_sample,
                                                   callbacks=[loss_mse_rtplot, annealing_schedule, checkpoint_saver],
                                                   verbose=1)

    # record
    record = np.column_stack((np.array(history_callback.epoch) + 1,
                              history_callback.history['loss'],
                              history_callback.history['val_loss'],
                              history_callback.history['mean_squared_error'],
                              history_callback.history['val_mean_squared_error']))

    np.savetxt(
        'training_procedure/convergence_input{}_fc{}body_div4topf_{}_1125imgx{}_ls{}_{}epoch_sgdlr{:.0e}m{}ae{}af{}_l1reg{:.0e}l2reg{:.0e}_reloc_model.csv'
            .format(img_height,
                    nb_hidden_node,
                    initial_weights,
                    aug_factor,
                    label_scalar,
                    nb_epoch,
                    learning_rate,
                    int(lr_multiplier),
                    nb_epoch_annealing,  # aepoch
                    annealing_factor,  # afactor
                    l1_regular,
                    l2_regular),
        record, delimiter=',')

    model_stacked.save_weights(
        'models/weights_input{}_fc{}body_div4topf_{}_1125imgx{}_ls{}_{}epoch_sgdlr{:.0e}m{}ae{}af{}_l1reg{:.0e}l2reg{:.0e}_reloc_model.h5'
            .format(img_height,
                    nb_hidden_node,
                    initial_weights,
                    aug_factor,
                    label_scalar,
                    nb_epoch,
                    learning_rate,
                    int(lr_multiplier),
                    nb_epoch_annealing,  # aepoch
                    annealing_factor,  # afactor
                    l1_regular,
                    l2_regular), overwrite=False)
