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
from keras.utils.visualize_util import plot

import numpy as np


def build_2path_vgg_bodytopf_model(img_height=448,
                                   weights='imagenet',
                                   nb_fc_hidden_layer=2,
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

    # freeze Block 1~5 in VGG
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
    # Block 5
    body_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1_body', trainable=False)(body_path_x)
    body_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2_body', trainable=False)(body_path_x)
    body_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3_body', trainable=False)(body_path_x)
    body_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool_body')(body_path_x)

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
    topf_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1_topf', trainable=False)(topf_path_x)
    topf_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2_topf', trainable=False)(topf_path_x)
    topf_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3_topf', trainable=False)(topf_path_x)
    topf_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool_topf')(topf_path_x)
    # if is_bn_enabled:
    #     # body
    #     body_path_x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv1_body',
    #                                 W_learning_rate_multiplier=learning_rate_multiplier,
    #                                 b_learning_rate_multiplier=learning_rate_multiplier*2,
    #                                 W_regularizer=l1l2(l1=l1_regularization,
    #                                                    l2=l2_regularization)
    #                                 if (l1_regularization > 0) or (l2_regularization > 0)
    #                                 else None,
    #                                 b_regularizer=None)(body_path_x)
    #     body_path_x = BatchNormalization(name='block5_bn1_body')(body_path_x)
    #     body_path_x = Activation('relu', name='block5_act1_body')(body_path_x)
    #     body_path_x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv2_body',
    #                                 W_learning_rate_multiplier=learning_rate_multiplier,
    #                                 b_learning_rate_multiplier=learning_rate_multiplier * 2,
    #                                 W_regularizer=l1l2(l1=l1_regularization,
    #                                                    l2=l2_regularization)
    #                                 if (l1_regularization > 0) or (l2_regularization > 0)
    #                                 else None,
    #                                 b_regularizer=None)(body_path_x)
    #     body_path_x = BatchNormalization(name='block5_bn2_body')(body_path_x)
    #     body_path_x = Activation('relu', name='block5_act2_body')(body_path_x)
    #     body_path_x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv3_body',
    #                                 W_learning_rate_multiplier=learning_rate_multiplier,
    #                                 b_learning_rate_multiplier=learning_rate_multiplier * 2,
    #                                 W_regularizer=l1l2(l1=l1_regularization,
    #                                                    l2=l2_regularization)
    #                                 if (l1_regularization > 0) or (l2_regularization > 0)
    #                                 else None,
    #                                 b_regularizer=None)(body_path_x)
    #     body_path_x = BatchNormalization(name='block5_bn3_body')(body_path_x)
    #     body_path_x = Activation('relu', name='block5_act3_body')(body_path_x)
    #     # top face
    #     topf_path_x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv1_topf',
    #                                 W_learning_rate_multiplier=learning_rate_multiplier,
    #                                 b_learning_rate_multiplier=learning_rate_multiplier * 2,
    #                                 W_regularizer=l1l2(l1=l1_regularization,
    #                                                    l2=l2_regularization)
    #                                 if (l1_regularization > 0) or (l2_regularization > 0)
    #                                 else None,
    #                                 b_regularizer=None)(topf_path_x)
    #     topf_path_x = BatchNormalization(name='block5_bn1_topf')(topf_path_x)
    #     topf_path_x = Activation('relu', name='block5_act1_topf')(topf_path_x)
    #     topf_path_x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv2_topf',
    #                                 W_learning_rate_multiplier=learning_rate_multiplier,
    #                                 b_learning_rate_multiplier=learning_rate_multiplier * 2,
    #                                 W_regularizer=l1l2(l1=l1_regularization,
    #                                                    l2=l2_regularization)
    #                                 if (l1_regularization > 0) or (l2_regularization > 0)
    #                                 else None,
    #                                 b_regularizer=None)(topf_path_x)
    #     topf_path_x = BatchNormalization(name='block5_bn2_topf')(topf_path_x)
    #     topf_path_x = Activation('relu', name='block5_act2_topf')(topf_path_x)
    #     topf_path_x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv3_topf',
    #                                 W_learning_rate_multiplier=learning_rate_multiplier,
    #                                 b_learning_rate_multiplier=learning_rate_multiplier * 2,
    #                                 W_regularizer=l1l2(l1=l1_regularization,
    #                                                    l2=l2_regularization)
    #                                 if (l1_regularization > 0) or (l2_regularization > 0)
    #                                 else None,
    #                                 b_regularizer=None)(topf_path_x)
    #     topf_path_x = BatchNormalization(name='block5_bn3_topf')(topf_path_x)
    #     topf_path_x = Activation('relu', name='block5_act3_topf')(topf_path_x)
    #     # Exception: The name "block1_conv1" is used 2 times in the model. All layer names should be unique.
    # else:
    #     # body
    #     body_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1_body',
    #                                 W_learning_rate_multiplier=learning_rate_multiplier,
    #                                 b_learning_rate_multiplier=learning_rate_multiplier * 2,
    #                                 W_regularizer=l1l2(l1=l1_regularization,
    #                                                    l2=l2_regularization)
    #                                 if (l1_regularization > 0) or (l2_regularization > 0)
    #                                 else None,
    #                                 b_regularizer=None)(body_path_x)
    #     body_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2_body',
    #                                 W_learning_rate_multiplier=learning_rate_multiplier,
    #                                 b_learning_rate_multiplier=learning_rate_multiplier * 2,
    #                                 W_regularizer=l1l2(l1=l1_regularization,
    #                                                    l2=l2_regularization)
    #                                 if (l1_regularization > 0) or (l2_regularization > 0)
    #                                 else None,
    #                                 b_regularizer=None)(body_path_x)
    #     body_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3_body',
    #                                 W_learning_rate_multiplier=learning_rate_multiplier,
    #                                 b_learning_rate_multiplier=learning_rate_multiplier * 2,
    #                                 W_regularizer=l1l2(l1=l1_regularization,
    #                                                    l2=l2_regularization)
    #                                 if (l1_regularization > 0) or (l2_regularization > 0)
    #                                 else None,
    #                                 b_regularizer=None)(body_path_x)
    #     # top face
    #     topf_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1_topf',
    #                                 W_learning_rate_multiplier=learning_rate_multiplier,
    #                                 b_learning_rate_multiplier=learning_rate_multiplier * 2,
    #                                 W_regularizer=l1l2(l1=l1_regularization,
    #                                                    l2=l2_regularization)
    #                                 if (l1_regularization > 0) or (l2_regularization > 0)
    #                                 else None,
    #                                 b_regularizer=None)(topf_path_x)
    #     topf_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2_topf',
    #                                 W_learning_rate_multiplier=learning_rate_multiplier,
    #                                 b_learning_rate_multiplier=learning_rate_multiplier * 2,
    #                                 W_regularizer=l1l2(l1=l1_regularization,
    #                                                    l2=l2_regularization)
    #                                 if (l1_regularization > 0) or (l2_regularization > 0)
    #                                 else None,
    #                                 b_regularizer=None)(topf_path_x)
    #     topf_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3_topf',
    #                                 W_learning_rate_multiplier=learning_rate_multiplier,
    #                                 b_learning_rate_multiplier=learning_rate_multiplier * 2,
    #                                 W_regularizer=l1l2(l1=l1_regularization,
    #                                                    l2=l2_regularization)
    #                                 if (l1_regularization > 0) or (l2_regularization > 0)
    #                                 else None,
    #                                 b_regularizer=None)(topf_path_x)

    # body: add rr pooling
    body_path_x = MaxPooling2D(pool_size=(1, (img_height / (2**5)) * 4),
                               strides=None,
                               name='rr_pool_body')(body_path_x)
    body_path_x = Flatten(name='flatten_body')(body_path_x)     # only one flatten layer so far, no index

    # top face: one max pooling to (1 x 1 x nb_kernels)
    topf_path_x = MaxPooling2D(((img_height / (2**5)), (img_height / (2**5))),
                               strides=None,
                               name='block5_1x1pool_topf')(topf_path_x)
    topf_path_x = Flatten(name='flatten_topf')(topf_path_x)

    body_topf_comb_x = merge([body_path_x, topf_path_x],
                             mode='concat',
                             concat_axis=1,
                             name='concat_body_and_topf')

    # hidden dense layers, default = 2
    for i in np.arange(nb_fc_hidden_layer):
        body_topf_comb_x = Dense(
            nb_fc_hidden_node,
            name='fc_dense{}_body_topf_comb'.format(i+1),
            activation='linear',    # default is linear
            W_learning_rate_multiplier=learning_rate_multiplier,
            b_learning_rate_multiplier=learning_rate_multiplier * 2,
            W_regularizer=l1l2(l1=l1_regularization, l2=l2_regularization)
            if (l1_regularization > 0) or (l2_regularization > 0)
            else None,
            b_regularizer=None)(body_topf_comb_x)
        if is_bn_enabled:
            body_topf_comb_x = BatchNormalization(name='fc_bn{}_body_topf_comb'.format(i+1))(body_topf_comb_x)
        body_topf_comb_x = Activation('relu', name='fc_act{}_body_topf_comb'.format(i+1))(body_topf_comb_x)
        if is_do_enabled:
            body_topf_comb_x = Dropout(dropout_ratio, name='fc_do{}_body_topf_comb'.format(i+1))(body_topf_comb_x)

    x = Dense(2,
              name='fc_dense{}_body_topf_comb'.format(nb_fc_hidden_layer + 1),
              activation='linear',
              W_learning_rate_multiplier=learning_rate_multiplier,
              b_learning_rate_multiplier=learning_rate_multiplier*2,
              W_regularizer=l1l2(l1=l1_regularization, l2=l2_regularization)
              if (l1_regularization > 0) or (l2_regularization > 0)
              else None,
              b_regularizer=None)(body_topf_comb_x)
    inputs = get_source_inputs(img_input)
    model = Model(inputs, x, name='vgg_body_topf_2path_model')

    if weights == 'imagenet':
        print 'loading imagenet weights ..'
        model = _load_imagenet_weights(model)
    elif weights == 'places':
        print ("places still under construction ..")
        model = _load_places365_weights(model)
    elif weights == 'office':
        print ("office still under construction ..")
        exit(1)
    else:
        print 'NOTE: no weights loaded to the model ..'
        exit(1)

    # compile after loading weights
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(lr=global_learning_rate, momentum=0.9),
                  metrics=['mean_squared_error'])
    return model


def _load_imagenet_weights(model_to_be_loaded):
    img_width = 224
    img_height = 224
    img_size = (3, img_width, img_height)
    input_tensor = Input(batch_shape=(None,) + img_size)
    model_vgg_imagenet_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)

    for layer in model_vgg_imagenet_notop.layers:
        if layer.get_weights().__len__() > 0:   # not pooling, activation etc.
            layer_name = layer.get_config()['name']     # get_config() returns a dictionary
            # # verify
            # old_weights_body = model_to_be_loaded.get_layer(layer_name+'_body').get_weights()
            model_to_be_loaded.get_layer(layer_name+'_body')\
                .set_weights(model_vgg_imagenet_notop.get_layer(layer_name).get_weights())
            # old_weights_topf = model_to_be_loaded.get_layer(layer_name+'_topf').get_weights()
            model_to_be_loaded.get_layer(layer_name+'_topf')\
                .set_weights(model_vgg_imagenet_notop.get_layer(layer_name).get_weights())
    print 'finished loading imagenet parameters ..'
    return model_to_be_loaded


def _load_places365_weights(model_to_be_loaded):
    img_width = 224
    img_height = 224
    img_size = (3, img_width, img_height)
    input_tensor = Input(batch_shape=(None,) + img_size)
    model_vgg_places365_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)
    model_vgg_places365_notop.load_weights('relocation_office_rrtop/models/vgg16_places365_notop_weights_20170125.h5', by_name=True)

    for layer in model_vgg_places365_notop.layers:
        if layer.get_weights().__len__() > 0:   # not pooling, activation etc.
            layer_name = layer.get_config()['name']
            model_to_be_loaded.get_layer(layer_name+'_body')\
                .set_weights(model_vgg_places365_notop.get_layer(layer_name).get_weights())
            model_to_be_loaded.get_layer(layer_name+'_topf')\
                .set_weights(model_vgg_places365_notop.get_layer(layer_name).get_weights())
    print 'finished loading places365 parameters ..'
    return model_to_be_loaded


if __name__ == '__main__':
    # build model from scratch
    img_height = 448
    initial_weights = 'imagenet'
    nb_hidden_dense_layer = 2   # nb of hidden fc layers, output dense excluded
    nb_hidden_node = 2048
    learning_rate = 1e-3        # to conv layers
    lr_multiplier = 1.0         # to top fc layers
    l1_regular = 1e-3           # weight decay in L1 norm
    l2_regular = 1e-3           # L2 norm
    label_scalar = 1            # expend from [0, 1]
    flag_add_bn = True
    flag_add_do = True
    do_ratio = 0.5
    batch_size = 16              # tried 32 (224), 8(448) half of GPU
    nb_epoch = 100              # due to higher dimension of 448 img @ network bottle-neck
    nb_epoch_annealing = 30      # anneal for every <> epochs
    annealing_factor = 0.1
    np.random.seed(7)           # to repeat results
    model_stacked = build_2path_vgg_bodytopf_model(img_height=img_height,
                                                   weights=initial_weights,
                                                   nb_fc_hidden_layer=nb_hidden_dense_layer,
                                                   nb_fc_hidden_node=nb_hidden_node,
                                                   dropout_ratio=do_ratio,
                                                   global_learning_rate=learning_rate,
                                                   learning_rate_multiplier=lr_multiplier,
                                                   l1_regularization=l1_regular,
                                                   l2_regularization=l2_regular,
                                                   is_bn_enabled=flag_add_bn,
                                                   is_do_enabled=flag_add_do)
    model_stacked.summary()
    plot(model_stacked,
         './models/model_2path_comb_body_topf_20170331.png',
         show_layer_names=True,
         show_shapes=True)