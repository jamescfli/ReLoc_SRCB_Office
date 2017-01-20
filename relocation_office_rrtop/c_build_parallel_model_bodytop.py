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
from keras.utils.data_utils import get_file
from keras.optimizers import SGD

import numpy as np


TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'


def build_vggrrfc_bn_model(img_height=224,
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
    img_input = Input(batch_shape=(None,) + img_size, name='body_top_concat_input')   # i.e. input_tensor

    body_path_x = Cropping2D(cropping=((0,0), (0,img_height)),  name='cut_body_input')(img_input) # cut right hxh patch
    topf_path_x = Cropping2D(cropping=((0,0), (img_height*4,0)), name='cut_top_input')(img_input) # cut left hx4h patch

    # freeze Block 1~4 in VGG
    # body
    #   block 1
    body_path_x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1', trainable=False)(body_path_x)
    body_path_x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2', trainable=False)(body_path_x)
    body_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(body_path_x)
    #   block 2
    body_path_x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block2_conv1', trainable=False)(body_path_x)
    body_path_x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block2_conv2', trainable=False)(body_path_x)
    body_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', trainable=False)(body_path_x)
    #   block 3
    body_path_x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1', trainable=False)(body_path_x)
    body_path_x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2', trainable=False)(body_path_x)
    body_path_x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3', trainable=False)(body_path_x)
    body_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', trainable=False)(body_path_x)
    #   block 4
    body_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1', trainable=False)(body_path_x)
    body_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2', trainable=False)(body_path_x)
    body_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3', trainable=False)(body_path_x)
    body_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', trainable=False)(body_path_x)

    # top face
    topf_path_x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1', trainable=False)(topf_path_x)
    topf_path_x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2', trainable=False)(topf_path_x)
    topf_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', trainable=False)(topf_path_x)
    #   block 2
    topf_path_x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block2_conv1', trainable=False)(topf_path_x)
    topf_path_x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block2_conv2', trainable=False)(topf_path_x)
    topf_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', trainable=False)(topf_path_x)
    #   block 3
    topf_path_x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1', trainable=False)(topf_path_x)
    topf_path_x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2', trainable=False)(topf_path_x)
    topf_path_x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3', trainable=False)(topf_path_x)
    topf_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', trainable=False)(topf_path_x)
    #   block 4
    topf_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1', trainable=False)(topf_path_x)
    topf_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2', trainable=False)(topf_path_x)
    topf_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3', trainable=False)(topf_path_x)
    topf_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', trainable=False)(topf_path_x)

    # Block 5
    if is_bn_enabled:
        # body
        body_path_x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv1')(body_path_x)
        body_path_x = BatchNormalization(name='block5_bn1')(body_path_x)
        body_path_x = Activation('relu', name='block5_act1')(body_path_x)
        body_path_x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv2')(body_path_x)
        body_path_x = BatchNormalization(name='block5_bn2')(body_path_x)
        body_path_x = Activation('relu', name='block5_act2')(body_path_x)
        body_path_x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv3')(body_path_x)
        body_path_x = BatchNormalization(name='block5_bn3')(body_path_x)
        body_path_x = Activation('relu', name='block5_act3')(body_path_x)
        # top face
        topf_path_x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv1')(topf_path_x)
        topf_path_x = BatchNormalization(name='block5_bn1')(topf_path_x)
        topf_path_x = Activation('relu', name='block5_act1')(topf_path_x)
        topf_path_x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv2')(topf_path_x)
        topf_path_x = BatchNormalization(name='block5_bn2')(topf_path_x)
        topf_path_x = Activation('relu', name='block5_act2')(topf_path_x)
        topf_path_x = Convolution2D(512, 3, 3, border_mode='same', name='block5_conv3')(topf_path_x)
        topf_path_x = BatchNormalization(name='block5_bn3')(topf_path_x)
        topf_path_x = Activation('relu', name='block5_act3')(topf_path_x)
        # TODO check consistency after training for all conv* layers, maybe duplicate names will be the problem
    else:
        # body
        body_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(body_path_x)
        body_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(body_path_x)
        body_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(body_path_x)
        # top face
        topf_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(topf_path_x)
        topf_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(topf_path_x)
        topf_path_x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(topf_path_x)

    # body: add rr pooling
    body_path_x = MaxPooling2D(pool_size=(1, (img_height / (2**4)) * 4), strides=None, name='rr_pool')(body_path_x)
    body_path_x = Flatten(name='flatten_body')(body_path_x)
    body_path_x = Dense(
            nb_fc_hidden_node,
            name='fc_body_dense_1',
            W_learning_rate_multiplier=learning_rate_multiplier,
            b_learning_rate_multiplier=learning_rate_multiplier*2,    # *2 Caffe practice
            W_regularizer=l1l2(l1=l1_regularization, l2=l2_regularization) if (l1_regularization > 0)
                                                                              or (l2_regularization > 0) else None,
            b_regularizer=l1l2(l1=l1_regularization, l2=l2_regularization) if (l1_regularization > 0)
                                                                              or (l2_regularization > 0) else None)\
        (body_path_x)
    if is_bn_enabled:
        body_path_x = BatchNormalization(name='fc_body_bn1')(body_path_x)
    body_path_x = Activation('relu', name='fc_body_act1')(body_path_x)
    if is_do_enabled:
        body_path_x = Dropout(dropout_ratio, name='fc_body_do_1')(body_path_x)
    # top face: normal max pooling
    topf_path_x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(topf_path_x)
    topf_path_x = Flatten(name='flatten_topf')(topf_path_x)
    topf_path_x = Dense(
            nb_fc_hidden_node/4,    # due img size 1x1 rather than 1x4 for top face, reduce dimen of fc by 4 times
            name='fc_topf_dense_1',
            W_learning_rate_multiplier=learning_rate_multiplier,
            b_learning_rate_multiplier=learning_rate_multiplier*2,    # *2 Caffe practice
            W_regularizer=l1l2(l1=l1_regularization, l2=l2_regularization) if (l1_regularization > 0)
                                                                              or (l2_regularization > 0) else None,
            b_regularizer=l1l2(l1=l1_regularization, l2=l2_regularization) if (l1_regularization > 0)
                                                                              or (l2_regularization > 0) else None)\
        (topf_path_x)
    if is_bn_enabled:
        topf_path_x = BatchNormalization(name='fc_topf_bn1')(topf_path_x)
    topf_path_x = Activation('relu', name='fc_topf_act1')(topf_path_x)
    if is_do_enabled:
        topf_path_x = Dropout(dropout_ratio, name='fc_topf_do_1')(topf_path_x)

    body_topf_comb_x = merge([body_path_x, topf_path_x], mode='concat', concat_axis=1)

    x = Dense(2,
              name='fc_comb_dense_2',
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

    # compile after loading weights
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(lr=global_learning_rate, momentum=0.9),
                  metrics=['mean_squared_error'])
    return model


if __name__ == '__main__':
    # build model from scratch
    img_height = 224
    initial_weights = 'imagenet'
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
    model_stacked = build_vggrrfc_bn_model(img_height=img_height,
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