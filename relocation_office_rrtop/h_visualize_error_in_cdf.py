__author__ = 'bsl'

from utils.custom_image import ImageDataGenerator
from relocation_office_rrtop.d_build_parallel_model_bodytop import build_2path_vgg_bodytopf_model

import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt

from utils.timer import Timer

if __name__ == '__main__':
    img_height = 224
    initial_weights = 'imagenet'
    nb_hidden_node = 2048  # where fc layer for topf will be divided by 4, i.e. 512
    learning_rate = 1.e-3  # to conv layers
    lr_multiplier = 1.0  # to top fc layers
    l1_regular = 1.e-3  # weight decay in L1 norm
    l2_regular = 1.e-3  # L2 norm
    label_scalar = 10  # expend from [0, 1]
    flag_add_bn = True
    flag_add_do = True
    do_ratio = 0.5
    batch_size = 32  # tried 32 (224)

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
    model_path = 'models/'
    weight_filename = 'weights_input224_fc2048body_div4topf_imagenet_1125imgx10_ls10_10epoch_sgdlr1e-3m1ae3af0.1_l1reg1e-3l2reg1e-3_reloc_model.h5'
    model_stacked.load_weights(model_path+weight_filename)
    model_stacked.summary()

    img_width = img_height*5
    batch_size = 1
    nb_train_sample = 15182
    datagen_train = ImageDataGenerator(rescale=1./255)
    generator_train = datagen_train.flow_from_directory('datasets/train_960x1920_20161125/concat/',
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        class_mode='xy_pos',
                                                        label_file="../../label_list_train1125_15182_x{}.csv".format(label_scalar))
    nb_valid_sample = 2000
    datagen_valid = ImageDataGenerator(rescale=1. / 255)
    generator_valid = datagen_valid.flow_from_directory('datasets/valid_480x2400_concat_nb2000_20161215/concat/',
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        class_mode='xy_pos',
                                                        label_file="../../label_list_valid1215_2000_x{}.csv".format(label_scalar))
    with Timer('Generate training position xy'):
        train_pos = model_stacked.predict_generator(generator_train,
                                                    val_samples=nb_train_sample)
    with Timer('Generate validation position xy'):
        valid_pos = model_stacked.predict_generator(generator_valid,
                                                    val_samples=nb_valid_sample)
    # check
    print "train pos xy shape: {}".format(train_pos.shape)
    print "valid pos xy shape: {}".format(valid_pos.shape)
    np.save(open('predicted_data/train_position_result_w20161125img.npy', 'w'), train_pos)
    np.save(open('predicted_data/valid_position_result_w20161215img.npy', 'w'), valid_pos)

    # load the xy pos
    train_pos = np.load(open('predicted_data/train_position_result_w20161125img.npy', 'r'))
    valid_pos = np.load(open('predicted_data/valid_position_result_w20161215img.npy', 'r'))
    train_pos_gt = np.loadtxt('datasets/label_list_train1125_15182_x{}.csv'
                              .format(label_scalar), dtype='float32', delimiter=',')
    valid_pos_gt = np.loadtxt('datasets/label_list_valid1215_2000_x{}.csv'
                              .format(label_scalar), dtype='float32', delimiter=',')
    # derive L2 error for both sets
    train_error = np.linalg.norm(train_pos - train_pos_gt, ord=2, axis=1)     # if axis=0 is the sample index
    valid_error = np.linalg.norm(valid_pos - valid_pos_gt, ord=2, axis=1)
    train_loss_mse = (train_error**2).mean()
    valid_loss_mse = (valid_error**2).mean()
    train_error = train_error*3600/label_scalar     # resume to cm
    valid_error = valid_error*3600/label_scalar

    # draw pdf and cdf of the error in both sets
    ecdf_train = sm.distributions.ECDF(train_error)
    ecdf_valid = sm.distributions.ECDF(valid_error)

    x_train = np.linspace(0, max(train_error), 500)
    x_valid = np.linspace(0, max(valid_error), 500)
    # pdf =
    cdf_train = ecdf_train(x_train)
    cdf_valid = ecdf_valid(x_valid)
    plt.step(x_train, cdf_train)
    plt.step(x_valid, cdf_valid)
    plt.xlim([0, max([max(train_error), max(valid_error)])])
    plt.xlabel("Error (cm)")
    plt.ylabel('CDF')
    plt.legend(['Train', 'Valid'], loc='lower right')
    plt.show()
