__author__ = 'bsl'

from utils.custom_image import ImageDataGenerator
from relocation_office.g_finetune_topconvfc_layers import build_vggrrfc_model

import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt

from utils.timer import Timer

if __name__ == '__main__':
    # derive x-y values for both training and testing set
    nb_hidden_node = 2048
    do_ratio = 0.5
    nb_fzlayer = 24         # all un-trainable layers
    label_scalar = 100      # expend from [0, 1]
    model_stacked = build_vggrrfc_model(nb_fc_hidden_node=nb_hidden_node,
                                        dropout_ratio=do_ratio,
                                        nb_frozen_layer=nb_fzlayer)
    model_path = 'models/'
    weight_filename = 'weights_vggrr2fc2048_20161125img_11fzlayer_ls100_30epoch_sgdlr1e-5m1_l2reg1e-3_reloc_model.h5'
    model_stacked.load_weights(model_path+weight_filename)
    model_stacked.summary()

    img_height = 448
    img_width = img_height*4
    batch_size = 2
    nb_train_sample = 13182
    datagen_train = ImageDataGenerator(rescale=1./255)
    generator_train = datagen_train.flow_from_directory('datasets/train_test_split_480x1920_20161125/train/train_subdir/',
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        class_mode='xy_pos',
                                                        label_file="../../train_label_x{}.csv".format(label_scalar))
    nb_test_sample = 2000
    datagen_test = ImageDataGenerator(rescale=1./255)
    generator_test = datagen_test.flow_from_directory('datasets/test_image_20161215/image_480x1920_2000_for_test/image_480x1920_2000/',
                                                      target_size=(img_height, img_width),
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      class_mode='xy_pos',
                                                      label_file="../../label_list_480x1920_2000_x{}.csv".format(label_scalar))
    with Timer('Generate training position xy'):    # 1185.50 secs
        train_pos = model_stacked.predict_generator(generator_train,
                                                    val_samples=nb_train_sample)
    with Timer('Generate testing position xy'):     # 179.61 secs
        test_pos = model_stacked.predict_generator(generator_test,
                                                   val_samples=nb_test_sample)
    # check
    print "train pos xy shape: {}".format(train_pos.shape)
    print "test pos xy shape: {}".format(test_pos.shape)
    np.save(open('predicted_data/train_position_result_w20161125img.npy', 'w'), train_pos)
    np.save(open('predicted_data/test_position_result_w20161215img.npy', 'w'), test_pos)

    # load the xy pos
    train_pos = np.load(open('predicted_data/train_position_result_w20161125img.npy', 'r'))
    test_pos = np.load(open('predicted_data/test_position_result_w20161215img.npy', 'r'))
    train_pos_gt = np.loadtxt('datasets/train_test_split_480x1920_20161125/train_label_x{}.csv'
                              .format(label_scalar), dtype='float32', delimiter=',')
    test_pos_gt = np.loadtxt('datasets/test_image_20161215/label_list_480x1920_2000_x{}.csv'
                             .format(label_scalar), dtype='float32', delimiter=',')
    # derive L2 error for both sets
    train_error = np.linalg.norm(train_pos - train_pos_gt, ord=2, axis=1)     # if axis=0 is the sample index
    test_error = np.linalg.norm(test_pos - test_pos_gt, ord=2, axis=1)
    train_loss_mse = (train_error**2).mean()
    test_loss_mse = (test_error**2).mean()
    train_error = train_error*3600/label_scalar     # resume to cm
    test_error = test_error*3600/label_scalar

    # draw pdf and cdf of the error in both sets
    ecdf_train = sm.distributions.ECDF(train_error)
    ecdf_test = sm.distributions.ECDF(test_error)

    x_train = np.linspace(0, max(train_error), 500)
    x_test = np.linspace(0, max(test_error), 500)
    # pdf =
    cdf_train = ecdf_train(x_train)
    cdf_test = ecdf_test(x_test)
    plt.step(x_train, cdf_train)
    plt.step(x_test, cdf_test)
    plt.xlim([0, max([max(train_error), max(test_error)])])
    plt.xlabel("Error (cm)")
    plt.ylabel('CDF')
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.show()
