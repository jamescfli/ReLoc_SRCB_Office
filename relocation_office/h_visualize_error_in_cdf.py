__author__ = 'bsl'

from utils.custom_image import ImageDataGenerator
from relocation_office.g_2_finetune_topconvbnfc_layers import build_vggrrfc_bn_model

import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt

from utils.timer import Timer

if __name__ == '__main__':
    # derive x-y values for both training and testing set
    nb_hidden_node = 2048
    learning_rate = 1e-3  # to conv layers
    lr_multiplier = 1.0  # to top fc layers
    l1_regular = 1e-3  # weight decay in L1 norm
    l2_regular = 1e-3  # L2 norm
    label_scalar = 100  # expend from [0, 1]
    flag_add_bn = True
    flag_add_do = True
    do_ratio = 0.5

    model_stacked = build_vggrrfc_bn_model(nb_fc_hidden_node=nb_hidden_node,
                                           dropout_ratio=do_ratio,
                                           global_learning_rate=learning_rate,
                                           learning_rate_multiplier=lr_multiplier,
                                           l1_regularization=l1_regular,
                                           l2_regularization=l2_regular,
                                           is_bn_enabled=flag_add_bn,
                                           is_do_enabled=flag_add_do)
    model_path = 'models/'
    weight_filename = 'weights_vggrr2fc2048bn_imagenet_1125imgvshift_ls100_100epoch_sgdlr1e-3m1ae30af0.1_l1reg1e-3l2reg1e-3_reloc_model.h5'
    model_stacked.load_weights(model_path+weight_filename)
    model_stacked.summary()
    print '# of layers: {}'.format(model_stacked.layers.__len__())

    img_height = 224
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
    with Timer('Generate training position xy'):
        train_pos = model_stacked.predict_generator(generator_train,
                                                    val_samples=nb_train_sample)
    with Timer('Generate testing position xy'):
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
