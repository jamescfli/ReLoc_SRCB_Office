__author__ = 'bsl'

from utils.custom_image import ImageDataGenerator
from relocation_office.a_finetune_topconvfc_layers import build_vggrrfc_model

import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt

# derive x-y values for both training and testing set
nb_hidden_node = 2048
do_ratio = 0.5
nb_fzlayer = 11         # 11 block4, 15 block5, 19 top fc
learning_rate = 1e-5    # to conv layers
lr_multiplier = 10.0   # to top fc layers
label_scalar = 100      # expend from [0, 1]
model_stacked = build_vggrrfc_model(nb_fc_hidden_node=nb_hidden_node,
                                    dropout_ratio=do_ratio,
                                    nb_frozen_layer=nb_fzlayer,
                                    global_learning_rate=learning_rate,
                                    learning_rate_multiplier=lr_multiplier,
                                    label_scaling_factor=label_scalar)
model_stacked.load_weights('models/train_vggrr2fc2048_20161125img_11fzlayer_ls100_50epoch_sgdlr1e-5m1_reloc_model.h5')
model_stacked.summary()

img_height = 448
img_width = img_height*4
batch_size = 16
nb_train_sample = 13182
datagen_train = ImageDataGenerator(rescale=1./255)
generator_train = datagen_train.flow_from_directory('datasets/train_test_split_480x1920_20161125/train/train_subdir/',
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    class_mode='xy_pos',
                                                    label_file="../../train_label_x{}.csv".format(label_scalar))
# prepare test data
nb_test_sample = 2000
datagen_test = ImageDataGenerator(rescale=1./255)
# no shuffle
generator_test = datagen_test.flow_from_directory('datasets/train_test_split_480x1920_20161125/test/test_subdir/',
                                                  target_size=(img_height, img_width),
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  class_mode='xy_pos',
                                                  label_file="../../test_label_x{}.csv".format(label_scalar))
train_pos = model_stacked.predict_generator(generator_train)
test_pos = model_stacked.predict_generator(generator_test)
# check
print "train pos xy shape: {}".format(train_pos.shape)
print "test pos xy shape: {}".format(test_pos.shape)

# derive L2 error for both sets
train_error = np.linalg.norm(train_pos, axis=1)     # if axis=0 is the sample index
test_error = np.linalg.norm(test_pos, axis=1)
train_error = train_error*3600/label_scalar
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
plt.xlabel("Error (cm)")
plt.ylabel('CDF')
plt.legend(['Train', 'Test'])
plt.show()
