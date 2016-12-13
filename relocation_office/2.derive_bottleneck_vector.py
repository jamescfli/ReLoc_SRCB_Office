__author__ = 'bsl'

from keras.layers import Input, MaxPooling2D, Flatten
from keras.models import Model
from keras.applications import vgg16

from utils.custom_image import ImageDataGenerator
import numpy as np
import os

# prepare small set with 20 images from testing set
nb_sample = 20
candidate_dir = 'datasets/train_test_split_480x1920/test/test_subdir/'
candidate_namelist = sorted(os.listdir(candidate_dir))      # reverse = False
# read x100 label
candidate_label = np.loadtxt('datasets/train_test_split_480x1920/test_label_x100.csv', dtype='float32', delimiter=',')
assert candidate_label.shape[0] == candidate_namelist.__len__(), 'file number != label number'
# selected
selected_namelist = list(candidate_namelist[i] for i in np.arange(nb_sample)*100)
selected_label = candidate_label[np.arange(nb_sample)*100, :]
# save to directory



img_height = 448  # options: 448, 224, and original 450*1920
img_width = img_height * 4


def build_vggrr_model():
    img_size = (3, img_height, img_width)  # expected: shape (nb_sample, 3, 480, 1920)
    input_tensor = Input(batch_shape=(None,) + img_size)

    vgg_office_model_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)
    model_path = 'models/'
    model_weight_filename = 'weights_vgg2fc256_largeset_11fzlayer_60epoch_sgdlr5e-5m10anneal20epoch_HomeOrOff_model.h5'
    vgg_office_model_notop.load_weights(model_path + model_weight_filename, by_name=True)

    vggrr_office_model_output = vgg_office_model_notop.output
    vggrr_office_model_output = MaxPooling2D(pool_size=(1, (img_height/32)*4),
                                             strides=None)(vggrr_office_model_output)
    vggrr_office_model_output = Flatten()(vggrr_office_model_output)    # output dimension 512*img_height/32*1
    vggrr_office_model = Model(vgg_office_model_notop.input, vggrr_office_model_output)

    vggrr_office_model.compile(loss='mean_squared_error',
                               optimizer='RMSprop',     # does not matter
                               metrics=['mean_squared_error'])
    return vggrr_office_model

batch_size = 4  # 8 if feed-forward only, 4 if with back-prop

