__author__ = 'bsl'

from keras.layers import Input, MaxPooling2D, Flatten
from keras.models import Model
from keras.applications import vgg16

from utils.custom_image import ImageDataGenerator

import numpy as np


img_height = 448  # options: 448, 224, and original 450*1920
img_width = img_height * 4


def build_vggrr_model():
    img_size = (3, img_height, img_width)
    input_tensor = Input(batch_shape=(None,) + img_size)

    vgg_office_model_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)
    model_path = 'models/'
    model_weight_filename = 'weights_vgg2fc256_largeset_11fzlayer_60epoch_sgdlr5e-5m10anneal20epoch_HomeOrOff_model.h5'
    vgg_office_model_notop.load_weights(model_path + model_weight_filename, by_name=True)   # drop top layers

    vggrr_office_model_output = vgg_office_model_notop.output
    vggrr_office_model_output = MaxPooling2D(pool_size=(1, (img_height/32)*4),
                                             strides=None)(vggrr_office_model_output)
    vggrr_office_model_output = Flatten()(vggrr_office_model_output)    # output dimension 512*(img_height/32)*1
    vggrr_office_model = Model(vgg_office_model_notop.input, vggrr_office_model_output)

    vggrr_office_model.compile(loss='mean_squared_error',
                               optimizer='RMSprop',     # does not matter
                               metrics=['mean_squared_error'])
    return vggrr_office_model

batch_size = 4
nb_sample = 20  # 4 * 5 batches
datagen_train = ImageDataGenerator(rescale=1./255)
# add subdir 'test' for class_mode=None recognition
generator_train = datagen_train\
    .flow_from_directory('datasets/train_test_split_480x1920/20_image_set/',    # leave folders not images
                         target_size=(img_height, img_width),
                         batch_size=batch_size,
                         shuffle=False,  # consisted with label sequence
                         class_mode=None)
model = build_vggrr_model()
bottleneck_feature_vector = model.predict_generator(generator_train, nb_sample)
nb_kernel_block5 = 512
nb_vgg_pool_layer = 5
assert bottleneck_feature_vector.shape == \
       (nb_sample, nb_kernel_block5*(img_height/(2**nb_vgg_pool_layer))),\
    'shape of bottle neck vector is not (20, 7168)'

# apply
np.save(open('bottleneck_data/bottleneck_feature_vggrr_20image_{}x{}.npy'
             .format(img_height, img_width), 'w'), bottleneck_feature_vector)