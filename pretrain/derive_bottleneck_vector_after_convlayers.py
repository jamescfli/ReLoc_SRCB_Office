__author__ = 'bsl'

from keras.layers import Input, MaxPooling2D, Flatten
from keras.models import Model
from keras.applications import vgg16

from utils.custom_image import ImageDataGenerator
import numpy as np


# use small dataset to make the top layers overfitted
img_height = 224
img_width = 224


# build vgg + places365 trained parameters
def build_vgg_places_model():
    img_size = (3, img_height, img_width)
    input_tensor = Input(batch_shape=(None,) + img_size)
    # load with warning
    vgg_places_model_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)

    vgg_places_model_notop.load_weights('models/vgg16_places365_notop_weights.h5')

    vgg_model_output = vgg_places_model_notop.output
    vgg_model_output = Flatten()(vgg_model_output)
    vgg_model_flatten = Model(vgg_places_model_notop.input, vgg_model_output)

    vgg_model_flatten.compile(loss='categorical_crossentropy',
                              optimizer='adadelta',
                              metrics=['accuracy'])
    return vgg_model_flatten

batch_size = 32


# # bottleneck feature vector for smallset
# nb_sample = 3000*2  # 2 classes
# datagen_train = ImageDataGenerator(rescale=1./255)  # substract 128, check custom_image.py
# generator_train = datagen_train\
#     .flow_from_directory('datasets/data_256_HomeOrOff/test/',
#                          target_size=(img_height, img_width),   # 256^2 -> 224^2
#                          batch_size=batch_size,
#                          shuffle=False,
#                          class_mode=None)
# model = build_vgg_places_model()
# bottleneck_feature_vector = model.predict_generator(generator_train, nb_sample)
# nb_kernel_block5 = 512
# size_feature_map = 7*7
# assert bottleneck_feature_vector.shape == (nb_sample, nb_kernel_block5*size_feature_map),\
#     'shape of bottle neck vector is not (6000, 512*7*7)'
#
# # save 602.1MB file
# np.save(open('bottleneck_data/bottleneck_feature_vgg_places_smallset.npy', 'w'),
#         bottleneck_feature_vector)

# bottleneck feature vector for largeset
nb_sample = 18344+29055  # 2 classes
datagen_train = ImageDataGenerator(rescale=1./255)
generator_train = datagen_train\
    .flow_from_directory('datasets/data_256_HomeOrOff/train/',
                         target_size=(img_height, img_width),   # 256^2 -> 224^2
                         batch_size=batch_size,
                         shuffle=False,
                         class_mode=None)
model = build_vgg_places_model()
bottleneck_feature_vector = model.predict_generator(generator_train, nb_sample)
nb_kernel_block5 = 512
size_feature_map = 7*7
assert bottleneck_feature_vector.shape == (nb_sample, nb_kernel_block5*size_feature_map),\
    'shape of bottle neck vector does not match (47399, 512*7*7)'

# save 4.8GB file
np.save(open('bottleneck_data/bottleneck_feature_vgg_places_largeset.npy', 'w'),
        bottleneck_feature_vector)
