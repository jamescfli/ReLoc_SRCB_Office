__author__ = 'bsl'

from keras.layers import Input, MaxPooling2D, Flatten
from keras.models import Model
from keras.applications import vgg16
from utils.custom_image import ImageDataGenerator
import numpy as np


img_height = 480  # 480 image size is > doubled to 224
img_width = 480 * 4

# build vgg + rotation robust layer, ~143MB GPU mem occupation
def build_vggrr_model():
    img_size = (3, img_height, img_width)  # expected: shape (nb_sample, 3, 480, 1920)
    input_tensor = Input(batch_shape=(None,) + img_size)

    vgg_places_model_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)
    vgg_places_model_notop.load_weights('models/vgg16_places365_notop.h5')
    vggrr_model_output = vgg_places_model_notop.output
    vggrr_model_output = MaxPooling2D(pool_size=(1, 15 * 4), strides=None)(vggrr_model_output)
    vggrr_model_output = Flatten()(vggrr_model_output)    # output dimension 512*15*1 = 7680
    vggrr_model = Model(vgg_places_model_notop.input, vggrr_model_output)
    vggrr_model.compile(loss='mean_squared_error',
                        optimizer='RMSprop',
                        metrics=[])
    return vggrr_model

batch_size = 8  # feed-forward only, no back-prop
nb_sample = 2000
# derive vector from small set, apart from rescaling by 255, we also substract 128 for 0 mean
# check 'custom_image.py' for further detail
datagen_train = ImageDataGenerator(rescale=1./255)
# add subdir 'test' for class_mode=None recognition
generator_train = datagen_train.flow_from_directory('datasets/train_test_split_480x1920/test',
                                                    target_size=(img_height, img_width),    # order checked
                                                    batch_size=batch_size,
                                                    shuffle=False,  # consisted with label sequence
                                                    class_mode=None)
model = build_vggrr_model()
bottleneck_feature_vector = model.predict_generator(generator_train, nb_sample)
nb_kernel_block5 = 512
nb_vgg_pool_layer = 5
assert bottleneck_feature_vector.shape == \
       (nb_sample, nb_kernel_block5*(img_height/(2**nb_vgg_pool_layer))),\
    'shape of bottle neck vector is not (2000, 7680)'

# apply
np.save(open('bottleneck_data/bottleneck_feature_vggrr_smallset.npy', 'w'), bottleneck_feature_vector)
