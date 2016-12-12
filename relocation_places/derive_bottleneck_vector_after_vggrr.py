__author__ = 'bsl'

from keras.layers import Input, MaxPooling2D, Flatten
from keras.models import Model
from keras.applications import vgg16

from utils.custom_image import ImageDataGenerator
import numpy as np


img_height = 224  # options: 448, 224, original 450*1920
img_width = img_height * 4

# build vgg + rotation robust layer, ~143MB GPU mem occupation
def build_vggrr_model():
    img_size = (3, img_height, img_width)  # expected: shape (nb_sample, 3, 480, 1920)
    input_tensor = Input(batch_shape=(None,) + img_size)

    vgg_places_model_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)
    # # debug: verify loading weights
    # w_init_weights = vgg_places_model_notop.layers[17].get_weights()[0]     # (512, 512, 3, 3)
    # b_init_weights = vgg_places_model_notop.layers[17].get_weights()[1]     # (512,)

    vgg_places_model_notop.load_weights('models/vgg16_places365_notop.h5')
    # # debug: verify loading weights
    # w_load_weights = vgg_places_model_notop.layers[17].get_weights()[0]
    # b_load_weights = vgg_places_model_notop.layers[17].get_weights()[1]
    # print w_init_weights.shape
    # print w_init_weights.shape == w_load_weights.shape      # True
    # print np.array_equal(w_init_weights, w_load_weights)    # False
    # print b_init_weights.shape == b_load_weights.shape      # True
    # print np.array_equal(b_init_weights, b_load_weights)    # False

    vggrr_model_output = vgg_places_model_notop.output
    vggrr_model_output = MaxPooling2D(pool_size=(1, (img_height/32) * 4), strides=None)(vggrr_model_output)
    vggrr_model_output = Flatten()(vggrr_model_output)    # output dimension 512*img_height/32*1
    vggrr_model = Model(vgg_places_model_notop.input, vggrr_model_output)
    # # debug: vector shape
    # vggrr_model.layers[19].get_output_shape_for((None, 512, 14, 56))
    # vggrr_model.layers[20].get_output_shape_for((None, 512, 14, 1))
    # vggrr_model.layers[20].get_config()

    vggrr_model.compile(loss='mean_squared_error',
                        optimizer='RMSprop',
                        metrics=['mean_squared_error'])
    return vggrr_model

batch_size = 8  # 8 if feed-forward only, 4 if with back-prop
nb_sample = 2000
# derive vector from small set, apart from rescaling by 255, we also substract 128 for 0 mean
# check 'custom_image.py' for further detail
datagen_train = ImageDataGenerator(rescale=1./255)
# add subdir 'test' for class_mode=None recognition
generator_train = datagen_train\
    .flow_from_directory('datasets/train_test_split_480x1920/test/',    # leave folders not images
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
    'shape of bottle neck vector is not (2000, 7168)'

# apply
np.save(open('bottleneck_data/bottleneck_feature_vggrr_smallset_{}x{}.npy'
             .format(img_height, img_width), 'w'), bottleneck_feature_vector)
