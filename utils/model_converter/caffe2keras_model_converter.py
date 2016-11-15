import keras.caffe.convert as convert
# import pprint
# import argparse

"""

    USAGE EXAMPLE
        python caffe2keras.py -load_path 'models/' -prototxt 'train_val_for_keras.prototxt' -caffemodel 'bvlc_googlenet.caffemodel'

"""

# parser = argparse.ArgumentParser(description='Converts a Caffe model to Keras.')
# parser.add_argument('-load_path', type=str,
#                    help='path where both the .prototxt and the .caffemodel files are stored')
# parser.add_argument('-prototxt', type=str,
#                    help='name of the .prototxt file')
# parser.add_argument('-caffemodel', type=str,
#                    help='name of the .caffemodel file')
# parser.add_argument('-store_path', type=str, default='',
#                    help='path to the folder where the Keras model will be stored (default: -load_path).')
# parser.add_argument('-debug', action='store_true', default=0,
# 		   help='use debug mode')
#
# args = parser.parse_args()


# def main(args):

# load_path = 'models/'
# prototxt = 'train_val_for_keras.prototxt'
# caffemodel = 'bvlc_googlenet.caffemodel'
load_path = 'vgg_w_places_train/'
prototxt = 'deploy_vgg16_places365_datalayer.prototxt'
caffemodel = 'vgg16_places365.caffemodel'
store_path = load_path
debug = False

print("Converting model...")
model = convert.caffe_to_keras(load_path+prototxt, load_path+caffemodel, debug=debug)
print("Finished converting model.")
# .. model summary is still dented but at least the model has been loaded without errors

from keras.applications import vgg16
from keras.layers import Input

img_width = 224
img_height = 224
img_size = (3, img_width, img_height)

input_tensor = Input(batch_shape=(None,) + img_size)
model_vgg_places365_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)

# def set_weights(self, weights):
#         Sets the weights of the layer, from Numpy arrays.

# 00 data (InputLayer)                (None, 3, 224, 224)   0
model_vgg_places365_notop.layers[0].get_weights()   # input player
# 01 conv1_1_zeropadding (ZeroPadding2(None, 3, 226, 226)   0           data[0][0]
# 02 conv1_1 (Convolution2D)          (None, 64, 224, 224)  1792        conv1_1_zeropadding[0][0]
model_vgg_places365_notop.layers[1].set_weights(model.layers[2].get_weights())
# 03 relu1_1 (Activation)             (None, 64, 224, 224)  0           conv1_1[0][0]
# 04 conv1_2_zeropadding (ZeroPadding2(None, 64, 226, 226)  0           relu1_1[0][0]
# 05 conv1_2 (Convolution2D)          (None, 64, 224, 224)  36928       conv1_2_zeropadding[0][0]
model_vgg_places365_notop.layers[2].set_weights(model.layers[5].get_weights())
# 06 relu1_2 (Activation)             (None, 64, 224, 224)  0           conv1_2[0][0]
# 07 pool1 (MaxPooling2D)             (None, 64, 112, 112)  0           relu1_2[0][0]
model_vgg_places365_notop.layers[3].get_weights()

# 08 conv2_1_zeropadding (ZeroPadding2(None, 64, 114, 114)  0           pool1[0][0]
# 09 conv2_1 (Convolution2D)          (None, 128, 112, 112) 73856       conv2_1_zeropadding[0][0]
model_vgg_places365_notop.layers[4].set_weights(model.layers[9].get_weights())
# 10 relu2_1 (Activation)             (None, 128, 112, 112) 0           conv2_1[0][0]
# 11 conv2_2_zeropadding (ZeroPadding2(None, 128, 114, 114) 0           relu2_1[0][0]
# 12 conv2_2 (Convolution2D)          (None, 128, 112, 112) 147584      conv2_2_zeropadding[0][0]
model_vgg_places365_notop.layers[5].set_weights(model.layers[12].get_weights())
# 13 relu2_2 (Activation)             (None, 128, 112, 112) 0           conv2_2[0][0]
# 14 pool2 (MaxPooling2D)             (None, 128, 56, 56)   0           relu2_2[0][0]
model_vgg_places365_notop.layers[6].get_weights()

# 15 conv3_1_zeropadding (ZeroPadding2(None, 128, 58, 58)   0           pool2[0][0]
# 16 conv3_1 (Convolution2D)          (None, 256, 56, 56)   295168      conv3_1_zeropadding[0][0]
model_vgg_places365_notop.layers[7].set_weights(model.layers[16].get_weights())
# 17 relu3_1 (Activation)             (None, 256, 56, 56)   0           conv3_1[0][0]
# 18 conv3_2_zeropadding (ZeroPadding2(None, 256, 58, 58)   0           relu3_1[0][0]
# 19 conv3_2 (Convolution2D)          (None, 256, 56, 56)   590080      conv3_2_zeropadding[0][0]
model_vgg_places365_notop.layers[8].set_weights(model.layers[19].get_weights())
# 20 relu3_2 (Activation)             (None, 256, 56, 56)   0           conv3_2[0][0]
# 21 conv3_3_zeropadding (ZeroPadding2(None, 256, 58, 58)   0           relu3_2[0][0]
# 22 conv3_3 (Convolution2D)          (None, 256, 56, 56)   590080      conv3_3_zeropadding[0][0]
model_vgg_places365_notop.layers[9].set_weights(model.layers[22].get_weights())
# 23 relu3_3 (Activation)             (None, 256, 56, 56)   0           conv3_3[0][0]
# 24 pool3 (MaxPooling2D)             (None, 256, 28, 28)   0           relu3_3[0][0]
model_vgg_places365_notop.layers[10].get_weights()

# 25 conv4_1_zeropadding (ZeroPadding2(None, 256, 30, 30)   0           pool3[0][0]
# 26 conv4_1 (Convolution2D)          (None, 512, 28, 28)   1180160     conv4_1_zeropadding[0][0]
model_vgg_places365_notop.layers[11].set_weights(model.layers[26].get_weights())
# 27 relu4_1 (Activation)             (None, 512, 28, 28)   0           conv4_1[0][0]
# 28 conv4_2_zeropadding (ZeroPadding2(None, 512, 30, 30)   0           relu4_1[0][0]
# 29 conv4_2 (Convolution2D)          (None, 512, 28, 28)   2359808     conv4_2_zeropadding[0][0]
model_vgg_places365_notop.layers[12].set_weights(model.layers[29].get_weights())
# 30 relu4_2 (Activation)             (None, 512, 28, 28)   0           conv4_2[0][0]
# 31 conv4_3_zeropadding (ZeroPadding2(None, 512, 30, 30)   0           relu4_2[0][0]
# 32 conv4_3 (Convolution2D)          (None, 512, 28, 28)   2359808     conv4_3_zeropadding[0][0]
model_vgg_places365_notop.layers[13].set_weights(model.layers[32].get_weights())
# 33 relu4_3 (Activation)             (None, 512, 28, 28)   0           conv4_3[0][0]
# 34 pool4 (MaxPooling2D)             (None, 512, 14, 14)   0           relu4_3[0][0]
model_vgg_places365_notop.layers[14].get_weights()

# 35 conv5_1_zeropadding (ZeroPadding2(None, 512, 16, 16)   0           pool4[0][0]
# 36 conv5_1 (Convolution2D)          (None, 512, 14, 14)   2359808     conv5_1_zeropadding[0][0]
model_vgg_places365_notop.layers[15].set_weights(model.layers[36].get_weights())
# 37 relu5_1 (Activation)             (None, 512, 14, 14)   0           conv5_1[0][0]
# 38 conv5_2_zeropadding (ZeroPadding2(None, 512, 16, 16)   0           relu5_1[0][0]
# 39 conv5_2 (Convolution2D)          (None, 512, 14, 14)   2359808     conv5_2_zeropadding[0][0]
model_vgg_places365_notop.layers[16].set_weights(model.layers[39].get_weights())
# 40 relu5_2 (Activation)             (None, 512, 14, 14)   0           conv5_2[0][0]
# 41 conv5_3_zeropadding (ZeroPadding2(None, 512, 16, 16)   0           relu5_2[0][0]
# 42 conv5_3 (Convolution2D)          (None, 512, 14, 14)   2359808     conv5_3_zeropadding[0][0]
model_vgg_places365_notop.layers[17].set_weights(model.layers[42].get_weights())
# 43 relu5_3 (Activation)             (None, 512, 14, 14)   0           conv5_3[0][0]
# 44 pool5 (MaxPooling2D)             (None, 512, 7, 7)     0           relu5_3[0][0]
model_vgg_places365_notop.layers[18].get_weights()

# # Save converted model structure
# print("Storing model...")
# # json_string = model.to_json()
# # open(store_path + 'Keras_model_structure.json', 'w').write(json_string)
# # Save converted model weights
# model.save_weights(store_path + 'vgg16_places365_wtop.h5', overwrite=True)
# print("Finished storing the converted model to "+ store_path)

# # convert model parameters to model_vgg_places365_notop
# model_vgg_places365_notop.load_weights(store_path+'vgg16_places365_wtop.h5', by_name=True)

# verify change to Places365
model_vgg_imagenet_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)
import numpy as np
model_vgg_imagenet_notop.layers[1].get_weights()[0].shape == model_vgg_places365_notop.layers[1].get_weights()[0].shape
np.array_equal(model_vgg_imagenet_notop.layers[1].get_weights()[0],
               model_vgg_places365_notop.layers[1].get_weights()[0])    # should be False

print("Storing model...")
model_vgg_places365_notop.save_weights(store_path+'vgg16_places365_notop.h5', overwrite=True)
print("Finished storing the converted model to "+ store_path)

# verify saved file
model_vgg_places365_notop_loaded = vgg16.VGG16(input_tensor=input_tensor, include_top=False)
model_vgg_places365_notop_loaded.load_weights(store_path+'vgg16_places365_notop.h5')
model_vgg_imagenet_notop.layers[1].get_weights()[0].shape == model_vgg_places365_notop_loaded.layers[1].get_weights()[0].shape
np.array_equal(model_vgg_imagenet_notop.layers[1].get_weights()[0],
               model_vgg_places365_notop_loaded.layers[1].get_weights()[0])    # should be False
model_vgg_places365_notop.layers[1].get_weights()[0].shape == model_vgg_places365_notop_loaded.layers[1].get_weights()[0].shape
np.array_equal(model_vgg_places365_notop.layers[1].get_weights()[0],
               model_vgg_places365_notop_loaded.layers[1].get_weights()[0])    # should be True