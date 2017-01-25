__author__ = 'bsl'

import keras.caffe.convert as convert
from keras.applications import vgg16
from keras.layers import Input

from utils.timer import Timer


def load_vgg16_notop_from_caffemodel(load_path='models/'):

    prototxt = 'train_vgg16_places365.prototxt'
    caffemodel = 'vgg16_places365.caffemodel'
    debug = False   # display input shape etc. and saved to 'debug.prototxt' if True

    with Timer("Converting caffe model .."):    # 151 secs
        model_loaded_from_caffe = convert.caffe_to_keras(load_path+prototxt, load_path+caffemodel, debug=debug)

    # prepare a vgg model_loaded_from_caffe without top layers
    img_width = 224
    img_height = 224
    img_size = (3, img_width, img_height)
    input_tensor = Input(batch_shape=(None,) + img_size)
    model_vgg_places365_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)

    model_vgg_places365_notop.get_layer('block1_conv1')\
        .set_weights(model_loaded_from_caffe.get_layer('conv1_1').get_weights())
    model_vgg_places365_notop.get_layer('block1_conv2') \
        .set_weights(model_loaded_from_caffe.get_layer('conv1_2').get_weights())
    model_vgg_places365_notop.get_layer('block2_conv1') \
        .set_weights(model_loaded_from_caffe.get_layer('conv2_1').get_weights())
    model_vgg_places365_notop.get_layer('block2_conv2') \
        .set_weights(model_loaded_from_caffe.get_layer('conv2_2').get_weights())
    model_vgg_places365_notop.get_layer('block3_conv1') \
        .set_weights(model_loaded_from_caffe.get_layer('conv3_1').get_weights())
    model_vgg_places365_notop.get_layer('block3_conv2') \
        .set_weights(model_loaded_from_caffe.get_layer('conv3_2').get_weights())
    model_vgg_places365_notop.get_layer('block3_conv3') \
        .set_weights(model_loaded_from_caffe.get_layer('conv3_3').get_weights())
    model_vgg_places365_notop.get_layer('block4_conv1') \
        .set_weights(model_loaded_from_caffe.get_layer('conv4_1').get_weights())
    model_vgg_places365_notop.get_layer('block4_conv2') \
        .set_weights(model_loaded_from_caffe.get_layer('conv4_2').get_weights())
    model_vgg_places365_notop.get_layer('block4_conv3') \
        .set_weights(model_loaded_from_caffe.get_layer('conv4_3').get_weights())
    model_vgg_places365_notop.get_layer('block5_conv1') \
        .set_weights(model_loaded_from_caffe.get_layer('conv5_1').get_weights())
    model_vgg_places365_notop.get_layer('block5_conv2') \
        .set_weights(model_loaded_from_caffe.get_layer('conv5_2').get_weights())
    model_vgg_places365_notop.get_layer('block5_conv3') \
        .set_weights(model_loaded_from_caffe.get_layer('conv5_3').get_weights())

    return model_vgg_places365_notop

if __name__ == "__main__":
    # verify change to Places365, for comparison
    img_width = 224
    img_height = 224
    img_size = (3, img_width, img_height)
    input_tensor = Input(batch_shape=(None,) + img_size)
    model_vgg_imagenet_notop = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # check weight change from imagenet
    model_vgg_places365_notop = load_vgg16_notop_from_caffemodel()
    from utils.model_converter.compare_model_parameters import equal_model
    print 'places365 vgg and imagenet vgg are : ' \
          + ('the same' if equal_model(model_vgg_imagenet_notop, model_vgg_places365_notop) else 'different')

    # compare with old 'places' parameters
    model_vgg_imagenet_notop.load_weights('models/vgg16_places365_notop_weights_20161205.h5')
    print 'places365_20161205 vgg and places365_20170125 vgg are : ' \
          + ('the same' if equal_model(model_vgg_imagenet_notop, model_vgg_places365_notop) else 'different')

    # save model structure as json
    model_vgg_places365_notop_json = model_vgg_places365_notop.to_json()
    with open('models/vgg16_places365_notop_structure_20170125.json', "w") as json_file_model_stacked:
        json_file_model_stacked.write(model_vgg_places365_notop_json)
    # and weights as h5 file
    model_vgg_places365_notop.save_weights('models/vgg16_places365_notop_weights_20170125.h5')
