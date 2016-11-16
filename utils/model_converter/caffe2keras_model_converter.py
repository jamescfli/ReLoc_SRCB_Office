__author__ = 'bsl'

import keras.caffe.convert as convert
from keras.applications import vgg16
from keras.layers import Input


def load_vgg16_notop_from_caffemodel(load_path='../../pretrain/models/'):

    prototxt = 'train_vgg16_places365.prototxt'
    caffemodel = 'vgg16_places365.caffemodel'
    debug = False

    print("Converting model...")
    model = convert.caffe_to_keras(load_path+prototxt, load_path+caffemodel, debug=debug)
    print("Finished converting model.")

    # prepare a vgg model without top layers
    img_width = 224
    img_height = 224
    img_size = (3, img_width, img_height)
    input_tensor = Input(batch_shape=(None,) + img_size)
    model_vgg_places365_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)

    # # set the weights to the following model
    # # note the difference btw the old and new vgg
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

    return model_vgg_places365_notop

if __name__ == "__main__":

    # verify change to Places365
    img_width = 224
    img_height = 224
    img_size = (3, img_width, img_height)
    input_tensor = Input(batch_shape=(None,) + img_size)
    model_vgg_imagenet_notop = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

    model_vgg_places365_notop = load_vgg16_notop_from_caffemodel()
    from compare_model_parameters import equal_model
    print 'places vgg and imagenet vgg are : ' \
          + ('the same' if equal_model(model_vgg_places365_notop, model_vgg_imagenet_notop) else 'different')

