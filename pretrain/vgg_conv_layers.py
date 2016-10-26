from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
import h5py


def vgg_conv_layers(weights_path='/home/bsl/.keras/models/vgg16_weights.h5',   # saved para file to .keras/models
                    img_size=(224, 224)):    # default width, height = 224, ImageNet size
    model = Sequential()
    model.add(ZeroPadding2D((1,1), batch_input_shape=(None,3)+img_size))

    # build the rest of the network
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    # layer_dict = dict([(layer.name, layer) for layer in model.layers])  # total 31 layers

    if weights_path:
        # we don't need fc layer, so we load the parameters layer by layer
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we skip the last fc layers
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        f.close()
        print('Model loaded (without FC layers)')

    # return model, layer_dict    # return both the model and its layer names
    return model

if __name__ == "__main__":
    model_vgg_conv = vgg_conv_layers(img_size=(128,128))
    layer_dict = dict([(layer.name, layer) for layer in model_vgg_conv.layers])  # total 31 layers
    print model_vgg_conv.layers     # list of layers
    print layer_dict
