import keras.backend as K
from keras.applications import vgg16, imagenet_utils
from utils.timer import Timer
import numpy as np
import cPickle as pickle

from deconvnet2_utils import find_top9_mean_act
from deconvnet2_utils import get_deconv_images
from deconvnet2_utils import plot_max_activation
from deconvnet2_utils import plot_deconv


class DeconvNet(object):
    """DeconvNet class"""

    def __init__(self, model):
        self.model = model
        list_layers = self.model.layers
        self.lnames = [l.name for l in list_layers]
        assert len(self.lnames) == len(set(self.lnames)), "Non unique layer names"
        # dict of layers indexed by layer name
        self.d_layers = {}
        for l_name, l in zip(self.lnames, list_layers):
            self.d_layers[l_name] = l
        # tensor for function definitions
        self.x = K.T.tensor4('x')   # 4: sample, channel, width, height

    def __getitem__(self, layer_name):
        try:
            return self.d_layers[layer_name]
        except KeyError:
            print "Erroneous layer name"

    def _deconv(self, X, lname, d_switch, feat_map=None):
        o_width, o_height = self[lname].output_shape[-2:]   # output width and height

        # get filter size
        f_width = self[lname].W_shape[2]
        f_height = self[lname].W_shape[3]

        # compute padding needed
        i_width, i_height = X.shape[-2:]
        pad_width = (o_width - i_width + f_width - 1) / 2
        pad_height = (o_height - i_height + f_height - 1) / 2

        assert isinstance(pad_width, int), "Pad width size issue at layer %s" % lname
        assert isinstance(pad_height, int), "Pad height size issue at layer %s" % lname

        # set to zero based on switch values
        X[d_switch[lname]] = 0
        # get activation function
        activation = self[lname].activation
        X = activation(X)
        if feat_map is not None:
            print "Setting other feat map to zero"
            for i in range(X.shape[1]):     # kernel index
                if i != feat_map:
                    X[:,i,:,:] = 0  # delete all other feat weights
            print "Setting non max activations to zero"
            for i in range(X.shape[0]):
                iw, ih = np.unravel_index(X[i, feat_map, :, :].argmax(), X[i, feat_map, :, :].shape)
                m = np.max(X[i, feat_map, :, :])
                X[i, feat_map, :, :] = 0
                X[i, feat_map, iw, ih] = m
        # get filters. no bias for now
        W = self[lname].W
        # transpose filter
        W = W.transpose([1, 0, 2, 3])
        W = W[:, :, ::-1, ::-1]     # deconv layer
        # CUDNN for conv2d
        conv_out = K.T.nnet.conv2d(input=self.x, filters=W, border_mode='valid')
        # add padding to get correct size
        pad = K.function([self.x], K.spatial_2d_padding(self.x, padding=(pad_width, pad_height), dim_ordering="th"))
        X_pad = pad([X])
        # get deconv output
        deconv_func = K.function([self.x], conv_out)
        X_deconv = deconv_func([X_pad])
        assert X_deconv.shape[-2:] == (o_width, o_height), "Deconv output at %s has wrong size" % lname
        return X_deconv

    def _forward_pass(self, X, target_layer):   # would not be imported if starting with "_"
        # for all layers up to the target layer
        # store the max activation in switch
        d_switch = {}
        layer_index = self.lnames.index(target_layer)
        for lname in self.lnames[:layer_index + 1]:
            # get layer output
            inc, out = self[lname].input, self[lname].output
            f = K.function([inc], out)
            X = f([X])
            if "conv" in lname:
                d_switch[lname] = np.where(X <= 0)
        return d_switch

    def _backward_pass(self, X, target_layer, d_switch, feat_map):
        # Run deconv/maxunpooling until input pixel space
        layer_index = self.lnames.index(target_layer)
        # Get the output of the target_layer of interest
        layer_output = K.function([self[self.lnames[0]].input], self[target_layer].output)
        X_outl = layer_output([X])
        # Special case for the starting layer where we may want to switch off some maps / activations
        print "Deconvolving %s..." % target_layer
        if "pool" in target_layer:
            X_maxunp = K.pool.max_pool_2d_same_size(
                self[target_layer].input, self[target_layer].pool_size)
            unpool_func = K.function([self[self.lnames[0]].input], X_maxunp)
            X_outl = unpool_func([X])
            if feat_map is not None:
                for i in range(X_outl.shape[1]):
                    if i != feat_map:
                        X_outl[:,i,:,:] = 0
                for i in range(X_outl.shape[0]):
                    iw, ih = np.unravel_index(
                        X_outl[i,feat_map,:,:].argmax(), X_outl[i,feat_map,:,:].shape)
                    m = np.max(X_outl[i,feat_map,:,:])
                    X_outl[i,feat_map,:,:] = 0
                    X_outl[i,feat_map,iw,ih] = m
        elif "conv" in target_layer:
            X_outl = self._deconv(X_outl, target_layer, d_switch, feat_map=feat_map)
        else:
            raise ValueError(
                "Invalid layer name: %s \n Can only handle maxpool and conv" % target_layer)
        # Iterate over layers (deepest to shallowest)
        for lname in self.lnames[:layer_index][::-1]:
            print "Deconvolving %s..." % lname
            # Unpool, Deconv or do nothing
            if "pool" in lname:
                p1, p2 = self[lname].pool_size
                uppool = K.function([self.x], K.resize_images(self.x, p1, p2, "th"))
                X_outl = uppool([X_outl])
            elif "conv" in lname:
                X_outl = self._deconv(X_outl, lname, d_switch)
            elif "input" in lname:
                break   # already at the input layer
            else:
                raise ValueError(
                    "Invalid layer name: %s \n Can only handle maxpool and conv" % lname)
        return X_outl

    def get_layers(self):
        list_layers = self.model.layers
        list_layers_name = [l.name for l in list_layers]
        return list_layers_name

    def get_deconv(self, X, target_layer, feat_map=None):
        # first make predictions to get feature maps
        self.model.predict(X)
        # forward pass storing switches
        with Timer('Forward pass'):
            d_switch = self._forward_pass(X, target_layer)
        # then deconvolve starting from target layer
        X_out = self._backward_pass(X, target_layer, d_switch, feat_map)
        return X_out

def load_img():
    image_path_list = ['deconvnet/images/ceo_3ch.png',
                       'deconvnet/images/husky.jpg',
                       'deconvnet/images/tesla_fat_3ch.png']
    img_savor = []  # temp numpy array list
    from PIL import Image
    for img_path in image_path_list:
        img = Image.open(img_path)
        img_resize = img.resize((224, 224), Image.ANTIALIAS)
        img_array = np.array(img_resize)
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = img_array.astype(np.float)
        img_savor.append(img_array)
    return imagenet_utils.preprocess_input(np.asarray(img_savor))

if __name__ == "__main__":
    model = vgg16.VGG16(weights='imagenet', include_top=True)
    decnet = DeconvNet(model)      # initialise DeconvNet with model
    # load data
    data = load_img()   # shape (len(list_img), 3, 224, 224)

    # Action 1) get max activation for a selection of feat maps
    get_max_act = False
    if get_max_act:
        d_act_path = 'deconvnet/data/dict_top9_mean_act.pickle'
        d_act = {"block5_conv3": {}, "block4_conv2": {}, "block3_conv3": {}}
        batch_size = 3      # for 3 imgs
        for feat_map in range(10):
            d_act["block5_conv3"][feat_map] = find_top9_mean_act(data, decnet, "block5_conv3", feat_map, batch_size=batch_size)
            d_act["block4_conv2"][feat_map] = find_top9_mean_act(data, decnet, "block4_conv2", feat_map, batch_size=batch_size)
            d_act["block3_conv3"][feat_map] = find_top9_mean_act(data, decnet, "block3_conv3", feat_map, batch_size=batch_size)
            with open(d_act_path, 'w') as f:
                pickle.dump(d_act, f)

    # Action 2) get deconv images of images that maxly activate the feat maps selected in the step above
    deconv_img = False
    if deconv_img:
        d_act_path = 'deconvnet/data/dict_top9_mean_act.pickle'
        d_deconv_path = 'deconvnet/data/dict_top9_deconv.pickle'
        get_deconv_images(d_act_path, d_deconv_path, data, decnet)

    # Action 3) get deconv images of images that maxly activate the feat maps selected in the step above
    plot_deconv_img = False
    if plot_deconv_img:
        d_act_path = 'deconvnet/data/dict_top9_mean_act.pickle'
        d_deconv_path = 'deconvnet/data/dict_top9_deconv.npz'
        target_layer = "block4_conv2"
        plot_max_activation(d_act_path, d_deconv_path, data, target_layer, save=True)

    # Action 4) get deconv images of some images for some feat map
    deconv_specific = True
    if deconv_specific:
        target_layer = "block4_conv2"
        feat_map = 12
        num_img = 4     # <= data.shape[0] if with replace=False
        # img_index = np.random.choice(data.shape[0], num_img, replace=False)
        img_index = np.random.choice(data.shape[0], num_img, replace=True)
        plot_deconv(img_index, data, decnet, target_layer, feat_map, save=True)