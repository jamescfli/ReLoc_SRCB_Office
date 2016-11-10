import numpy as np
from PIL import Image
from scipy import misc
from skimage.io import imread, imsave, imshow

# image_path = 'deconvnet/images/tesla_3ch.png'
image_path = 'deconvnet/images/tesla_fat_3ch.png'
# image_path = 'deconvnet/images/husky.jpg'

# # convert 4 ch png to 3ch
# x = imread('deconvnet/images/tesla_fat_4ch.png')
# x = x[:, :, 0:3]
# imsave('deconvnet/images/tesla_fat_3ch.png', x)

# Load data and preprocess

img_pil = Image.open(image_path)    # class: PIL.PngImagePlugin.PngImageFile
img_sci = misc.imread(image_path)   # class: np array
img_ski = imread(image_path)        # class: np array

img_array_pil = np.array(img_pil)   # (row, col, channel) = (532, 676, 3 or 4)
img_array_sci = np.array(img_sci)   # (row, col, channel) = (532, 676, 3 or 4)
img_array_ski = np.array(img_ski)   # (row, col, channel) = (532, 676, 3 or 4)
# where img_array_pil == img_array_sci == img_array_ski, order of height and width is opposite to Tensor
# and img_array_sci == img_sci, img_array_ski == img_ski

# test RGB order
img_array_sci = np.array(img_sci)
img_array_sci[:, :, 1] = 0
img_array_sci[:, :, 2] = 0
imsave('deconvnet/results/tesla_fat_ch0red.png', img_array_sci)
img_array_sci = np.array(img_sci)
img_array_sci[:, :, 0] = 0
img_array_sci[:, :, 2] = 0
imsave('deconvnet/results/tesla_fat_ch1green.png', img_array_sci)
img_array_sci = np.array(img_sci)
img_array_sci[:, :, 0] = 0
img_array_sci[:, :, 1] = 0
imsave('deconvnet/results/tesla_fat_ch2blue.png', img_array_sci)

# test whether predictions_248_all and max are the same
img_pre_248_all = Image.open('deconvnet/results/predictions_248_all.png')
img_pre_248_max = Image.open('deconvnet/results/predictions_248_max.png')
np.array_equal(np.array(img_pre_248_all), np.array(img_pre_248_max))    # True, due to one dimension