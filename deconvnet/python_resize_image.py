from PIL import Image
from resizeimage import resizeimage
import matplotlib.pyplot as plt

image_path = 'deconvnet/images/tesla_fat_3ch.png'

img_pil_read = Image.open(image_path)    # original image
img_pil_resized = img_pil_read.resize((224, 224), Image.ANTIALIAS)

img_crop_resized = resizeimage.resize_crop(img_pil_read, [224, 224])
img_cover_resized = resizeimage.resize_cover(img_pil_read, [224, 224])
img_contain_resized = resizeimage.resize_contain(img_pil_read, [224, 224])
img_thumbnail_resized = resizeimage.resize_thumbnail(img_pil_read, [224, 224])


# image show
fig = plt.figure()
ax1 = fig.add_subplot(231)
plt.imshow(img_pil_read)
plt.title('origin')
ax2 = fig.add_subplot(232)
plt.imshow(img_pil_resized)
plt.title('PIL resize')
ax3 = fig.add_subplot(233)
plt.imshow(img_crop_resized)
plt.title('crop')
ax4 = fig.add_subplot(234)
plt.imshow(img_cover_resized)
plt.title('cover')
ax5 = fig.add_subplot(235)
plt.imshow(img_contain_resized)
plt.title('contain')
ax6 = fig.add_subplot(236)
plt.imshow(img_thumbnail_resized)
plt.title('thumbnail')

# image read resize and save
from scipy.misc import imread, imsave, imresize
img_scipy_read = imread(image_path)
img_scipy_resized = imresize(img_scipy_read, (224, 224))
plt.figure()
plt.imshow(img_scipy_resized)
imsave('deconvnet/images/img_scipy_resized.png', img_scipy_resized)     # both jpg and png can be saved
