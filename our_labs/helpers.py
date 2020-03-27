import skimage.io
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage import data, img_as_float
from skimage import exposure




def plot_brain(im, original_size = True, img_size = (6,6)): 
    im_h, im_w = im.shape
    if original_size:
        dpi = plt.rcParams['figure.dpi']
        fig, ax = plt.subplots(1, 1, figsize=(im_w/dpi, im_h/dpi))
    else:
        fig, ax = plt.subplots(1, 1, figsize=img_size) 
    ax.imshow(im, cmap='gray')
    ax.set_title('MRI brain image ({} px, {} px)'.format(im_h, im_w))
    ax.axis('off')
    plt.show()
    return

                               
def set_th(im, up_th = 255, down_th = 0):
    output = np.zeros(im.shape,dtype=int)
    h = output.shape[0]
    w = output.shape[1]
    for y in range(h):
        for x in range(w):
            if im[y,x] > up_th:
                output[y,x] = 0
            elif im[y,x] < down_th:
                output[y,x] = 0
            else:
                output[y,x] = im[y,x]
    return output


### Code from skimage docs

def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf