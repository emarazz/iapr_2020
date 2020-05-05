import skimage.io
import matplotlib.pyplot as plt
import numpy as np


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
