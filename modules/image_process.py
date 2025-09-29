import numpy as np
import cv2




def crop(im, t, b, l, r):
    img = im.copy()
    return img[t:b, l:r]


def resize(im, width, height, interpolate=None):
    img = im.copy()
    if interpolate is None:
        sz = im.shape
        pixels_im = sz[0] * sz[1]
        pixels_resize = width * height
        if pixels_resize > pixels_im:
            interpolate = cv2.INTER_LINEAR
        else:
            interpolate = cv2.INTER_AREA

    return cv2.resize(img, (width, height), interpolation=interpolate)


def fill_boundary_black(im, bt, bb, bl, br):
    img = im.copy()
    img[:bt, :, :] = 0
    img[bb:, :, :] = 0
    img[:, :bl, :] = 0
    img[:, br:, :] = 0
    return img

