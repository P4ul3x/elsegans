from __future__ import division
import numpy as np
from imageio import imread, imsave
import scipy.ndimage
from numpy.ma.core import exp
from skimage import io, color
from math import isnan
from glob import glob
import pandas as pd
import os
import random
import sys


def load_image(path, mode='RGBA', blackwhite=False):
    # rgb = imread(path, mode='RGBA').astype(float)[:,:,:3]
    rgb = imread(path, mode=mode).astype(float)[:, :, :3]
    # lab = color.rgb2lab(rgb)

    if blackwhite:
        return rgb
    else:
        return rgb


# 1 if images are equal
# 0 otherwise
def rmse(img_a, img_b, blackwhite=False):
    if not blackwhite:
        return 1 - (np.sqrt(np.sum((img_a - img_b) ** 2)) / np.sqrt(255 ** 2 * len(img_a[0]) * len(img_a[0]) * 3))
    else:
        return 1 - (np.sqrt(np.sum((img_a - img_b) ** 2)) / np.sqrt(img_a.size))


# 1 if images are equal
# 0 otherwise
# -1 if invalid
def norm_cross_correlation(img_a, img_b, blackwhite=False):
    if not blackwhite:
        img_a = img_a / 255
        img_b = img_b / 255

    num = np.sum(img_a * img_b)
    den = np.sqrt(np.sum(img_a ** 2) * np.sum(img_b ** 2))

    try:
        val = num / den
        if isnan(val):
            return -1
        return val
    except:
        return -1


# 1 if images are equal
# 0 otherwise
# -1 if invalid
def zero_mean_norm_cross_correlation(img_a, img_b, blackwhite=False):
    if not blackwhite:
        mean_a = [np.mean(img_a[:, :, 0]), np.mean(img_a[:, :, 1]), np.mean(img_a[:, :, 2])]
        mean_b = [np.mean(img_b[:, :, 0]), np.mean(img_b[:, :, 1]), np.mean(img_b[:, :, 2])]
    if blackwhite:
        mean_a = np.mean(img_a)
        mean_b = np.mean(img_b)

    num = np.sum((img_a - mean_a) * (img_b - mean_b))
    den = np.sqrt(np.sum((img_a - mean_a) ** 2) * np.sum((img_b - mean_b) ** 2))

    try:
        val = ((num / den) + 1) / 2
        if isnan(val):
            return -1
        return val
    except:
        return -1


# @author: Antoine Vacavant, ISIT lab, antoine.vacavant@iut.u-clermont1.fr, http://isit.u-clermont1.fr/~anvacava
def compute_ssim(img_mat_1, img_mat_2):
    # Variables for Gaussian kernel definition

    gaussian_kernel_sigma = 1.5
    gaussian_kernel_width = 11
    gaussian_kernel = np.zeros((gaussian_kernel_width, gaussian_kernel_width))

    # Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i, j] = \
                (1 / (2 * np.pi * (gaussian_kernel_sigma ** 2))) * \
                exp(-(((i - 5) ** 2) + ((j - 5) ** 2)) / (2 * (gaussian_kernel_sigma ** 2)))

            # Convert image matrices to double precision (like in the Matlab version)
    img_mat_1 = img_mat_1.astype(np.float)
    img_mat_2 = img_mat_2.astype(np.float)

    # Squares of input matrices
    img_mat_1_sq = img_mat_1 ** 2
    img_mat_2_sq = img_mat_2 ** 2
    img_mat_12 = img_mat_1 * img_mat_2

    # Means obtained by Gaussian filtering of inputs
    img_mat_mu_1 = scipy.ndimage.filters.convolve(img_mat_1, gaussian_kernel)
    img_mat_mu_2 = scipy.ndimage.filters.convolve(img_mat_2, gaussian_kernel)

    # Squares of means
    img_mat_mu_1_sq = img_mat_mu_1 ** 2
    img_mat_mu_2_sq = img_mat_mu_2 ** 2
    img_mat_mu_12 = img_mat_mu_1 * img_mat_mu_2

    # Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq = scipy.ndimage.filters.convolve(img_mat_1_sq, gaussian_kernel)
    img_mat_sigma_2_sq = scipy.ndimage.filters.convolve(img_mat_2_sq, gaussian_kernel)

    # Covariance
    img_mat_sigma_12 = scipy.ndimage.filters.convolve(img_mat_12, gaussian_kernel)

    # Centered squares of variances
    img_mat_sigma_1_sq = img_mat_sigma_1_sq - img_mat_mu_1_sq
    img_mat_sigma_2_sq = img_mat_sigma_2_sq - img_mat_mu_2_sq
    img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12;

    # c1/c2 constants
    # First use: manual fitting
    c_1 = 6.5025
    c_2 = 58.5225

    # Second use: change k1,k2 & c1,c2 depend on L (width of color map)
    l = 255
    k_1 = 0.01
    c_1 = (k_1 * l) ** 2
    k_2 = 0.03
    c_2 = (k_2 * l) ** 2

    # Numerator of SSIM
    num_ssim = (2 * img_mat_mu_12 + c_1) * (2 * img_mat_sigma_12 + c_2)
    # Denominator of SSIM
    den_ssim = (img_mat_mu_1_sq + img_mat_mu_2_sq + c_1) * \
               (img_mat_sigma_1_sq + img_mat_sigma_2_sq + c_2)
    # SSIM
    ssim_map = num_ssim / den_ssim
    index = np.average(ssim_map)

    return (index + 1) / 2


def euclidean_distance(img_a, img_b, scale_factor=255):
    def reshape(arr):
        width = len(arr)
        height = len(arr[0])

        return np.reshape(arr, -1)

    def dist(i, j):
        return np.abs(i - j)

    def g(i, j, theta=1):
        return (1 / (2 * np.pi * theta ** 2)) * np.exp(-dist(i, j) ** 2 / (2 * theta ** 2))

    im_a = reshape(img_a) / scale_factor
    im_b = reshape(img_b) / scale_factor

    d_imed = 0.0
    for i in range(len(im_a)):
        for j in range(len(im_b)):
            d_imed += g(i, j) * (im_a[i] - im_b[i]) * (im_a[j] - im_b[j])
        print(i, d_imed)

    return np.sqrt(d_imed)


# 1 if images are equal
# 0 otherwise
def ssim(img_a, img_b, blackwhite=False):
    if not blackwhite:
        ch = []
        for channel in range(3):
            ch.append(compute_ssim(img_a[:, :, channel], img_b[:, :, channel]))

        return np.average(ch)
    else:
        return compute_ssim(img_a[:, :, 0], img_b[:, :, 0])


def similarity(metrics, img_a, img_b):
    dic = {}

    im_a = load_image(img_a)
    im_b = load_image(img_b)

    for metric in metrics:
        if metric == 'rmse':
            dic['rmse'] = rmse(im_a, im_b)

        elif metric == 'norm_cross_correlation':
            dic['norm_cross_correlation'] = norm_cross_correlation(im_a, im_b)

        elif metric == 'zero_mean_norm_cross_correlation':
            dic['zero_mean_norm_cross_correlation'] = zero_mean_norm_cross_correlation(im_a, im_b)

        elif metric == 'ssim':
            dic['ssim'] = ssim(im_a, im_b)

    return dic

def paired_similarity(dataset, metric, **argv):
    similarity = 0
    size = dataset.shape[0]
    for i in range(size):
        for j in range(i+1,size):
            similarity += metric(dataset[i],dataset[j],**argv)
    return similarity/(size*(size-1)/2)

def centroid_similarity(dataset, metric, metric_args):
    similarity = 0
    size = dataset.shape[0]
    centroid = np.mean(dataset, axis=0)
    for img in dataset:
        similarity += metric(img, centroid, **metric_args)
    return similarity/size

if __name__ == "__main__":
    print(euclidean_distance(np.array([[0,0],[0,0]]), np.array([[1,1],[1,1]]),1))
    # metrics_color = ['rmse', 'norm_cross_correlation', 'zero_mean_norm_cross_correlation', 'ssim']
    # metrics_blackwhite = ['rmse','norm_cross_correlation','zero_mean_norm_cross_correlation','ssim']
    """
    metrics_blackwhite = ['rmse', 'ssim', 'zero_mean_norm_cross_correlation']

    # dic = similarity(metrics_color, sys.argv[1], sys.argv[2])
    imgs_p = glob('./00058/*/*.png')
    print len(imgs_p)


    first = True
    thresh = 0.95
    output = 'selection-00058-rmse/'
    try:
        os.makedirs(output)
    except Exception:
        pass

    retain = 40
    # final_list = {str(imgs_p[0]) : load_image(imgs_p[0])}

    final_list = [(str(imgs_p[0]), load_image(imgs_p[0]))]

    # print final_list
    print final_list[0][0]

    # exit(0)
    # final_list[imgs_p[0]] = load_image(imgs_p[0])

    imsave('%s/%s.png' % (output, '{:0>4d}'.format(len(final_list))), final_list[0][1])
    # print (final_list)
    # exit(0)

    not_found = True
    random.seed(2015)
    random.shuffle(imgs_p)

    for one in imgs_p:

        # load next image
        img_tmp = load_image(one)

        # check the existing ones..
        for idx in range(len(final_list)):

            other = final_list[idx]
            if one == other[0]:
                print 'same'
                not_found = False
                #continue

            #print '%s -compared- %s' % (one, other[0])

            #print final_list[other].shape

            #print similarity(metrics_blackwhite,one,other[0])
            #dif = zero_mean_norm_cross_correlation(img_tmp, other[1])
            if img_tmp.shape == img_tmp.shape:
                print "hello"

            dif = rmse(img_tmp,other[1])

            # print rmse(img_tmp,other[1])
            print dif

            if dif > thresh:
                not_found = False
                break

        if not_found:
            #print 'gather !'
            print len(final_list)

            imsave('%s/%s.png' % (output, '{:0>4d}'.format(len(final_list))), img_tmp)
            final_list.append((one, img_tmp))

            if retain + 1 == len(final_list):
                print '--- Done ---'
                exit(0)
                break
        not_found = True
        """