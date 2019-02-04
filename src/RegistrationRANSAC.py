from __future__ import print_function

import random
import time
import gc

import numpy as np
import tensorflow as tf
from VGG16 import VGG16mo
from utils.utils import *
import cv2
from lap import lapjv
from utils.shape_context import ShapeContext
import matplotlib.pyplot as plt
from itertools import product as product
import os


class CNN_RANSAC(object):
    def __init__(self):
        self.height = 224
        self.width = 224
        self.shape = np.array([224.0, 224.0])

        self.sift_weight = 2.0
        self.cnn_weight = 1.0

        self.max_itr = 200

        self.tolerance = 1e-2
        self.freq = 5  # k in the paper
        self.epsilon = 0.5
        self.omega = 0.5
        self.beta = 2.0
        self.lambd = 0.5

        self.cnnph = tf.placeholder("float", [2, 224, 224, 3])
        self.vgg = VGG16mo()
        self.vgg.build(self.cnnph)
        self.SC = ShapeContext()

    def draw_matches(self, IX_orig, IY_orig, path, SIFT=True):
        # set parameters

        # resize image
        Xscale = 1.0 * np.array(IX_orig.shape[:2]) / self.shape
        Yscale = 1.0 * np.array(IY_orig.shape[:2]) / self.shape
        IX = cv2.resize(IX_orig, (self.height, self.width))
        IY = cv2.resize(IY_orig, (self.height, self.width))

        IX_for_plot = cv2.resize(IX_orig, (self.height, self.width))
        IY_for_plot = cv2.resize(IY_orig, (self.height, self.width))

        # CNN feature
        # propagate the images through VGG16
        IX = np.expand_dims(IX, axis=0)
        IY = np.expand_dims(IY, axis=0)
        cnn_input = np.concatenate((IX, IY), axis=0)
        with tf.Session() as sess:
            feed_dict = {self.cnnph: cnn_input}
            D1, D2, D3 = sess.run([
                self.vgg.pool3, self.vgg.pool4, self.vgg.pool5_1
            ], feed_dict=feed_dict)

        # flatten
        DX1, DY1 = np.reshape(D1[0], [-1, 256]), np.reshape(D1[1], [-1, 256])
        DX2, DY2 = np.reshape(D2[0], [-1, 512]), np.reshape(D2[1], [-1, 512])
        DX3, DY3 = np.reshape(D3[0], [-1, 512]), np.reshape(D3[1], [-1, 512])

        # normalization
        DX1, DY1 = DX1 / np.std(DX1), DY1 / np.std(DY1)
        DX2, DY2 = DX2 / np.std(DX2), DY2 / np.std(DY2)
        DX3, DY3 = DX3 / np.std(DX3), DY3 / np.std(DY3)

        del D1, D2, D3

        # compute feature space distance
        PD1 = pairwise_distance(DX1, DY1)
        PD2 = pd_expand(pairwise_distance(DX2, DY2), 2)
        PD3 = pd_expand(pairwise_distance(DX3, DY3), 4)
        PD = 1.414 * PD1 + PD2 + PD3

        del DX1, DY1, DX2, DY2, DX3, DY3, PD1, PD2, PD3
        if not SIFT:
            #We do brute-force matching by taking the lowest distance for every point.
            brute_force_match = np.array(list(zip(range(len(PD)), PD.argmin(axis=1))))
            brute_force_match = brute_force_match[:int(len(brute_force_match) * 0.25)]
            points_ref = brute_force_match.T[0]
            points_new = brute_force_match.T[1]
            points_ref = (np.array(np.unravel_index(points_ref, (28, 28))).T + 0.5) * (self.height / 28)
            points_new = (np.array(np.unravel_index(points_new, (28, 28))).T + 0.5) * (self.height / 28)
            transform = RANSAC(points_new, points_ref)
        else:
            sift = cv2.xfeatures2d.SIFT_create(sigma=1)
            keypoints_ref, descr_ref = sift.detectAndCompute(IX_for_plot, None)
            keypoints_new, descr_new = sift.detectAndCompute(IY_for_plot, None)
            matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
            matches = matcher.match(descr_new, descr_ref, None)
            matches.sort(key=lambda x: x.distance, reverse=False)
            good_matches = matches[:int(len(matches) * 0.5)]
            points_new = np.zeros((len(good_matches), 2), dtype=np.float32)
            points_ref = np.zeros((len(good_matches), 2), dtype=np.float32)

            for i, match in enumerate(good_matches):
                points_new[i, :] = keypoints_new[match.queryIdx].pt
                points_ref[i, :] = keypoints_ref[match.trainIdx].pt

            transform = RANSAC(points_new, points_ref)


        # (np.array(np.unravel_index(points_new, (28, 28))).T + 0.5) * (224 / 28)
        # seq = np.array([[i, j] for i in range(28) for j in range(28)], dtype='int32')
        #
        # X = np.array(seq, dtype='float32') * 8.0 + 4.0
        # Y = np.array(seq, dtype='float32') * 8.0 + 4.0
        #
        # # normalize
        # X = (X - 112.0) / 224.0
        # Y = (Y - 112.0) / 224.0


        cv2.imwrite(os.path.join(path, "ix.jpg"), IX_for_plot)
        cv2.imwrite(os.path.join(path, "iy.jpg"), IY_for_plot)

        result = cv2.warpAffine(IY_for_plot, transform, (self.height, self.height))
        cv2.imwrite(os.path.join(path, "registered.jpg"), result)


        print("Starting images")

        # IX_with_points = IX_orig.copy()
        # IY_with_points = IY_orig.copy()
        # for i, j in product(range(28), repeat=2):
        #     c_i = int((i + 0.5) * (IX_with_points.shape[0] / 28))
        #     c_j = int((j + 0.5) * (IX_with_points.shape[1] / 28))
        #     cv2.circle(IX_with_points, center=(c_i, c_j), radius=2, color=(255, 0, 0))
        #
        # for i, j in product(range(28), repeat=2):
        #     c_i = int((i + 0.5) * (IY_with_points.shape[0] / 28))
        #     c_j = int((j + 0.5) * (IY_with_points.shape[1] / 28))
        #     cv2.circle(IY_with_points, center=(c_i, c_j), radius=2, color=(0, 0, 255))
        #
        # cv2.imwrite(os.path.join(path, "ix.jpg"), IX_with_points)
        # cv2.imwrite(os.path.join(path, "iy.jpg"), IY_with_points)
        #
        # matched = np.hstack([IX_with_points, IY_with_points])
        # for x, y in C:
        #     if y == 75:
        #         continue
        #     x_row, x_col = np.unravel_index(x, (28, 28))
        #     y_row, y_col = np.unravel_index(y, (28, 28))
        #     c_i_x = int((x_row + 0.5) * (IX_with_points.shape[0] / 28))
        #     c_j_x = int((x_col + 0.5) * (IX_with_points.shape[1] / 28))
        #     c_i_y = int((y_row + 0.5) * (IY_with_points.shape[0] / 28) + IY_with_points.shape[0])
        #     c_j_y = int((y_col + 0.5) * (IY_with_points.shape[1] / 28))
        #     cv2.line(matched, (c_i_x, c_j_x), (c_i_y, c_j_y), (255, 255, 0))
        #
        # cv2.imwrite(os.path.join(path, "matched.jpg"), matched)
        print("DONE")

def RANSAC(points_new, points_ref, max_iterations=70000, inliner_thresh=1):
    best_ic=0
    best_transform=None
    new_for_transform = np.array([points_new])
    for i in range(max_iterations):
        chosen_points = random.sample(range(len(points_new)), 3)
        t = cv2.getAffineTransform(points_new[chosen_points].astype(np.float32), points_ref[chosen_points].astype(np.float32))
        transformed = cv2.transform(new_for_transform, t)[0]
        ic = (np.sqrt(np.sum((points_ref-transformed)**2, axis=1)) < inliner_thresh).sum()
        if ic > best_ic:
            best_ic = ic
            best_transform=t
            print(f"New IC! {ic}")
    return best_transform