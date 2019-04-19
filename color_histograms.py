# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 21:57:09 2019

@author: Bananin
"""

import numpy as np
import pandas as pd
import os
import cv2

# PARAMETERS
n_bins = 128 # bins in each channel's histogram
train_root = "data/train-jpg/"
test_root = "data/test-jpg/"
color_histograms_root = "data/color_histograms/"
# possible image tags
atmospheric = ["clear","partly_cloudy","cloudy","haze"]
common = ["agriculture","bare_ground","cultivation","habitation","primary","road","water"]
rare = ["artisinal_mine","blooming","blow_down","conventional_mine", "selective_logging","slash_burn"]
other_tags = common + rare

# train set labels
labels = pd.read_csv("data/train_v2.csv")
labels.set_index("image_name", inplace=True)

train_paths = sorted([train_root+filename for filename in os.listdir(train_root)])
test_paths = sorted([test_root+filename for filename in os.listdir(test_root)])

X_train = np.zeros((len(train_paths), n_bins*3))
tags_train = np.zeros((len(train_paths), len(other_tags)+1))

X_test = np.zeros((len(test_paths), n_bins*3))

# vectorize training images
for i_img in range(len(train_paths)):
    # append this image's feature vector to X_train
    img = cv2.imread(train_paths[i_img])
    img_features = np.zeros(3*n_bins)
    # all images are RGB
    for channel in range(3):
        # append each channel's histogram to the image's color histogram
        channel_histogram = np.histogram(img[:,:,channel], density=True, bins=n_bins, range=(0,255))[0]
        X_train[i_img, channel*n_bins:(channel+1)*n_bins] = channel_histogram
    
    # fill in the image's tags
    tags = labels.loc[train_paths[i_img].split("/")[-1].split(".")[0],"tags"]
    # which atmospheric tag does this image have?
    for j in range(len(atmospheric)):
         if atmospheric[j] in tags:
             tags_train[i_img,0] = j+1
             break
    # check for other tags
    tags_train[i_img,1:] = [tag in tags for tag in other_tags]
    
    # keep us informed about progress
    if i_img%200 == 0:
        print("Vectorizing train image "+str(i_img+1)+" of "+str(len(train_paths)))

# store the training features and responses
X_train = pd.DataFrame(X_train)
X_train.index = os.listdir(train_root)
X_train.to_csv(color_histograms_root+"X_train.csv")

tags_train = pd.DataFrame(tags_train)
tags_train.index = os.listdir(train_root)
tags_train.to_csv(color_histograms_root+"tags_train.csv")

# vectorize test images
for i_img in range(len(test_paths)):
    # append this image's feature vector to X_test
    img = cv2.imread(test_paths[i_img])
    img_features = np.zeros(3*n_bins)
    # all images are RGB
    for channel in range(3):
        # append each channel's histogram to the image's color histogram
        channel_histogram = np.histogram(img[:,:,channel], density=True, bins=n_bins, range=(0,255))[0]
        X_test[i_img, channel*n_bins:(channel+1)*n_bins] = channel_histogram

    # keep us informed about progress
    if i_img%200 == 0:
        print("Vectorizing test image "+str(i_img+1)+" of "+str(len(test_paths)))
        
# store the test features
X_test = pd.DataFrame(X_test)
X_test.index = os.listdir(test_root)
X_test.to_csv(color_histograms_root+"X_test.csv")