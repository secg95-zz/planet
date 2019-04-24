# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:28:55 2019

@author: secg9
"""
from sklearn.svm import SVC
import os
import pandas as pd
import numpy as np
import pickle
import random
#reads the training data and traing tags
train = pd.read_csv('./data/color_histograms/X_train.csv', index_col=0)
tags = pd.read_csv('./data/color_histograms/tags_train.csv', index_col=0)
print(train.shape)
trainIndexes = random.sample(range(1,train.shape[0]), int(train.shape[0]*0.8) )
train = train.iloc[trainIndexes,:]
tags = tags.iloc[trainIndexes,:]
print(train.shape)
print(tags.shape)
np.save('./data/models/trainIndexes.npy', trainIndexes)
#this method write a dictionary with all the classes predicted by one of each SVM
#there is one SVM foe each class tag.
SVM = {}
for i in range(0, len(tags.columns)):
    clf = SVC(C=1.0, class_weight=None,  gamma='auto',decision_function_shape='ovr',  kernel='linear',max_iter=-1, probability=True, random_state=None, shrinking=True,tol=0.001, verbose=False)
    clf.fit(np.array(train), np.array(tags.iloc[:,i]))  
    SVM[i] = clf
    
#pickle models
model_file = open("./data/models/"+ "SVM" +".obj", "wb")
pickle.dump(SVM, model_file)
    
