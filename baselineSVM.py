
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:13:01 2019

@author: secg9
"""
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
from sklearn.metrics import confusion_matrix
trainIndexes = np.load('./data/models/trainIndexes.npy')

#reads the training data and traing tags
train = pd.read_csv('./data/color_histograms/X_train.csv', index_col=0)
tags = pd.read_csv('./data/color_histograms/tags_train.csv', index_col=0)
#exctracts the test set
test = train.drop(train.index[trainIndexes])
test_tags = tags.drop(tags.index[trainIndexes])

#reads the SVM created for each category
model_file = open('./data/models/SVM.obj','rb')
SVM = pickle.load(model_file)
model_file.close()

#calcula las predicciones
#prediccionesTest = np.zeros((len(SVM.keys()),test.shape[0]))   
#print('bandera1')
#for i in range(0, len(tags.columns)):
#    predicciones = SVM[i].predict(test)
#    prediccionesTest[i,:] = predicciones

#file_name = open('predicciones.obj', 'wb')
#pickle.dump(prediccionesTest, file_name) 
#computes F2 measure by class

#pdb.set_trace()

file_name = open('predicciones.obj','rb')
prediccionesTest = pickle.load(file_name)
file_name.close()

F2measureByClass = np.zeros(len(SVM.keys()))

for i in range(1, len(SVM.keys())):
    y_score = SVM[i].decision_function(test)
    precision  = np.zeros(10)
    recall = np.zeros(10)
    temp = np.zeros(test.shape[0])
    for i in  range(1,10):
        max = y_score[y_score.argmax()]
        min = y_score[y_score.argmin()]
        temp[y_score > ( i*((max-min)/10)  + min)] = 1
        matrix = confusion_matrix(test_tags.iloc[:,i],temp)
        recall[i] = (matrix[0,0] / (matrix[0,1] + matrix[0,0]))
        precision[i] = (matrix[0,0] / (matrix[1,0] + matrix[0,0]))
    #precision, recall, thresholds = precision_recall_curve(prediccionesTest[i,:],  y_score)

    plt.plot(recall, precision)
    plt.savefig('PRcurve_model_lr' + 'categoria' + str(i) + '.jpg')
    np.save('./data/models/precision'+ str(i) + '.npy', precision)
    np.save('./data/models/recall' + str(i) + '.npy', recall)
    precision = precision[-9:]
    recall = recall[-9:]
    F2Measure = 5 * (precision*recall)/(4*precision + recall)
    pdb.set_trace()
    F2measureByClass[i]= F2Measure[F2Measure.argmax()]
#prints out F2 measure
print(F2measureByClass)
print(np.mean(F2measureByClass))





    
