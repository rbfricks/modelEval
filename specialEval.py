# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 09:16:51 2020

@author: Rafael Fricks

"""

from __future__ import absolute_import, print_function, division
# from __future__ import generator_stop

import tensorflow as tf

import pickle
import numpy as np
# import cv2
import random
import os, sys
import cv2
import datetime
import sklearn.utils as sk
import pandas as pd
from textwrap import wrap

from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import backend as K

from model_evaluations import *
# from tfRecordFunctions import *

def readRawFileIm(dataDir, thisEntry):
    imLoc = os.path.join(dataDir, thisEntry['Image File'])
    x_dim = thisEntry['X_Res']
    y_dim = thisEntry['Y_Res']
    imFile = open(imLoc, 'rb')
    
    im = np.fromfile(imFile, dtype=np.uint16, count=x_dim*y_dim)
    imOut = np.reshape(im, (y_dim,x_dim))
    imOut = np.float32(imOut)*(1/4095.0)
    imOut = cv2.resize(imOut, (512,512))
    
    imOut = np.stack([imOut, imOut, imOut], axis = -1)
    imOut = np.expand_dims(imOut,0)
    
    imFile.close()
    
    return imOut

def datasetToIms(model1, img, label, name, outPath, count):
    
    prd = model1.predict(img)
    pred = prd[0,0]
    
    img = np.squeeze(img)
    cam = grad_cam(model1, img, 0, 'bn')   
    
    im = img-np.min(img)
    im = im/np.max(im)

    ## Start plot
    fig = plt.figure(50)
    fig.set_size_inches(12, 6)
    plt.clf()
    plt.ion()    
           
    ## Set plot settings
    axs = fig.subplots(1,2)
    plt.rcParams.update({'axes.titlesize' : 12})
    plt.rcParams.update({'font.size' : 12})
    plt.subplots_adjust(wspace=0.2)
    plt.subplots_adjust(hspace=0.2)

    ax = axs.flat[0]

   
            
    ax.imshow(im, cmap=plt.cm.gray)
    if(label==1):
        status = 'POSITIVE'
    else:   
        status = 'NEGATIVE'
            
    # name = names[i].numpy().decode('utf-8')
    ax.set_title("\n".join(wrap(name, 60)))
    ax.set_xlabel('RT-PCR Result: ' + status)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
         
    ax = axs.flat[1]
    # ax.axis('off') 
    ax.imshow(im, cmap=plt.cm.gray)
    ax.imshow(cam, cmap=plt.cm.jet, alpha=.4)
    ax.set_title('Prediction GradCAM')
    ax.set_xlabel('Network Output: {:.3f}'.format(pred))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
            
    # plt.pause(0.1) #this one for image-by-image pause
    plt.savefig(os.path.join(outPath, str(count)+'.tif'), dpi=100)
    plt.close(fig)
            

    return pred

t1 = datetime.datetime.now()

# initialize the number of epochs to train for and batch size
print('Loading network and settings...\n')


imList = pd.read_csv('Info\\im_stats_simulations.csv', index_col=0)

covidMean = np.float32(156.3268545274254/255.0)
covidSTD = np.float32(52.959808775820505/255.0)

xdim = 512
ydim = 512

cy_len = 16

dataDir = 'data'
resultPath = 'images'

# covidInfoPath = os.path.join('Info','CV19_index.csv')
# covidList = pd.read_csv(covidInfoPath, keep_default_na=False,na_values=[], index_col=0)
# n_covid = len(covidList)

# labels = ['COVID-19','Non-Covid Pneumonia']
labels = ['COVID-19']
n_classes = len(labels)
mlb = MultiLabelBinarizer()
mlb.fit([labels])

# Seed
SEED = 187
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

n_images = len(imList)

chunk = 2048

### Generate the glob patterns for finding the TFR files in fold k
# dataGlobs = os.path.join(covPath, '*.tfrecord') #'COV_vol_0.tfrecord'


### Preloading a COVID-19 trained network ###

name = 'COVIX_PathAK2.h5'
model1 = tf.keras.models.load_model(name)

t2 = datetime.datetime.now()
print('Making predictions and generating Grad-CAM images using COVID-19 network...\n')

preds = []
for i in range(n_images):
    thisEntry = imList.iloc[i]

    img = readRawFileIm(dataDir, thisEntry)
    m = np.mean(img)
    s = np.std(img)
    print(i, m, s)
    img = (img - m)/s
    label = thisEntry['COVID-19']
    name = thisEntry['Image File']

    pred = datasetToIms(model1, img, label, name, resultPath, i)
    preds.append(pred)
# pred = model1.predict(thisIm)


# ### Generate Grad-CAM Images ###
# valid_filepath_dataset = tf.data.Dataset.list_files(dataGlobs, seed = SEED)
# dataset = valid_filepath_dataset.interleave(lambda filepath: tf.data.TFRecordDataset(filepath), cycle_length = cy_len, num_parallel_calls = cy_len)
# valDS = dataset.map(lambda x: decode_phase2_single_raw(x, n_classes, normMean, normSTD), num_parallel_calls = cy_len).batch(BS)

# hist1 = datasetToIms(valDS, 'covid_CAM_TopMod', model1, n_covid)
print(' ')

t3 = datetime.datetime.now()
fpr, tpr, thres = roc_curve(imList['COVID-19'].values,preds)
auc_val = auc(fpr,tpr)

# path = 'SaveOut'
# np.save(os.path.join(path, 'Synth_fpr.npy'), fpr)
# np.save(os.path.join(path, 'Synth_tpr.npy'), tpr)
## Start plot
fig = plt.figure(2)
fig.set_size_inches(6, 6)
plt.clf()
plt.ion()    

plt.plot(fpr, tpr)
plt.plot([0, 1],[0, 1], ':k', alpha=.5)
plt.title("Receiver Operating Characteristic (AUROC = {:.4f})".format(auc_val))
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')

print(" ")
print(">>> Runtime Summary <<<")
print("Begin program at:",t1)
print("End program at:",t3)
print("Total runtime:", t3-t1)
print("Model load time:", t2-t1)
print("Grad-CAM Image generation time:", t3-t2)
print(">>> End Summary <<<")
print(" ")
