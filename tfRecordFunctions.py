# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 17:40:29 2020

@author: Rafael Fricks

Encodes case dictionaries into protocol buffers, then writes those buffers
into TFRecords for faster runtime access. Based on pipeline by Fakrul Tushar.
"""

import tensorflow as tf
import numpy as np
import os, sys
import cv2
import tensorflow_addons as tfa
import pydicom as pyd
import pandas as pd
import time

### ImageNet Weights ###
# imNetMean = tf.constant((0.485, 0.456, 0.406), dtype=tf.float32)
# imNetSTD = tf.constant((0.229, 0.224, 0.225), dtype=tf.float32)

### Batch Estimated Weights, estimated from 16384 images ###
imNetMean = tf.constant(138.543275/255.0, dtype=tf.float32)
imNetSTD = tf.constant(62.2542865/255.0, dtype=tf.float32)

cxcMean = tf.constant(120.84653148986399/255.0, dtype=tf.float32)
cxcSTD = tf.constant(59.816529680663464/255.0, dtype=tf.float32)

rscMean = tf.constant(124.98440871760249/255.0, dtype=tf.float32)
rscSTD = tf.constant(63.33487522749552/255.0, dtype=tf.float32)

covidMean = tf.constant(156.3268545274254/255.0, dtype=tf.float32)
covidSTD = tf.constant(52.959808775820505/255.0, dtype=tf.float32)

# normMean = cxcMean 
# normSTD = cxcSTD

imSize = (512, 512)

datPath = 'D:\\Data\\ChestX\\' if sys.platform == 'win32' else '/ChestX/images/'

def cropToRange(pix_array, thisEntry):
    
    cropRangeX = np.divide([thisEntry['X1'], thisEntry['X1']+thisEntry['X2']], 100)
    cropRangeY = np.divide([thisEntry['Y1'], thisEntry['Y1']+thisEntry['Y2']], 100)
                
    xrange = np.int32(np.dot(cropRangeX, pix_array.shape[0]))
    yrange = np.int32(np.dot(cropRangeY, pix_array.shape[1]))
    
    xrange[1] = np.min((xrange[1], pix_array.shape[0]-1))
    yrange[1] = np.min((yrange[1], pix_array.shape[1]-1))
                
    imOut = pix_array[yrange[0]:yrange[1],xrange[0]:xrange[1]]
    
    return imOut

# def calcNormStats(folds):
#     print('placeholder')
#     return means, stds

def writeCasesToTFRecord(names, images, labels, fname):
    N = len(names)
    # print('Writing ' + str(N) + ' cases to TF Record....')
    with tf.io.TFRecordWriter(fname) as f:
        for i in range(0,N):
            # print(names[i])
            name = names[i]
            image = images[i]
            label = labels[i]

            img_shape=image.shape

            if(i==0):
                print('Name of first case: ' + name)
            elif(i==N-1):
                print('Name of last case: ' + name)
            # print('Name: ' + name)
            # print('Image Shape: ',img_shape)
            # print('Image type: ',image.dtype)
            # print(' ')
            feature = {'case_name': _bytes_feature(name.encode()),
                       'image':_bytes_feature(image.tostring()),
                       'label':_int64_list(label[0]),
                       'H':_int64_feature(img_shape[0]),
                       'W':_int64_feature(img_shape[1]),
                       'D':_int64_feature(img_shape[2]),
                       }
    
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            f.write(example.SerializeToString())

    f.close()        
    # print(str(N) + ' cases written to TF Record ' + fname + '.')
    # print(' ')

    return

def writeSingleLabelCasesToTFR(names, images, labels, fname):
    N = len(names)
    # print('Writing ' + str(N) + ' cases to TF Record....')
    with tf.io.TFRecordWriter(fname) as f:
        for i in range(0,N):
            # print(names[i])
            name = names[i]
            image = images[i]
            label = labels[i]

            img_shape=image.shape

            if(i==0):
                print('Name of first case: ' + name)
            elif(i==N-1):
                print('Name of last case: ' + name)
            # print('Name: ' + name)
            # print('Image Shape: ',img_shape)
            # print('Image type: ',image.dtype)
            # print(' ')
            feature = {'case_name': _bytes_feature(name.encode()),
                       'image':_bytes_feature(image.tostring()),
                       'label':_int64_feature(label[0]),
                       'H':_int64_feature(img_shape[0]),
                       'W':_int64_feature(img_shape[1]),
                       'D':_int64_feature(img_shape[2]),
                       }
    
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            f.write(example.SerializeToString())

    f.close()        
    # print(str(N) + ' cases written to TF Record ' + fname + '.')
    # print(' ')

    return


def writeDatalistToTFR(datPath,datList,tfrPath,mlb,chunk,dset):
    NUM_TRAIN_IMAGES = len(datList)
    print('Number of cases in ' +dset + ': ' + str(NUM_TRAIN_IMAGES))
    print(' ')
    
    for i in range(0,NUM_TRAIN_IMAGES,chunk):
        # Read all images
        print('i:' + str(i))
        fname =  os.path.join(tfrPath, "ChestX_vol_" + str(i//chunk) + ".tfrecord")
        
        if(i+chunk>NUM_TRAIN_IMAGES):
            chunk = NUM_TRAIN_IMAGES - i
            
        images = []
        labels = []
        names = []
        for j in range(0,chunk):
            # print('j:' + str(j))
            # Record them one 'chunk' at a time
            itr = i+j
            # print(itr)
            imp = os.path.join(datPath, datList['Image Index'][itr])
            # print(imp)
            imOut = cv2.resize(cv2.imread(imp), (512,512))
            # imOut = np.float32(cv2.resize(cv2.imread(imp), (512, 512)))/255.0
            # imOut = np.float32((imOut - imNetMean)/imNetSTD)
            # imOut = (imOut -  138.543275)/62.2542865 #mean, std estimated from first 16384 images
            # print(imOut.shape, imOut.dtype, np.min(imOut), np.max(imOut))
            # print('Mean: ',np.mean(imOut))
            # print('STD: ',np.std(imOut))
            # print(' ')
                		
            # extract the label
            label = mlb.transform([datList['Finding Labels'][itr].split('|')])
            # assert(imOut.shape==ImShape)
            # assert(label.shape==labelShape)
            images.append(imOut)
            labels.append(label)
            names.append(datList['Image Index'][itr])
            
        print('Writing', len(names), 'cases to', fname , '...')
        writeCasesToTFRecord(names, images, labels, fname)
        print(' ')
        # disCase = caseReader_AERTS(AERTS_ImgList[i], AERTS_SegList[i], names, outcomes, target_res = (1,1,1), verbose = True)
        # print(disCase['case_name'])
        # tripleView(disCase, 4)
        # writeCaseToTFRecord(disCase)
        # plt.pause(10)
        # with tf.io.TFRecordWriter(tfr) as f:
        #     f.write(disCase)
    return 0

def writeKFoldToTFR(setFolds, phase2Path, shuffleList, chunk, dset, diseaseName):
    # Store a composite dataset as TFR, divided into folds
    n_folds = len(setFolds)

    for k in range(n_folds):
        kpath = os.path.join(phase2Path, dset, 'K'+str(k))
        # os.mkdir(kpath)
        
        chnk = chunk # this value needs to be refreshed every time it moves on to a new fold
        KSET =  setFolds[k] #pd.DataFrame.append(set1Folds[k], set2Folds[k], ignore_index=True)
        NUM_IMAGES = len(KSET)

        if(shuffleList):   
            KSET = KSET.sample(frac=1).reset_index(drop=True)

        print('Number of cases in ' + dset + "_K" + str(k) + ': ' + str(NUM_IMAGES))
        print(' ')

        for i in range(0,NUM_IMAGES,chnk):
            # Read all images
            # print('i:' + str(i))
            fname =  os.path.join(kpath, dset + "_K" + str(k) + "_vol_" + str(i//chunk) + ".tfrecord")
            
            ### correct for last chunks
            if(i+chnk>NUM_IMAGES):
                chnk = NUM_IMAGES - i
                
            images = []
            labels = []
            names = []
            for j in range(0,chnk):
                # Record them one 'chunk' at a time
                # print('j:' + str(j))
                itr = i+j                
                # print(itr)

                thisEntry = KSET.iloc[itr]
                name = thisEntry['Center']+'_PT:'+str(thisEntry['Patient'])
                
                if('NIH' in thisEntry['Center']):
                    # print('NIH CASE')
                    imOut = cv2.resize(cv2.imread(thisEntry['Image Path']), imSize)
                    
                elif('RSNAP' in thisEntry['Center']):
                    # print('RSNAP CASE')
                    dcmf = pyd.filereader.dcmread(thisEntry['Image Path'])
                    imOut = cv2.resize(dcmf.pixel_array, imSize)

                elif('Beheshti' in thisEntry['Center']):
                    # print('BEHESHTI CASE')
                    dcmf = pyd.filereader.dcmread(thisEntry['Image Path'])

                    imOut = cropToRange(dcmf.pixel_array, thisEntry)
                    
                    imOut = cv2.resize(imOut, imSize)
                    imOut = np.uint8(np.round(np.float32(imOut) * (255/4095)))
                    
                elif('Busto' in thisEntry['Center']):
                    # print('BUSTO CASE')
                    dcmf = pyd.filereader.dcmread(thisEntry['Image Path'])
                    
                    imOut = cropToRange(dcmf.pixel_array, thisEntry)
                    
                    imOut = cv2.resize(imOut, imSize)
                    imOut = np.clip(imOut, 8000, 22000)
                    imOut = np.round(np.float32(imOut-8000) * (255/14000))
                    imOut = np.uint8(np.absolute(imOut - 255))
                    
                elif('RUN04-20200519-CHEST' in thisEntry['Center']):                    
                    # print('RUN04 CASE')
                    dcmf = pyd.filereader.dcmread(thisEntry['Image Path'])
                    
                    imOut = cropToRange(dcmf.pixel_array, thisEntry)
                    
                    imOut = cv2.resize(imOut, imSize)
                    imOut = np.uint8(np.round(np.float32(imOut) * (255/4095)))
                
                # The ImageNet DenseNet expects a 3-channel image, like the NIH set
                if(len(imOut.shape)==2):
                    imOut = np.stack([imOut, imOut, imOut], axis = -1)
                    		
                # extract the label
                # label = np.array((thisEntry['COVID19_Status'], thisEntry['NonCOVID_Status']))
                # label = label.reshape((1,2))
                label = np.array((thisEntry[diseaseName]))
                label = label.reshape((1,))

                
                images.append(imOut)
                labels.append(label)
                names.append(name)
                
            print('Writing', len(labels), 'cases to', fname , '...')
            # print('Mean: ', np.mean(images))
            # print('STD: ', np.std(images))
            writeSingleLabelCasesToTFR(names, images, labels, fname)
            print(' ')

    return 0


def writeCOVID(COVIDLIST, phase2Path, chunk, dset):
    # Store the COVID data as TFR
    NUM_IMAGES = len(COVIDLIST)
    dpath = os.path.join(phase2Path, dset, 'TFR')

    print('Number of cases in ' + dset + ': ' + str(NUM_IMAGES))
    print(' ')

    for i in range(0,NUM_IMAGES,chunk):
        # Read all images
        # print('i:' + str(i))
        fname =  os.path.join(dpath, dset + "_vol_" + str(i//chunk) + ".tfrecord")
            
        ### correct for last chunks
        if(i+chunk>NUM_IMAGES):
            chunk = NUM_IMAGES - i
                
        images = []
        labels = []
        names = []
        for j in range(0,chunk):
            # Record them one 'chunk' at a time
            # print('j:' + str(j))
            itr = i+j                
            # print(itr)
            thisEntry = COVIDLIST.iloc[itr]
            name = thisEntry['Center']+'_PT:'+thisEntry['Patient']    

            
            if('Beheshti' in thisEntry['Center']):
                # print('BEHESHTI CASE')
                dcmf = pyd.filereader.dcmread(thisEntry['Image Path'])
                
                imOut = cropToRange(dcmf.pixel_array, thisEntry)
                
                imOut = cv2.resize(imOut, imSize)
                imOut = np.uint8(np.round(np.float32(imOut) * (255/4095)))
                    
            elif('Busto' in thisEntry['Center']):
                # print('BUSTO CASE')
                dcmf = pyd.filereader.dcmread(thisEntry['Image Path'])
                
                imOut = cropToRange(dcmf.pixel_array, thisEntry)
                
                imOut = cv2.resize(imOut, imSize)
                imOut = np.clip(imOut, 8000, 22000)
                imOut = np.round(np.float32(imOut-8000) * (255/14000))
                imOut = np.uint8(np.absolute(imOut - 255))
                    
            elif('RUN04-20200519-CHEST' in thisEntry['Center']):                    
                # print('RUN04 CASE')
                dcmf = pyd.filereader.dcmread(thisEntry['Image Path'])
                
                imOut = cropToRange(dcmf.pixel_array, thisEntry)
                
                imOut = cv2.resize(imOut, imSize)
                imOut = np.uint8(np.round(np.float32(imOut) * (255/4095)))
                
            # The ImageNet DenseNet expects a 3-channel image, like the NIH set
            if(len(imOut.shape)==2):
                imOut = np.stack([imOut, imOut, imOut], axis = -1)
                    
            # print(imOut.shape, imOut.dtype, np.min(imOut), np.max(imOut), np.mean(imOut))
            # time.sleep(1)
            # print('Mean: ',np.mean(imOut))
            # print('STD: ',np.std(imOut))
            # print(' ')
                    		
            # extract the label
            # label = np.array((thisEntry['COVID19_Status'], thisEntry['NonCOVID_Status']))
            # label = label.reshape((1,2))
            label = np.array((thisEntry['COVID19_Status']))
            label = label.reshape((1,))
                # assert(imOut.shape==ImShape)
                # assert(label.shape==labelShape)
            images.append(imOut)
            labels.append(label)
            names.append(name)
                
        print('Writing', len(labels), 'cases to', fname , '...')
        print('Mean: ', np.mean(images))
        print('STD:  ' ,np.std(images))
        writeCasesToTFRecord(names, images, labels, fname)
        print(' ')

    return 0

@tf.function
# def decode_single_raw(Serialized_example, augr):
def decode_single_raw(Serialized_example):

    feature = {'case_name': tf.io.FixedLenFeature([],tf.string),
               'image':tf.io.FixedLenFeature([],tf.string),
               'label':tf.io.FixedLenFeature([14,],tf.int64),
               'H':tf.io.FixedLenFeature([],tf.int64),
               'W':tf.io.FixedLenFeature([],tf.int64),
               'D':tf.io.FixedLenFeature([],tf.int64),
               }
    examples=tf.io.parse_example(Serialized_example,feature)

    ##Decode_image_as uint8
    img = tf.io.decode_raw(examples['image'], tf.uint8)
    #Decode_mask_as_int32
    ##Subject id is already in bytes format
    name=examples['case_name']
    # tf.print(name)


    img_shape=[examples['H'],examples['W'],examples['D']]
    # tf.print(img_shape)
    #Reshape the image
    image = tf.reshape(img,img_shape)
    image = tf.cast(image, dtype=tf.float32)/255.0
    image = (image-imNetMean)/imNetSTD
    #Because CNN expect(batch,H,W,D,CHANNEL)
    # img=tf.expand_dims(img, axis=-1)
    ###casting_values
    # img=tf.cast(img, tf.float32)
    # mask=tf.cast(mask,tf.int32)
    label=examples['label']

    return image,label

@tf.function
def decode_phase2_single_raw(Serialized_example, n_classes, normMean, normSTD):

    feature = {'case_name': tf.io.FixedLenFeature([],tf.string),
               'image':tf.io.FixedLenFeature([],tf.string),
               'label':tf.io.FixedLenFeature([n_classes,],tf.int64),
               'H':tf.io.FixedLenFeature([],tf.int64),
               'W':tf.io.FixedLenFeature([],tf.int64),
               'D':tf.io.FixedLenFeature([],tf.int64),
               }
    examples=tf.io.parse_example(Serialized_example,feature)

    ##Decode_image_as uint8
    img = tf.io.decode_raw(examples['image'], tf.uint8)
    #Decode_mask_as_int32
    ##Subject id is already in bytes format
    name=examples['case_name']
    # tf.print(name)


    img_shape=[examples['H'],examples['W'],examples['D']]
    # tf.print(img_shape)
    #Reshape the image
    image = tf.reshape(img,img_shape)
    # image = tf.cast(image, dtype=tf.float32)/255.0
    # image = (image-imNetMean)/imNetSTD
    image = tf.cast(image, dtype=tf.float32)/255.0
    image = (image-normMean)/normSTD
    #Because CNN expect(batch,H,W,D,CHANNEL)
    # img=tf.expand_dims(img, axis=-1)
    ###casting_values
    # img=tf.cast(img, tf.float32)
    # mask=tf.cast(mask,tf.int32)
    label=examples['label']

    return image,label

@tf.function
def decode_forShow(Serialized_example):

    feature = {'case_name': tf.io.FixedLenFeature([],tf.string),
               'image':tf.io.FixedLenFeature([],tf.string),
               'label':tf.io.FixedLenFeature([2,],tf.int64),
               'H':tf.io.FixedLenFeature([],tf.int64),
               'W':tf.io.FixedLenFeature([],tf.int64),
               'D':tf.io.FixedLenFeature([],tf.int64),
               }
    examples=tf.io.parse_example(Serialized_example,feature)

    ##Decode_image_as uint8
    img = tf.io.decode_raw(examples['image'], tf.uint8)
    #Decode_mask_as_int32
    ##Subject id is already in bytes format
    name=examples['case_name']
    # tf.print(name)


    img_shape=[examples['H'],examples['W'],examples['D']]
    # tf.print(img_shape)
    #Reshape the image
    image = tf.reshape(img,img_shape)
    image = tf.cast(image, dtype=tf.float32)/255.0
    image = (image-covidMean)/covidSTD
    #Because CNN expect(batch,H,W,D,CHANNEL)
    # img=tf.expand_dims(img, axis=-1)
    ###casting_values
    # img=tf.cast(img, tf.float32)
    # mask=tf.cast(mask,tf.int32)
    label=examples['label']

    return image,label,name

@tf.function
def decode_forShow2(Serialized_example):

    feature = {'case_name': tf.io.FixedLenFeature([],tf.string),
               'image':tf.io.FixedLenFeature([],tf.string),
               'label':tf.io.FixedLenFeature([14,],tf.int64),
               'H':tf.io.FixedLenFeature([],tf.int64),
               'W':tf.io.FixedLenFeature([],tf.int64),
               'D':tf.io.FixedLenFeature([],tf.int64),
               }
    examples=tf.io.parse_example(Serialized_example,feature)

    ##Decode_image_as uint8
    img = tf.io.decode_raw(examples['image'], tf.uint8)
    #Decode_mask_as_int32
    ##Subject id is already in bytes format
    name=examples['case_name']
    # tf.print(name)


    img_shape=[examples['H'],examples['W'],examples['D']]
    # tf.print(img_shape)
    #Reshape the image
    image = tf.reshape(img,img_shape)
    image = tf.cast(image, dtype=tf.float32)/255.0
    image = (image-imNetMean)/imNetSTD
    #Because CNN expect(batch,H,W,D,CHANNEL)
    # img=tf.expand_dims(img, axis=-1)
    ###casting_values
    # img=tf.cast(img, tf.float32)
    # mask=tf.cast(mask,tf.int32)
    label=examples['label']

    return image,label,name


def augmentExample(image, label):
    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    
    image = tf.image.random_flip_left_right(image)

    image, label = tf.cond(choice < 0.65, 
            lambda: pipeLine(image,label),
            lambda: (image,label))

    return image, label

def pipeLine(image, label):
    delt = 52
    angle =  tf.random.uniform(shape=[], minval=-0.140, maxval= 0.140, dtype=tf.float32) #range of +/-~8 degrees, in radians
    
    choice1 = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    choice2 = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    image = tf.cond(choice1 < 0.5, 
                        lambda: tf.image.resize_with_crop_or_pad(image, 512+delt, 512+delt),
                        lambda: image)    

    image = tf.cond(choice2 < 0.5, 
                        lambda: tfa.image.rotate(image, angle, interpolation='BILINEAR'),
                        lambda: image)

    image = tf.cond(choice1 < 0.5, 
                        lambda: tf.image.random_crop(image, size=[512, 512, 3]),
                        lambda: image)
    
    return image, label

# def getDataSet(tfrPath, NUM_EPOCHS, BS, cy_len, SEED):
#     filepath_dataset = tf.data.Dataset.list_files(os.path.join(tfrPath, '*.tfrecord'), seed = SEED)
#     dataset = filepath_dataset.interleave(lambda filepath: tf.data.TFRecordDataset(filepath), cycle_length = cy_len, num_parallel_calls = cy_len)
#     dataset = dataset.map(decode_single_raw, num_parallel_calls = cy_len).repeat(NUM_EPOCHS).batch(BS)
#     yield dataset # should be 'return'

########################-------Fucntions for tf records
# The following functions can be used to convert a value to a type compatible
# with tf.Example.
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# def flow_from_df(dataframe: pd.DataFrame, chunk_size):
#     for start_row in range(0, dataframe.shape[0], chunk_size):
#         end_row  = min(start_row + chunk_size, dataframe.shape[0])
#         yield dataframe.iloc[start_row:end_row, :]
