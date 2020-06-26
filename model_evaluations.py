# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:14:49 2019

@author: Rafael Fricks
"""
import matplotlib

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import numpy as np
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

if sys.platform == 'linux' :
    # set the matplotlib backend so figures can be saved in the background
    matplotlib.use("Agg") 
    
def evaluateModel(model,evalDS,H,NUM_EPOCHS,mlb,name="Default",**kwargs):
    print("[" + name + "] evaluating network...")
    # predIdxs = model.predict(evalDS)

    n_classes = len(mlb.classes_)

    tempLabels = []
    tempPreds = []
    i = 0
    for x,y in evalDS:
        i = i + len(y)
        tempLabels.append(y.numpy())
        
        preds = model.predict(x)
        tempPreds.append(preds)
        
    print(i)
    testLabels = np.concatenate((tempLabels[0],tempLabels[1]))
    testPreds = np.concatenate((tempPreds[0],tempPreds[1]))
    for i in range(2,len(tempLabels)):
        testLabels = np.concatenate((testLabels,tempLabels[i]))
        testPreds = np.concatenate((testPreds,tempPreds[i]))
    
    # print(testLabels.shape)
    # print(predIdxs.shape)
    assert(testLabels.shape==testPreds.shape)
    dash = '-' * 40
    print(dash)
    print('{:<19s}{:<1s}{:>12s}'.format('LABEL','|','AUROC'))
    print(dash)
    aucs = np.zeros((n_classes,1))
    fprs = []
    tprs = []
    for j in range(0,n_classes):
        fpr_mod, tpr_mod, thresholds_mod = roc_curve(np.int32(testLabels[:,j]),testPreds[:,j])
        fprs.append(fpr_mod)
        tprs.append(tpr_mod)
        cat_auc = auc(fpr_mod, tpr_mod)
        aucs[j] = cat_auc
        # cat_auc = roc_auc_score(testLabels[:,j],predIdxs[:,j])
        print('{:<19s}{:<1s}{:>12.4f}'.format(mlb.classes_[j],':',cat_auc))
    print(dash)
    print('{:<19s}{:<1s}{:>12.4f}'.format('AVERAGE',':',np.mean(aucs)))
    print(dash)

    return np.mean(aucs), fprs, tprs
    # print(classification_report(testLabels.argmax(axis=1), predIdxs, 	target_names=lb.classes_))
    
    # print("Confusion Matrix:")
    # print(confusion_matrix(testLabels.argmax(axis=1), predIdxs))
    
    # if(kwargs["make_plots"] == True):
    #     # plot the training loss and accuracy
    #     N = NUM_EPOCHS
        
    #     plt.style.use("ggplot")
    #     plt.figure(1)
    #     plt.clf()
    #     plt.plot(np.arange(0, N), H.history["loss"], label="train")
    #     plt.plot(np.arange(0, N), H.history["val_loss"], label="val")
    #     plt.title("Training Loss on Dataset [" + name + "]")
    #     plt.xlabel("Epoch #")
    #     plt.ylabel("Loss")
    #     plt.legend(loc="upper right")        
    #     plt.savefig(name+"_lossplot.png")
        
    #     plt.style.use("ggplot")
    #     plt.figure(2)  
    #     plt.clf()
    #     plt.plot(np.arange(0, N), H.history["accuracy"], label="train")
    #     plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val")
    #     plt.title("Training Accuracy on Dataset [" + name + "]")
    #     plt.xlabel("Epoch #")
    #     plt.ylabel("Accuracy")
    #     plt.legend(loc="lower right")        
    #     plt.savefig(name+"_accplot.png")
    
def grad_cam(model, image, cls, layer_name, H=512, W=512):
    # Function for calculating GradCAM Prediction Visualizations, given a model and image
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([image]))
        loss = predictions[:, cls]
    
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    
    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
    
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    
    cam = np.ones(output.shape[0: 2], dtype = np.float32)
    
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    
    cam = cv2.resize(cam.numpy(), (H, W))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heatmap = np.abs(heatmap-1)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    # output_image = cv2.addWeighted(cv2.cvtColor(image.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)
   
    return cam