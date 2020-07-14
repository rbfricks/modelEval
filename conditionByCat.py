# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:11:33 2020

@author: Rafael Fricks
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score

plt.close('all')
imList = pd.read_csv('OutputScores.csv', index_col=0)

# plt.scatter(imList['mAs'], imList['Diameter'], c=imList['Prediction Score'], cmap='jet', vmin=.5, vmax=.8) # This did not show a clear pattern

ranges = list(set(imList['mAs']))
lineStyles = ['-.r', 'b', ':g', '--m']

fig = plt.figure(1)
fig.set_size_inches(6, 6)
plt.clf()
names = []
allThres = []
for i in range(len(ranges)):
    thisSet = imList.loc[imList['mAs']==ranges[i]]
    fpr, tpr, thres = roc_curve(thisSet['COVID-19'].values,thisSet['Prediction Score'].values)
    cat_auc = auc(fpr, tpr)

    plt.ion()        
    plt.plot(fpr, tpr, lineStyles[i])
    
    allThres.append(thres)
    names.append("{:.1f} mAs (AUC = {:.3f})".format(ranges[i], cat_auc))

    
plt.plot([0, 1],[0, 1], ':k', alpha=.5)
plt.title("Receiver Operating Characteristic Curves")
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity') 
plt.legend(names, loc='lower right')


allThres = list(set(np.concatenate(allThres, axis=0)))


preds = np.int32(imList['Prediction Score'] > .69)
finalRepor = classification_report(imList['COVID-19'], preds, target_names=['Healthy','COVID-19'])

print(finalRepor)

Decision = []
for j in range(len(preds)):
    labl = imList['COVID-19'].iloc[j]
    pred = preds[j]

    if((labl==1) & (pred==1)): #TP
        Decision.append('TP')
    elif((labl==0) & (pred==0)): #TN
        Decision.append('TN')
    elif((labl==0) & (pred==1)): #FP
        Decision.append('FP')
    elif((labl==1) & (pred==0)): #FN
        Decision.append('FN')

imList['Decision'] = Decision
sns.catplot(x='mAs', y='Diameter', hue='Decision', data=imList, kind="swarm")
# plt.scatter(imList['mAs'], imList['Diameter'], c=conf_cat, cmap='jet', vmin=-1, vmax=4) # This did not show a clear pattern

    