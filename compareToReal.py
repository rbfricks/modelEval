# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:04:27 2020

@author: Rafael Fricks
"""

import os
import numpy as np
import matplotlib.pyplot as plt

datPath = 'SaveOut'

r_fpr = np.load(os.path.join(datPath, 'real_fpr.npy'))
r_tpr = np.load(os.path.join(datPath, 'real_tpr.npy'))
r_auc_val = .751

# insert an origin point for the real average
r_fpr = np.insert(r_fpr, 0, 0, axis=0)
r_tpr = np.insert(r_tpr, 0, 0, axis=0)

s_fpr = np.load(os.path.join(datPath, 'synth_fpr.npy'))
s_tpr = np.load(os.path.join(datPath, 'synth_tpr.npy'))
s_auc_val = .733

fig = plt.figure(2)
fig.set_size_inches(6, 6)
plt.clf()
plt.ion()    

plt.plot(r_fpr, r_tpr, '-.r')
plt.plot(s_fpr, s_tpr, 'b')

plt.plot([0, 1],[0, 1], ':k', alpha=.5)
plt.title("Receiver Operating Characteristic Curves")
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.legend(("Real Data (AUC = {:.3f})".format(r_auc_val), "Simulated (AUC = {:.3f})".format(s_auc_val)), loc='lower right')