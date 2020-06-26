# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:30:01 2020

@author: Rafael Fricks

"""

import numpy as np
import pandas as pd
import os
import re

rootPath = 'data'

imPaths = []
x_reses = []
y_reses = []
covid_status = []
for root, dirs, files in os.walk(rootPath):
    for f in files:
        if f.endswith(".raw"):
            # print(' ')
            # i = i + 1
            # print(i, ':')
            
            # thisPath = os.path.join(root, f)  
            thisPath = os.path.join(f)  
            
            # read the resolution out
            parts = thisPath.split('_')
            resRead = parts[-1].split('.')
            xres, yres = resRead[0].split('x')
            
            imPaths.append(thisPath)
            x_reses.append(int(xres))
            y_reses.append(int(yres))
            
            if('normal' in thisPath):
                covid_status.append(0)
            else:
                covid_status.append(1)
            
            
# covid_status = len(imPaths)*[1] # Assume these are all COVID images

df = pd.DataFrame(list(zip(imPaths, x_reses, y_reses, covid_status)), columns = ['Image File', 'X_Res', 'Y_Res', 'COVID-19'])
df.to_csv('im_stats.csv')
