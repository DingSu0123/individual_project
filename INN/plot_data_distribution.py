#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:36:14 2021

@author: suding
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing


def read_data(filename, column):
    
    """
    Parameters
    ----------
    filename : string
        input_csv.
    column : 
        choose which parameter to be visualized.

    Returns
    -------
    data : numpy array
        data visualized.

    """
    
    data = pd.read_csv(filename, usecols = column)
    data = np.array(data)
    
    # log scale the raw data
    data_log = np.log10(data)
    
    # quantile transformation
    data_quantile = preprocessing.quantile_transform(data_log, output_distribution='normal')

    # minmax_normalization 
    minmax = preprocessing.MinMaxScaler()
    data_final = minmax.fit_transform(data_quantile)
    
    #print(data_final.shape)
    
    
    return data_final


def plot_distribution(input, filename, title):   
    
    """
    Parameters
    ----------
    input : numpy array
        parameters visualized.
    filename : string
        define the png name.
    title : string
        title of pngs.

    Returns
    -------
    None.

    """
    
    
    """
    sns.set(style="whitegrid")
    sns.violinplot(input, orient=('v'))
    plt.title(title)
    plt.savefig(filename)
    plt.close('all')
    """
    
    fig,axes=plt.subplots(1,3)  
    sns.distplot(input,bins= 'auto', ax=axes[0])
    sns.distplot(input,hist=False,ax=axes[1])  
    sns.distplot(input,kde=False,ax=axes[2])  
    plt.title(title)
    plt.savefig(filename)
    plt.close('all')

    
# choose output directory
out_dir = "/Users/suding/Desktop/PHAS0077/plot_data_distribution/"
# setup output directory - if it does not exist
os.system('mkdir -p %s' % out_dir)


# visualize all distribution of parameters
filename1 = "model_inputs.csv"
initialDens = read_data(filename1, [1])
title = "initialDens_distribution"
plot_distribution(initialDens, '%sinitialDens.png' % out_dir, title)

initialTemp = read_data(filename1, [2])
title = "initialTemp_distribution"
plot_distribution(initialTemp, '%sinitialTemp.png' % out_dir, title)

NH2 = read_data(filename1, [3])
title = "NH2_distribution"
plot_distribution(NH2, '%sNH2.png' % out_dir, title)

zeta = read_data(filename1, [4])
title = "zeta_distribution"
plot_distribution(zeta, '%szeta.png' % out_dir, title)

radfield = read_data(filename1, [5])
title = "radfield_distribution"
plot_distribution(radfield, '%sradfield.png' % out_dir, title)

Av = read_data(filename1, [6])
title = "Av_distribution"
plot_distribution(Av, '%sAv.png' % out_dir, title)

rout = read_data(filename1, [7])
title = "rout_distribution"
plot_distribution(rout, '%srout.png' % out_dir, title)




