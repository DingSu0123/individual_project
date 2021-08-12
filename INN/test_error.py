#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 16:25:30 2021

@author: suding
"""


import os
import pandas as pd
import numpy as np
import statistics
import seaborn as sns
import matplotlib.pyplot as plt


def read_csv(filename, column):
    data_col = pd.read_csv(filename, usecols=[column])
    data_col = data_col.values.tolist()
  
    return data_col

    
def compute_error_list(list_pre, list_raw):
    frac_error = [(abs(abs(true) - abs(pre))/abs(true)) 
                  for true, pre in zip(np.array(list_raw), np.array(list_pre))]
    
    return np.array(frac_error)
    

def plot_distribution(input, percentile1, percentile2, filename, title):   
    
    """
    Parameters
    ----------
    input : numpy array
        error list visualized.
    filename : string
        define the png name.
    title : string
        title of pngs.

    Returns
    -------
    None.

    """
    
    fig,axes=plt.subplots(1,3)  
    sns.distplot(input,bins= 'auto', ax=axes[0])
    sns.distplot(input,hist=False,ax=axes[1])  
    sns.distplot(input,kde=False,ax=axes[2])
    plt.title(title)
    plt.axvline(input.mean(), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(percentile1, color='r', linewidth=0.5)
    plt.axvline(percentile2, color='g', linewidth=0.5)
    plt.savefig(filename)
    plt.close('all')


def plot_prediction(pre, raw, filename, title):
    plt.scatter(raw, pre, c ='b', marker = '.')
    plt.xlabel('real value')
    plt.ylabel('predicted value')
    plt.title(title)
    plt.savefig(filename)
    plt.close('all')
    


# input the csv generated from INN_model
filename1 = 'pre_co.csv'
filename2 = 'raw_co.csv'

# choose output directory
out_dir = "/Users/suding/Desktop/PHAS0077/plot_error_distribution/"
os.system('mkdir -p %s' % out_dir)



# plot error for initialDens and compute some statistics
column1 = 'initialDens'
initialDens_pre = read_csv(filename1, column1)
initialDens_raw = read_csv(filename2, column1)
plot_prediction(initialDens_pre, initialDens_raw, '%sscatter_initialDens.png' % out_dir, 'scatter plot for initialDens')
fractional_error_initialDens = compute_error_list(initialDens_pre, initialDens_raw)

# compute mean error for initialDens
initialDens_error_mean = sum(fractional_error_initialDens)/len(fractional_error_initialDens)
# compute 16th and 83rd percentiles
initialDens_percentiles_16th = np.percentile(fractional_error_initialDens,16)
initialDens_percentiles_83rd = np.percentile(fractional_error_initialDens, 83)
plot_distribution(fractional_error_initialDens, initialDens_percentiles_16th,
                   initialDens_percentiles_83rd, '%sinitialDens.png' % out_dir 
                  ,'fractional error distribution for initialDens')

print('the mean fractional error of initialDens is', initialDens_error_mean,' ',
      'the corresponding 16th pencentile is', initialDens_percentiles_16th,' ',
      'the corresponding 83rd pencentile is', initialDens_percentiles_83rd)



# plot error for initialTemp and compute some statistics
column2 = 'initialTemp'
initialTemp_pre = read_csv(filename1, column2)
initialTemp_raw = read_csv(filename2, column2)
plot_prediction(initialTemp_pre, initialTemp_raw, '%sscatter_initialTemp.png' % out_dir, 'scatter plot for initialTemp')
fractional_error_initialTemp = compute_error_list(initialTemp_pre, initialTemp_raw)

# compute mean error for initialTemp
initialTemp_error_mean = sum(fractional_error_initialTemp)/len(fractional_error_initialTemp)
# compute 16th and 83rd percentiles
initialTemp_percentiles_16th = np.percentile(fractional_error_initialTemp, 16)
initialTemp_percentiles_83rd = np.percentile(fractional_error_initialTemp, 83)

plot_distribution(fractional_error_initialTemp, initialTemp_percentiles_16th,
                  initialTemp_percentiles_83rd, '%sinitialTemp.png' % out_dir, 
                  'fractional error distribution for initialTemp')

print('the mean fractional error of initialTemp is', initialTemp_error_mean,' ',
      'the corresponding 16th pencentile is', initialTemp_percentiles_16th,' ',
      'the corresponding 83rd pencentile is', initialTemp_percentiles_83rd)



# plot error for NH2 and compute some statistics
column3 = 'NH2'
NH2_pre = read_csv(filename1, column3)
NH2_raw = read_csv(filename2, column3)
plot_prediction(NH2_pre, NH2_raw, '%sscatter_NH2.png' % out_dir, 
                'scatter plot for NH2')
fractional_error_NH2 = compute_error_list(NH2_pre, NH2_raw)

# compute mean error for NH2
NH2_error_mean = sum(fractional_error_NH2)/len(fractional_error_NH2)
# compute 16th and 83rd percentiles
NH2_percentiles_16th = np.percentile(fractional_error_NH2, 16)
NH2_percentiles_83rd = np.percentile(fractional_error_NH2, 83)

plot_distribution(fractional_error_NH2, NH2_percentiles_16th,
                  NH2_percentiles_83rd, '%sNH2.png' % out_dir 
                  ,'fractional error distribution for NH2')

print('the mean fractional error of NH2 is', NH2_error_mean,' ',
      'the corresponding 16th pencentile is', NH2_percentiles_16th,' ',
      'the corresponding 83rd pencentile is', NH2_percentiles_83rd)



# plot error for zeta and compute some statistics
column4 = 'zeta'
zeta_pre = read_csv(filename1, column4)
zeta_raw = read_csv(filename2, column4)
plot_prediction(zeta_pre, zeta_raw, '%sscatter_zeta.png' % out_dir, 
                'scatter plot for zeta')

fractional_error_zeta = compute_error_list(zeta_pre, zeta_raw)

# compute mean error for zeta
zeta_error_mean = sum(fractional_error_zeta)/len(fractional_error_zeta)
# compute 16th and 83rd percentiles
zeta_percentiles_16th = np.percentile(fractional_error_zeta, 16)
zeta_percentiles_83rd = np.percentile(fractional_error_zeta, 83)

plot_distribution(fractional_error_zeta, zeta_percentiles_16th,
                  zeta_percentiles_83rd, '%szeta.png' % out_dir 
                  ,'fractional error distribution for zeta')

print('the mean fractional error of zeta is', zeta_error_mean,' ',
      'the corresponding 16th pencentile is', zeta_percentiles_16th,' ',
      'the corresponding 83rd pencentile is', zeta_percentiles_83rd)



# plot error for radfield and compute some statistics
column5 = 'radfield'
radfield_pre = read_csv(filename1, column5)
radfield_raw = read_csv(filename2, column5)
plot_prediction(radfield_pre, radfield_raw, '%sscatter_radfield.png' % out_dir, 
                'scatter plot for radfield')
fractional_error_radfield = compute_error_list(radfield_pre, radfield_raw)

# compute mean error for radfield
radfield_error_mean = sum(fractional_error_radfield)/len(fractional_error_radfield)
# compute 16th and 83rd percentiles
radfield_percentiles_16th = np.percentile(fractional_error_radfield, 16)
radfield_percentiles_83rd = np.percentile(fractional_error_radfield, 83)

plot_distribution(fractional_error_radfield, radfield_percentiles_16th,
                  radfield_percentiles_83rd, '%sradfield.png' % out_dir, 
                  'fractional error distribution for radfield')

print('the mean fractional error of radfield is', radfield_error_mean,' ',
      'the corresponding 16th pencentile is', radfield_percentiles_16th,' ',
      'the corresponding 83rd pencentile is', radfield_percentiles_83rd)



# plot error for Av and compute some statistics
column6 = 'Av'
Av_pre = read_csv(filename1, column6)
Av_raw = read_csv(filename2, column6)
plot_prediction(Av_pre, Av_raw, '%sscatter_Av.png' % out_dir, 
                'scatter plot for Av')
fractional_error_Av = compute_error_list(Av_pre, Av_raw)

# compute mean error for Av
Av_error_mean = sum(fractional_error_Av)/len(fractional_error_Av)
# compute 16th and 83rd percentiles
Av_percentiles_16th = np.percentile(fractional_error_Av, 16)
Av_percentiles_83rd = np.percentile(fractional_error_Av, 83)

plot_distribution(fractional_error_Av, Av_percentiles_16th, 
                  Av_percentiles_83rd, '%sAv.png' % out_dir 
                  ,'fractional error distribution for Av')

print('the mean fractional error of Av is', Av_error_mean,' ',
      'the corresponding 16th pencentile is', Av_percentiles_16th,' ',
      'the corresponding 83rd pencentile is', Av_percentiles_83rd)



# plot error for rout and compute some statistics
column6 = 'rout'
rout_pre = read_csv(filename1, column6)
rout_raw = read_csv(filename2, column6)
plot_prediction(rout_pre, rout_raw, '%sscatter_rout.png' % out_dir, 
                'scatter plot for rout')
fractional_error_rout = compute_error_list(rout_pre, rout_raw)

# compute mean error for rout
rout_error_mean = sum(fractional_error_rout)/len(fractional_error_rout)
# compute 16th and 83rd percentiles
rout_percentiles_16th = np.percentile(fractional_error_rout, 16)
rout_percentiles_83rd = np.percentile(fractional_error_rout, 83)

plot_distribution(fractional_error_rout, rout_percentiles_16th,
                  rout_percentiles_83rd, '%srout.png' % out_dir 
                  ,'fractional error distribution for rout')

print('the mean fractional error of rout is', rout_error_mean,' ',
      'the corresponding 16th pencentile is', rout_percentiles_16th,' ',
      'the corresponding 83rd pencentile is', rout_percentiles_83rd)





