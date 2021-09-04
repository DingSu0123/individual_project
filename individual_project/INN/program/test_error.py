import os
import csv
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def write_csv(list_to_write, path):
    
    """
    Parameters
    ----------
    list_to_write : list
        data to write.
    path : string
        path to store the output csv files.

    Returns
    -------
    None.

    """  
    
    with open(path, 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        for row in list_to_write:
            writer.writerow(row)


def read_csv(filename, column):
    
    """
    Parameters
    ----------
    filename : string.
        choose the csv file.
    column : string.
        choose which column of dataset to read.

    Returns
    -------
    a list of data
    
    """
    
    data_col = pd.read_csv(filename, usecols=[column])
    data_col = data_col.values.tolist()
     
    return data_col

    
def compute_error_list(list_pre, list_raw):
    
    """
    Parameters
    ----------
    list_pre : list
        a list of predictions
    list_raw: list
        a list of raw data
        
    Returns
    -------
    a list of fractional errors
    
    """
    
    frac_error = [(abs(true - pre)/true) 
                  for true, pre in zip(np.array(list_raw), np.array(list_pre))]
    
    frac_error = np.nan_to_num(frac_error)
    
    return np.array(frac_error)
    

def plot_distribution(input, percentile1, percentile2, filename, title):   
    
    """
    Parameters
    ----------
    input : numpy array
        error list visualized.       
    percentile1: value
        16th percentile   
    percentile2: value
        83rd percentile       
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
    plt.title(title, fontsize=12)
    
    plt.axvline(input.mean(), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(percentile1, color='r', linewidth=0.5)
    plt.axvline(percentile2, color='g', linewidth=0.5)
    
    plt.savefig(filename, dpi = 500, bbox_inches = 'tight')
    plt.close('all')
    
       

def plot_prediction(pre, raw, filename, title):
    
    """
    Parameters
    ----------
    pre: list
        a list of predictions      
    raw: list
        a list of raw data    
    filename : string
        define the png name.
    title : string
        title of pngs.

    Returns
    -------
    None.

    """
    
    plt.scatter(raw, pre, c ='b', marker = '.')
    plt.xlabel('real value', fontsize=12)
    plt.ylabel('predicted value', fontsize=12)
    plt.title(title, fontsize=12)
    plt.savefig(filename, dpi = 500, bbox_inches = 'tight')
    plt.close('all')
    




# testing for one parameter 

# input the csv generated from INN_model
filename1 = 'pre.csv'
filename2 = 'raw.csv'

# choose output directory
out_dir = "/Users/suding/Desktop/individual_project/plot_error_distribution/"
os.system('mkdir -p %s' % out_dir)

# plot errors for initialDens and compute some statistics
column1 = 'initialDens'
initialDens_pre = read_csv(filename1, column1)
initialDens_raw = read_csv(filename2, column1)
plot_prediction(initialDens_pre, initialDens_raw, '%sscatter_initialDens.png' % out_dir, 'scatter plot for initialDens')

# compute fractional errors
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

