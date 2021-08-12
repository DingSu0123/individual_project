#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 17:05:24 2021

@author: suding
"""


import csv
import torch
import torch.optim
import torch.nn as nn
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from sklearn import preprocessing
from FrEIA.framework import InputNode, OutputNode, Node, GraphINN
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom


import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def MMD_multiscale(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*zz

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    for a in [0.2, 0.5, 0.9, 1.3]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1

    return torch.mean(XX + YY - 2.*XY)


def fit(input, target):
    return torch.mean((input - target)**2)


def plot_losses(losses,filename,logscale=False,legend=None):
    """ Make loss and accuracy plots and output to file.
    Plot with x and y log-axes is desired
    Parameters
    ----------
    losses: list
        list containing history of network loss and accuracy values
    filename: string
        string which specifies location of output directory and filename
    logscale: boolean
        if True: use logscale in plots, if False: do not use
    legend: boolean
        if True: apply legend, if False: do not
    pe_losses = [losstot_hist, losslatent_hist, lossrev_hist, lossf_hist]
    """
    # plot forward pass loss
    fig = plt.figure()
    losses = np.array(losses)
    ax1 = fig.add_subplot(611)	
    ax1.plot(losses[0],'b', label='Total')
    ax1.set_xlabel(r'epoch')
    ax1.set_ylabel(r'loss')
    if legend is not None:
    	ax1.legend(loc='upper left')
    
    # plot backward pass loss
    ax2 = fig.add_subplot(612)
    ax2.plot(losses[1],'r', label='latent')
    # rescale axis using a logistic function so that we see more detail
    # close to 0 and close 1
    ax2.set_xlabel(r'epoch')
    ax2.set_ylabel(r'loss')
    if legend is not None:
        ax2.legend(loc='upper left')

    ax3 = fig.add_subplot(613)
    ax3.plot(losses[2],'g', label='reversible')
    # rescale axis using a logistic function so that we see more detail
    # close to 0 and close 1
    ax3.set_xlabel(r'epoch')
    ax3.set_ylabel(r'loss')
    if legend is not None:
        ax3.legend(loc='upper left')

    ax4 = fig.add_subplot(614)
    ax4.plot(losses[3],'cyan', label='forward')
    # rescale axis using a logistic function so that we see more detail
    # close to 0 and close 1
    ax4.set_xlabel(r'epoch')
    ax4.set_ylabel(r'loss')
    if legend is not None:
        ax4.legend(loc='upper left')
        
    # plot forward pass loss --- test dataset
    ax5 = fig.add_subplot(615)	
    ax5.plot(losses[4],'cyan', label='forward_test')
    ax5.set_xlabel(r'epoch')
    ax5.set_ylabel(r'loss')
    if legend is not None:
    	ax5.legend(loc='upper left')
    
    # plot backward pass loss --- test dataset    
    ax6 = fig.add_subplot(616)	
    ax6.plot(losses[5],'g', label='backward_test')
    ax6.set_xlabel(r'epoch')
    ax6.set_ylabel(r'loss')
    if legend is not None:
    	ax6.legend(loc='upper left')

    if logscale==True:
        #ax1.set_xscale("log", nonposx='clip')
        ax1.set_yscale("log", nonposy='clip')
        #ax2.set_xscale("log", nonposx='clip')
        ax2.set_yscale("log", nonposy='clip')
        #ax3.set_xscale("log", nonposx='clip')
        ax3.set_yscale("log", nonposy='clip')
        #ax4.set_xscale("log", nonposx='clip')
        ax4.set_yscale("log", nonposy='clip')
        #ax5.set_xscale("log", nonposx='clip')
        ax5.set_yscale("log", nonposy='clip')
        #ax6.set_xscale("log", nonposx='clip')
        ax6.set_yscale("log", nonposy='clip')
    plt.savefig(filename)
    plt.close('all')


def violin_plot(losses, filename, title):

    # plot forward pass loss - test dataset
    sns.set(style="whitegrid")
    fig = plt.figure()
    losses = np.array(losses)
    ax1 = fig.add_subplot(211)	
    ax1 = sns.violinplot(losses[4],inner='box',orient=('v'))
    ax1.set_title(title)
        
    # plot backwardward pass loss - test dataset
    ax2 = fig.add_subplot(212)	
    ax2 = sns.violinplot(losses[5],inner='box',orient=('v'))
 
    plt.savefig(filename)
    plt.close('all')


def train(model,test_loader,train_loader,n_its_per_epoch,zeros_noise_scale,batch_size,ndim_tot,ndim_x,ndim_y,ndim_z,y_noise_scale,optimizer,lambd_predict,loss_fit,lambd_latent,loss_latent,lambd_rev,loss_backward,i_epoch=0):
    model.train()

    l_tot = 0
    batch_idx = 0

    t_start = time()
        
    loss_factor = 600**(float(i_epoch) / 300) /600
    print(i_epoch)
    if loss_factor > 1:
        loss_factor = 1
    

    for x, y in train_loader:
        
        batch_idx += 1
        if batch_idx > n_its_per_epoch:
            break

        x, y = x.to(device), y.to(device)

        y_clean = y.clone()
        pad_x = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                ndim_x, device=device)
        pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)

        y += y_noise_scale * torch.randn(batch_size, ndim_y, dtype=torch.float, device=device)

        x, y = (torch.cat((x, pad_x),  dim=1),
                torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz, y),
                          dim=1))

        optimizer.zero_grad()


        # Forward step:
        output = model(x)
        #print(output[0].shape)

        # Shorten output, and remove gradients wrt y, for latent loss
        y_short = torch.cat((y[:, :ndim_z], y[:, -ndim_y:]), dim=1)

        l = lambd_predict * loss_fit(output[0][:, ndim_z:], y[:, ndim_z:]) # forward_loss ly
        l_forward = l
    
        output_block_grad = torch.cat((output[0][:, :ndim_z],
                                       output[0][:, -ndim_y:].data), dim=1)

        l_latent = loss_latent(output_block_grad, y_short)    # latent_loss lz
        l += lambd_latent * l_latent
        l_tot += l.data.item()

        l.backward(retain_graph=True)


        # Backward step:
        pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)
        y = y_clean + y_noise_scale * torch.randn(batch_size, ndim_y, device=device)

        orig_z_perturbed = (output[0].data[:, :ndim_z] + y_noise_scale *
                            torch.randn(batch_size, ndim_z, device=device))
        y_rev = torch.cat((orig_z_perturbed, pad_yz,
                           y), dim=1)
        y_rev_rand = torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz,
                                y), dim=1)

        output_rev = model(y_rev, rev=True)
        output_rev_rand = model(y_rev_rand, rev=True)


        l_rev = (
            lambd_rev * 0.50
            * loss_factor
            * loss_backward(output_rev_rand[0][:, :ndim_x],
                            x[:, :ndim_x])
        )

        l_rev += 0.50 * lambd_predict * loss_fit(output_rev[0], x)
        

        l_tot += l_rev.data.item()
        l_rev.backward()

       
        optimizer.step()
    
    
    # return all the backward prediction 
    list_pre = []
    list_x = []
    array_pre = np.zeros(shape=(2816,7))     # row dimension = batch size * ⌊(test_split/batch size)⌋
    array_raw = np.zeros(shape=(2816,7))



    for i in range(0,100):
        
        for x, y in test_loader:
            
            x, y = x.to(device), y.to(device)
        
            y_clean = y.clone()

            pad_x = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                ndim_x, device=device)
            pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)

            y += y_noise_scale * torch.randn(batch_size, ndim_y, dtype=torch.float, device=device)

            x, y = (torch.cat((x, pad_x),  dim=1),
                torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz, y),
                          dim=1))
        
            output = model(x)
  
 
            # Shorten output, and remove gradients wrt y, for latent loss
            y_short = torch.cat((y[:, :ndim_z], y[:, -ndim_y:]), dim=1)

            l_test = loss_fit(output[0][:, ndim_z:], y[:, ndim_z:]) # forward_loss ly
            l_forward_test = l_test
             
        
            # Backward      
            pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)
        
            y = y_clean + y_noise_scale * torch.randn(batch_size, ndim_y, device=device)

            orig_z_perturbed = (output[0].data[:, :ndim_z] + y_noise_scale *
                            torch.randn(batch_size, ndim_z, device=device))
        
            y_rev = torch.cat((orig_z_perturbed, pad_yz,
                           y), dim=1)
        
            y_rev_rand = torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz,             
                            y), dim=1)
        
        
            # store the raw data for each batch
            x_raw = x[:,:ndim_x]
            for m in x_raw:
                list_x.append(m.detach().numpy())
            
            # store the predicted data for each batch
            output_rev = model(y_rev, rev=True)
            for u in output_rev[0][:, :ndim_x]:
                list_pre.append(u.detach().numpy())
                

            output_rev_rand = model(y_rev_rand, rev=True)
            
            l_rev_test = (
            0.50 * loss_factor
            * loss_backward(output_rev_rand[0][:, :ndim_x],
                            x[:, :ndim_x])
            )

            l_rev_test += 0.50 * lambd_predict * loss_fit(output_rev[0], x)
              
        # change the list of prediction and raw data into array for accumualation and taking average later
        array_pre = array_pre + np.array(list_pre)
        array_raw = array_raw + np.array(list_x)
        #print(np.array(list_pre))
        #print(array_pre)
        list_pre = []
        list_x = []
    
    # taking average of both predictions and observations
    # uncertainty estimation   
    array_pre_average = array_pre/100
    array_raw_average = array_raw/100


    return l_tot / batch_idx, l_latent, l_rev, l_forward, l_forward_test, l_rev_test, array_pre_average.tolist(), array_raw_average.tolist()


def read_data(filename, column):
    data = pd.read_csv(filename, usecols = column)
    data = np.array(data)
    
    # log scale the input
    data_log = np.log10(data)
    data_log = np.nan_to_num(data_log)
    # quantile transformation
    data_quantile = preprocessing.quantile_transform(data_log, output_distribution='normal')
      
    # minmax_normalization 
    minmax = preprocessing.MinMaxScaler()
    data_minmax = minmax.fit_transform(data_quantile)
    
    return data_minmax
     

def write_csv(list_to_write, path):
    
    with open(path, 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        for row in list_to_write:
            writer.writerow(row)
    
    
def main():
    
    # Set up simulation parameters
    batch_size = 256  # set batch size
    test_split = 3000   # number of testing samples to use
    
    out_dir = "/Users/suding/Desktop/PHAS0077/results_science_data/"
    # setup output directory - if it does not exist
    os.system('mkdir -p %s' % out_dir)

    # input parameters and line intensities
    filename1 = "model_inputs_corrected.csv"
    column1 = [1, 2, 3, 4, 5, 6, 7]
    filename2 = "HCO_fluxes.csv"
    column2 = [2, 3, 4, 5, 6, 7, 8, 9]   # change this line to predict different transitions
    x_input = torch.tensor(read_data(filename1, column1), dtype=torch.float)
    y_output = torch.tensor(read_data(filename2, column2), dtype=torch.float)
    
    # setting up the model 
    ndim_x = 7        # number of posterior parameter dimensions (x,y)
    ndim_y = 8     # number of label dimensions (noisy data samples)
    ndim_z = 20        # number of latent space dimensions
    ndim_tot = max(ndim_x,ndim_y+ndim_z)     # must be > ndim_x and > ndim_y + ndim_z

    # define different parts of the network
    nodes = [InputNode(ndim_tot, name='input')]

    # define hidden layer nodes
    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                         nn.Linear(512,  c_out))
        
    for k in range(4):
        nodes.append(Node(nodes[-1],
                      GLOWCouplingBlock,
                      {'subnet_constructor':subnet_fc, 'clamp':2.0},
                      name=F'coupling_{k}'))
        nodes.append(Node(nodes[-1],
                      PermuteRandom,
                      {'seed':k},
                      name=F'permute_{k}'))
            
    # define output layer node
    nodes.append(OutputNode(nodes[-1], name='output'))
    model = GraphINN(nodes)
    
    # Train model
    # Training parameters
    n_epochs = 100
    meta_epoch = 10 
    n_its_per_epoch = 12

    lr = 1e-2
    gamma = 0.01**(1./120)
    l2_reg = 2e-5

    y_noise_scale = 3e-2
    zeros_noise_scale = 3e-2

    # relative weighting of losses:
    lambd_predict = 1
    lambd_latent = 2.25
    lambd_rev = 2.5 

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.8, 0.8),
                             eps=1e-04, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=meta_epoch,
                                                gamma=gamma)

    # define the three loss functions
    loss_backward = MMD_multiscale
    loss_latent = MMD_multiscale
    loss_fit = fit

    # set up training set data loader  
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_input[test_split:], y_output[test_split:]),
        batch_size = batch_size, shuffle=True, drop_last=True)
    
    # set up test dataset
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_input[:test_split], y_output[:test_split]),
        batch_size = batch_size, shuffle=True, drop_last=True)
    
    
    # store loss values for training dataset
    train_lossf_hist = []
    train_lossrev_hist = []
    train_losstot_hist = []
    train_losslatent_hist = []
    
    # store loss values for test dataset
    test_lossf_hist = []
    test_lossrev_hist = []

    
    # start training loop
    try:
        
        t_start = time()
        # loop over number of epochs
        list_prediction = []
        list_raw_data = []

        
        for i_epoch in tqdm(range(n_epochs), ascii=True, ncols=80):

            scheduler.step()

            # Initially, the l2 reg. on x and z can give huge gradients, set
            # the lr lower for this
            if i_epoch < 0:
                print('inside this iepoch<0 thing')
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * 1e-2


            # train the model
            losstot, losslatent, lossrev, lossf, lossf_test, lossrev_test, list_pre, list_x = train(model,test_loader,train_loader,n_its_per_epoch,zeros_noise_scale,batch_size,
                ndim_tot,ndim_x,ndim_y,ndim_z,y_noise_scale,optimizer,lambd_predict,
                loss_fit,lambd_latent,loss_latent,lambd_rev,loss_backward,i_epoch)
            
            
            for i in list_pre:
                list_prediction.append(i)     
            
            for i in list_x:
                list_raw_data.append(i)
                
                
            # append current training loss value to loss histories
            train_lossf_hist.append(lossf.data.item())
            train_lossrev_hist.append(lossrev.data.item())
            train_losstot_hist.append(losstot)
            train_losslatent_hist.append(losslatent.data.item())
                       
            # append current test loss value to loss histories
            test_lossf_hist.append(lossf_test.data.item())
            test_lossrev_hist.append(lossrev_test.data.item())
        
            train_losses = [train_losstot_hist, train_losslatent_hist, train_lossrev_hist, train_lossf_hist, test_lossf_hist, test_lossrev_hist]
            
            #print(train_losses[4])
            plot_losses(train_losses,'%spe_losses.png' % out_dir,legend=['PE-GEN'])
            plot_losses(train_losses,'%spe_losses_logscale.png' % out_dir,logscale=True,legend=['PE-GEN'])
            
            title = "loss_distribution for test set"
            violin_plot(train_losses, '%sloss_distribution.png' % out_dir, title)
            
            
        return list_prediction, list_raw_data
            

                       
    except KeyboardInterrupt:
        pass
    
    finally:
        print("\n\nTraining took {(time()-t_start)/60:.2f} minutes\n")


 
# testing for writing csv files
path1 = '/Users/suding/Desktop/PHAS0077/INN/prediction.csv' 
path2 = '/Users/suding/Desktop/PHAS0077/INN/raw_data.csv' 
   
list1, list2 = main()
   
write_csv(list1, path1)
write_csv(list2, path2)
    
    

    
    
    
    
    
    



