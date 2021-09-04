import os
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'



def MMD_multiscale(x, y):
    
    """
    Goal: perform maximum mean discrepancy.
    Choice of kernel function: Inverse Multi-quadratic.
    input: two multi-dimensional matrices x and y.
        
    Returns
    -------
    a distance on the space of probability measures.

    """
    
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
    
    """
    Goal: compute mean squared error (MSE).
    input: two multi-dimensional matrices (i.e., input and target).
        
    Returns
    -------
    mean squared error.

    """
    
    return torch.mean((input - target)**2)


def plot_losses(losses,filename,logscale=False,legend=None):
    
    """
    Parameters
    ----------
    losses: a multi-dimensional list (contain training losses).
        contains total_losses, losses for latent variables(lz), backward losses(lx), forward losses(ly).
        
    filename: string
        define the png name.
        
    logscale: True or False (default: false).
        set true to log scale different losses.

    Returns
    -------
    None.

    """
    
    # plot total losses
    fig = plt.figure()
    losses = np.array(losses)
    ax1 = fig.add_subplot(611)	
    ax1.plot(losses[0],'b', label='Total')
    ax1.set_xlabel(r'epoch',fontsize=12)
    if legend is not None:
    	ax1.legend(loc='upper left')
    
    # plot losses for latent variables (lz)
    ax2 = fig.add_subplot(612)
    ax2.plot(losses[1],'r', label='latent')
    ax2.set_xlabel(r'epoch', fontsize=12)
    if legend is not None:
        ax2.legend(loc='upper left')
   
    # plot backward losses (lx)
    ax3 = fig.add_subplot(613)
    ax3.plot(losses[2],'g', label='reversible')
    ax3.set_xlabel(r'epoch', fontsize=12)
    if legend is not None:
        ax3.legend(loc='upper left')
    
    # plot forward losses (ly)
    ax4 = fig.add_subplot(614)
    ax4.plot(losses[3],'cyan', label='forward')
    ax4.set_xlabel(r'epoch', fontsize=12)
    if legend is not None:
        ax4.legend(loc='upper left')
 
    # take the logrithm of y_axis (log scale losses to check the magnitude)
    if logscale==True:
        ax1.set_yscale("log", nonposy='clip')
        ax2.set_yscale("log", nonposy='clip')
        ax3.set_yscale("log", nonposy='clip')
        ax4.set_yscale("log", nonposy='clip')

    plt.savefig(filename, dpi = 500, bbox_inches = 'tight')
    plt.close('all')


def violin_plot(losses, filename, title):
    
    """
    Parameters
    ----------
    losses: a multi-dimensional list (test losses).
        contains forward losses(ly), backward losses(lx).
        
    filename: string
        define the png name.
        
    title: string
        title of pngs.
        
    Returns
    -------
    None.

    """

    # plot forward pass loss - test dataset
    sns.set(style="whitegrid")
    fig = plt.figure()
    losses = np.array(losses)
    ax1 = fig.add_subplot(211)	
    ax1 = sns.violinplot(losses[2],inner='box',orient=('v'))
    ax1.set_title(title, fontsize=12)
        
    # plot backward pass loss - test dataset
    ax2 = fig.add_subplot(212)	
    ax2 = sns.violinplot(losses[3],inner='box',orient=('v'))
    
    # save pngs
    plt.savefig(filename, dpi = 500, bbox_inches = 'tight')
    plt.close('all')


def train(model,test_loader,train_loader,n_its_per_epoch,zeros_noise_scale,batch_size,ndim_tot,ndim_x,ndim_y,ndim_z,
          y_noise_scale,optimizer,lambd_predict,loss_fit,lambd_latent,loss_latent,lambd_rev,loss_backward,i_epoch=0):
    
    """
    Parameters
    ----------
    model: INN model   
    test_loader & train_loader: two loaders to extract batches from test and training dataset  
    n_its_per_epoch: number of batches loaded for each training    
    zeros_noise_scale: a scaler to reduce the value of random numbers for padding
    batch_size: batch size
    ndim_tot: total dimensions    
    ndim_x: dimensions for input x
    ndim_y: dimensions for output y
    ndim_z: dimensions for latent variables z
    y_noise_scale: a scaler to reduce the value of noises added on y
    optimizer: optimizer to optimize parameters for each layer
    lambd_predict: relative weight for forward losses
    loss_fit: loss function to compute forward losses
    lambd_latent: relative weight for losses of latent variables
    loss_latent: loss function to compute losses for latent variables
    lambd_rev: relative weight for backward losses
    loss_backward: loss function to compute backward losses

    Returns
    -------
    l_tot/batch_idx: average total losses
    l_latent: losses for latent variables for training dataset
    l_forward: forward losses for training dataset
    l_rev: backward losses for training dataset
    l_forward_test: forward losses for test dataset
    l_rev_test: backward losses for test dataset
    array_pre_average.tolist(): a list to store all predictions from the test dataset (after being taken average)
    array_raw_average.tolist(): a list to store all raw data from the test dataset (after being taken average)
    
    """
    
    model.train()
    l_tot = 0
    batch_idx = 0
    t_start = time()
    
    # lower the loss factor to avoid gradient explosion for early stage of training
    loss_factor = 600**(float(i_epoch) / 300) /600
    print(i_epoch)
    if loss_factor > 1:
        loss_factor = 1
    
    # use the train loader to extract data for training
    for x, y in train_loader:
        
        batch_idx += 1
        if batch_idx > n_its_per_epoch:
            break

        x, y = x.to(device), y.to(device)
        y_clean = y.clone()
        
        # add some padding
        pad_x = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                ndim_x, device=device)
        pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)
        
        # add some noises on y
        y += y_noise_scale * torch.randn(batch_size, ndim_y, dtype=torch.float, device=device)
       
        # make the dimensions of x and y equal total dimensions
        x, y = (torch.cat((x, pad_x),  dim=1),
                torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz, y),
                          dim=1))

        optimizer.zero_grad()



        # Forward step:
        output = model(x)

        # Shorten output, and remove gradients wrt y, for latent loss
        y_short = torch.cat((y[:, :ndim_z], y[:, -ndim_y:]), dim=1)

        # compute forward losses ly
        l = lambd_predict * loss_fit(output[0][:, ndim_z:], y[:, ndim_z:]) 
        l_forward = l
    
        # compute losses for latent variables z(i.e., lz)
        output_block_grad = torch.cat((output[0][:, :ndim_z],
                                       output[0][:, -ndim_y:].data), dim=1)
        l_latent = loss_latent(output_block_grad, y_short)   
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
        
        # compute backward output
        output_rev = model(y_rev, rev=True)
        output_rev_rand = model(y_rev_rand, rev=True)

        # compute backward losses from two sources(i.e., lx)
        l_rev = (
            lambd_rev * 0.50
            * loss_factor
            * loss_backward(output_rev_rand[0][:, :ndim_x],
                            x[:, :ndim_x])
        )

        l_rev += 0.50 * lambd_predict * loss_fit(output_rev[0], x)
        
        # update total training losses
        l_tot += l_rev.data.item()
        l_rev.backward()

        optimizer.step()
    
    
    
    # initialize lists to store all the backward predictions
    list_pre = []
    list_x = []
    array_pre = np.zeros(shape=(2816,7))     # row dimension = batch size * ⌊(test_split/batch size)⌋
    array_raw = np.zeros(shape=(2816,7))

    # initialize two lists to store the data from the test loader (avoid inconsistency of batches)
    list123x = []
    list123y = []
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        list123x.append(x)
        list123y.append(y)
        
    # run the model for the same test dataset 50 times to consider uncertainty estimation
    for i in range(0,50):
        
        for i in range(len(list123x)):
            
            x = list123x[i]
            y = list123y[i]
            y_clean = y.clone()


            # Forward:
                
            # add some padding
            pad_x = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                ndim_x, device=device)
            pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)

            y += y_noise_scale * torch.randn(batch_size, ndim_y, dtype=torch.float, device=device)
            x, y = (torch.cat((x, pad_x),  dim=1),
                torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz, y),
                          dim=1))
        
            # compute forward output
            output = model(x)
  
            # Shorten output, and remove gradients wrt y, for latent loss
            y_short = torch.cat((y[:, :ndim_z], y[:, -ndim_y:]), dim=1)

            # compute forward losses for test dataset
            l_test = loss_fit(output[0][:, ndim_z:], y[:, ndim_z:])
            l_forward_test = l_test
             
        
            # Backward:
                
            pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)
        
            y = y_clean + y_noise_scale * torch.randn(batch_size, ndim_y, device=device)

            orig_z_perturbed = (output[0].data[:, :ndim_z] + y_noise_scale *
                            torch.randn(batch_size, ndim_z, device=device))
        
            y_rev = torch.cat((orig_z_perturbed, pad_yz,
                           y), dim=1)
        
            y_rev_rand = torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz,             
                            y), dim=1)
               
            # store the raw data
            x_raw = x[:,:ndim_x]
            for m in x_raw:
                list_x.append(m.detach().numpy())
            
            # store the predicted data
            output_rev = model(y_rev, rev=True)
            for u in output_rev[0][:, :ndim_x]:
                list_pre.append(u.detach().numpy())
                
                
            # compute backward losses for test dataset
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
        
        # empty the lists before second loop
        list_pre = []
        list_x = []
             
    # taking average of both predictions and observations
    # uncertainty estimation   
    array_pre_average = array_pre/50
    array_raw_average = array_raw/50


    return l_tot / batch_idx, l_latent, l_rev, l_forward, l_forward_test, l_rev_test, array_pre_average.tolist(), array_raw_average.tolist()


def read_data(filename, column):
    
    """
    Parameters
    ----------
    filename : string.
        choose the csv file.
    column : string.
        choose which column of dataset to read.

    Returns
    -------
    data after preprocessing.

    """
    
    data = pd.read_csv(filename, usecols = column)
    data = np.array(data)
    
    # log scale the input
    data_log = np.log10(data)

    # quantile transformation
    data_quantile = preprocessing.quantile_transform(data_log, output_distribution='uniform')

    return data_quantile
     

def write_csv(list_to_write, path):
    
    """
    Parameters
    ----------
    list_to_write : list
        data to write.
    path : string
        path to store the ouput csv files.

    Returns
    -------
    None.

    """   
    
    with open(path, 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        for row in list_to_write:
            writer.writerow(row)
    
    
def main():
    
    """
    None.

    Returns
    -------
    list_prediction: list to store all predictions for each training.
    list_raw_data: list to store all corresponding raw data.

    """
    
    # Set up simulation parameters
    batch_size = 256  # set batch size
    test_split = 3000   # number of testing samples to use
    
    # setup output directory - if it does not exist, create it
    out_dir = "/Users/suding/Desktop/individual_project/results_science_data/"
    os.system('mkdir -p %s' % out_dir)

    # input parameters and molecular observations
    filename1 = "model_inputs.csv"
    column1 = [1, 2, 3, 4, 5, 6, 7]      # change this to input other physical parameters
    filename2 = "CO_fluxes.csv"
    column2 = [2, 3, 4, 5, 6, 7, 8, 9]   # change this to predict different transitions
    x_input = torch.tensor(read_data(filename1, column1), dtype=torch.float)
    y_output = torch.tensor(read_data(filename2, column2), dtype=torch.float)
    
    # setting up the model 
    ndim_x = 7        # dimensions of input x
    ndim_y = 8        # dimensions of output y
    ndim_z = 20       # dimensions of latent variables z
    ndim_tot = max(ndim_x,ndim_y+ndim_z)     # total dimensions: must be > ndim_x and > ndim_y + ndim_z

    # define different parts of the network
    
    # define input layer:
    nodes = [InputNode(ndim_tot, name='input')]
        
    # a feed-forward subnetwork to predict affine coefficients
    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                         nn.Linear(512,  c_out))
    
    # define hidden layers   
    for k in range(4):
        nodes.append(Node(nodes[-1],
                      GLOWCouplingBlock,
                      {'subnet_constructor':subnet_fc, 'clamp':2.0},
                      name=F'coupling_{k}'))
        nodes.append(Node(nodes[-1],
                      PermuteRandom,
                      {'seed':k},
                      name=F'permute_{k}'))
            
    # define output layer
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
    
    # set up test set data loader 
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
                
            # append current training loss values to loss histories
            train_lossf_hist.append(lossf.data.item())
            train_lossrev_hist.append(lossrev.data.item())
            train_losstot_hist.append(losstot)
            train_losslatent_hist.append(losslatent.data.item())
                       
            # append current test loss values to loss histories
            test_lossf_hist.append(lossf_test.data.item())
            test_lossrev_hist.append(lossrev_test.data.item())
            
            # store all training losses
            train_losses = [train_losstot_hist, train_losslatent_hist, train_lossrev_hist, train_lossf_hist]
            
            # visualize training losses
            plot_losses(train_losses,'%spe_losses.png' % out_dir,legend=['PE-GEN'])
            plot_losses(train_losses,'%spe_losses_logscale.png' % out_dir,logscale=True,legend=['PE-GEN'])
            title = "loss_distribution for training set"
            
            # visualize forward and backward losses for test dataset
            violin_plot(train_losses, '%sloss_distribution.png' % out_dir, title)
                        
        return list_prediction, list_raw_data
                                
    except KeyboardInterrupt:
        pass
    
    finally:
        print("\n\nTraining took %f minutes\n\n" % ((time()-t_start)/60.))



    
    

    
    
    
    
    
    
