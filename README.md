# Individual Project (PHAS0077)
Individual project for MSc Scientific and Data Intensive Computing 

## Content
1. INN <br> 
2. error_analysis <br> 
3. plot_data_distribution <br> 
4. training_loss <br> 

### INN
The main program called `INN_model.py` based on the Invertible Neural Networks can be found in `program` file. By running the `INN_model.py`, two lists will be returned. The first one called `prediction.csv` contains all predicted values for each training. All predictions are taken average after running the whole test dataset many times. `raw_data.csv` is the corresponding raw data of these predictions<br> 

`plot_data_distribution.py` is used to visualize the distribution of all parameters before and after pre-processing.<br> 

`test_error.py` computes the fractional errors for all predictions and visualizes the errors via histograms and violinplots. Scatter plots (predictions vs real data) are also used to show the capability of parameter estimation.

### error_analysis
`CO_frac_errors` file contains all the histograms and a csv file (storing some statistics of fractional errors). `CO_scatter_plots` file contains all the scatter plots.

### plot_data_distribution
This file contains all the figures related to data distribution before and after preprocessing.

### training_loss
Training losses are shown in `pe_losses_logscale.png` and `pe_losses.png`. `loss_distribution.png` is the violin plot of forward and backward losses for test dataset.














