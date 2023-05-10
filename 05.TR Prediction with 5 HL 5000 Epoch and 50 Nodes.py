#!/usr/bin/env python
# coding: utf-8

# In[1]:
import time
start_time = time.time()
# print('start_time: ', start_time)
# PySimpleGUI as sg
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasAgg
# import matplotlib.backends.tkagg as tkagg
import tkinter as Tk
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import sys
import pickle
import torch
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from torch import nn, optim
from torchvision import transforms
from collections import OrderedDict
get_ipython().run_line_magic('matplotlib', 'inline')
# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
## Transforms features by scaling each feature to a given range.
## This estimator scales and translates each feature individually such that it is in the given range on the training set, i.e. between zero and one.
## This transformation is often used as an alternative to zero mean, unit variance scaling.
## fit(X[, y])	Compute the minimum and maximum to be used for later scaling.
## transform(X)	Scaling features of X according to feature_range.
## fit_transform(X[, y])	Fit to data, then transform it.
## inverse_transform(X)	Undo the scaling of X according to feature_range.
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
no_of_output_nodes = 3
df_1 = pd.read_excel('Dataset_for_TRA_Prediction.xlsx', sheet_name = 'Sheet2')
datafile_1 = df_1.values                  ## stored data from xlsx file  WHY WE ARE STORING VALUES IN dtafile_1
print(datafile_1)
print(len(datafile_1))
print(datafile_1[:,0])
# In[2]:
df_1.head()
# In[3]:
df_1.columns
# In[4]:
########   just to see output variable values   #############################
out_var_datafile_1 = datafile_1[:,range(3,6)]  ## stored output_variable (4th column) from xlsx file Sensitivity to PL
print(out_var_datafile_1)
# In[5]:
out_var_datafile_1 = out_var_datafile_1.reshape((-1, no_of_output_nodes))    ## one column with unknown no. of rows
print(out_var_datafile_1)                                                    ## one WHY WE ARE DOING THIS?
print('no. of training points: ', len(out_var_datafile_1))
# In[6]:
A = scaler1.fit(datafile_1)
print(A)
B = scaler2.fit(out_var_datafile_1)
print(B)
print(datafile_1)
print('\n')
scaler_datafile_1 = scaler1.transform(datafile_1)              ## WHY WE ARE TRANSFORMING THESE VALUES
print(scaler_datafile_1)
print()
scaler_datafile_2 =  scaler1.inverse_transform(scaler_datafile_1)
print(scaler_datafile_2)
# In[7]:
X = scaler_datafile_1[:,range(0,3)] ## Input variables columns Major, Minor, Gap, Height
y = scaler_datafile_1[:,range(3,6)] ## Output variables columns Sensitivity, FWHM, Q-Factor, PL
print(X)
print('\n')
print(y)
X, y = shuffle(X, y)      ## WHY WE USING SHUFFLE HERE
print('\n')
print(X)
print('\n')
print(y)
# In[8]:
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.1)
X_train = X_train.reshape(-1, 3)                            ## 2nd column value is = no. of input variables columns
y_train = y_train.reshape(-1, no_of_output_nodes)           ## 2nd column value is = no. of output variables columns
X_validation = X_validation.reshape(-1, 3)                  ## 2nd column value is = no. of input variables columns
y_validation = y_validation.reshape(-1, no_of_output_nodes) ## 2nd column value is = no. of output variables columns
print('no. of training points: ', len(X_train))             ##  Validation = test
print('no. of validation points: ', len(X_validation))
# In[9]:
print(X_train)
print(y_validation)
# In[10]:
###########     manual testing  for Minor Variation  #############################################
df_2 = pd.read_excel('Dataset_for_TRA_Prediction.xlsx', sheet_name = 'Sheet3')
df_2.head()
datafile_2 = df_2.values                  ## stored data from xlsx file
print(datafile_2)
scaler_datafile_2 = scaler1.transform(datafile_2)
X_test = scaler_datafile_2[:,range(0,3)]                     ## input variables columns
y_test = scaler_datafile_2[:,range(3,6)]                     ## output variables columns
print(X_test)
print('\n')
print(y_test)
print('no. of test points: ', len(X_test))
X_test = X_test.reshape(-1, 3)                      ## 2nd column value is = no. of input variables columns
y_test = y_test.reshape(-1, no_of_output_nodes)     ## 2nd column value is = no. of output variables columns
###################################################################################################
# In[13]:
input_dim = 3                                       ## = no. of input variables columns
output_dim = no_of_output_nodes                     ## = no. of output variables columns
from collections import OrderedDict
'''
# ############     model without dropout     #####################
nodes_hidden_1 = 100
nodes_hidden_2 = 100
### nn.Linear() is fully connected layer
model = nn.Sequential(OrderedDict([
                         ('fc1', nn.Linear(input_dim, nodes_hidden_1)),
                         ('relu', nn.ReLU()),
                         ('fc2', nn.Linear(nodes_hidden_1, nodes_hidden_2)),
                         ('relu', nn.ReLU()),
                         ('fc3', nn.Linear(nodes_hidden_2, output_dim)),
                         ]))
'''
############     model with dropout - 3 layers    ###########################################
##########       dropout_prob leads to variations in mse curve ##############################
dropout_prob = 0.0
nodes_hidden_1 = 50
nodes_hidden_2 = 50
nodes_hidden_3 = 50
nodes_hidden_4 = 50
nodes_hidden_5 = 50
## nn.Linear() is fully connected layer
model = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(input_dim, nodes_hidden_1)),
                        ('relu', nn.ReLU()),
                        ('dropout', nn.Dropout(dropout_prob)),
                        ('fc2', nn.Linear(nodes_hidden_1, nodes_hidden_2)),
                        ('relu', nn.ReLU()),
                        ('dropout', nn.Dropout(dropout_prob)),
                        ('fc3', nn.Linear(nodes_hidden_2, nodes_hidden_3)),
                        ('relu', nn.ReLU()),
                        ('dropout', nn.Dropout(dropout_prob)),
                        ('fc4', nn.Linear(nodes_hidden_3, nodes_hidden_4)),
                        ('relu', nn.ReLU()),
                        ('dropout', nn.Dropout(dropout_prob)),
                        ('fc5', nn.Linear(nodes_hidden_4, nodes_hidden_5)),
                        ('relu', nn.ReLU()),
                        ('dropout', nn.Dropout(dropout_prob)),
                        ('fc6', nn.Linear(nodes_hidden_5, output_dim)),
                        ]))

#############     model with dropout - 2 layers     ####################################
# ####             dropout_prob leads to variations in mse curve    #####################
# dropout_prob = 0.1           # 0.5 - used in nvidia model-behavioural cloning
# nodes_hidden_1 = 50
# nodes_hidden_2 = 50
# ## nn.Linear() is fully connected layer
# model = nn.Sequential(OrderedDict([
#                         ('fc1', nn.Linear(input_dim, nodes_hidden_1)),
#                         ('relu', nn.ReLU()),
#                         ('dropout', nn.Dropout(dropout_prob)),
#                         ('fc2', nn.Linear(nodes_hidden_1, nodes_hidden_2)),
#                         ('relu', nn.ReLU()),
#                         ('dropout', nn.Dropout(dropout_prob)),
#                         ('fc3', nn.Linear(nodes_hidden_2, output_dim)),
#                         ]))
print(model)
# model.double()
print(X_train)
print(y_train)
print(X_train.shape, y_train.shape)
# In[14]:
print(X_train)
# In[15]:
criterion = nn.MSELoss()
learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
print(device)
## move model to gpu if available, else cpu
# model.to(device)
epochs = 5000
# Convert numpy array to torch Variable
# inputs = torch.from_numpy(X_train).requires_grad_()
# labels = torch.from_numpy(y_train)
inputs = torch.Tensor((X_train))
labels = torch.Tensor((y_train))
inputs_validation = torch.Tensor((X_validation))
labels_validation = torch.Tensor((y_validation))
running_loss = []
running_loss_validation = []
for epoch in range(epochs):
    epoch += 1
        ##############   train the model   ######################
    model.train()    # prep model for training
    # Clear gradients w.r.t. parameters, else gradients will be added up with every previous pass
    optimizer.zero_grad()
    # Forward to get output
    outputs = model(inputs)
    # Calculate Loss
    loss = criterion(outputs, labels)       ## mean squared error
    # Getting gradients w.r.t. parameters
    loss.backward()
    # Updating parameters
    optimizer.step()         ## take a step with optimizer to update the weights
    running_loss.append(loss.item())
    
    # ###############    validate the model (not showing fluctuations)      ###################
    # # Turn off gradients for validation, saves memory and computations
    # with torch.no_grad():
    #     ## this turns off dropout for evaluation mode of model
    #     model.eval()      # prep model for evaluation
    #     outputs_validation = model(inputs_validation)
    #     loss_validation = criterion(outputs_validation, labels_validation)
    #     running_loss_validation.append(loss_validation.item())

    
    # ###############    validate the model (showing fluctuations)      ###################
    outputs_validation = model(inputs_validation)
    loss_validation = criterion(outputs_validation, labels_validation)
    running_loss_validation.append(loss_validation.item())

    print('epoch: {}, mse_loss: {:.6f}, mse_loss_validation: {:.6f}'.format(epoch, loss.item(), 
                                                                            loss_validation.item()))
# In[16]:
# print(mean_squared_error(outputs_validation,labels_validation))
# if (epoch == 1000):
#     torch.save(model.state_dict(), 'checkpoint_1000.pth')
# elif (epoch == 2500):
#     torch.save(model.state_dict(), 'checkpoint_2500.pth')
# elif (epoch == 5000):
#     torch.save(model.state_dict(), 'checkpoint_5000.pth')
# elif (epoch == 7500):
#     torch.save(model.state_dict(), 'checkpoint_7500.pth')
# elif (epoch == 10000):
#     torch.save(model.state_dict(), 'checkpoint_10000.pth')
# elif (epoch == 12500):
#     torch.save(model.state_dict(), 'checkpoint_12500.pth')
# elif (epoch == 15000):
#     torch.save(model.state_dict(), 'checkpoint_15000.pth')

# save the model, as weights & parameters are stored in model.state_dict()
# print(model.state_dict().keys())
# print(model.state_dict())
#### torch.save(model.state_dict(), 'checkpoint-epochs-{}.pth'.format(epochs))
torch.save(model.state_dict(), 'checkpoint.pth')
# # load the saved model at particular epochs to compare
state_dict = torch.load('checkpoint.pth')
# load the saved model
#### state_dict = torch.load('checkpoint-epochs-{}.pth'.format(epochs))
# state_dict = torch.load('checkpoint.pth')
# state_dict = torch.load('checkpoint-simple_waveguide_neff_pytorch_1_epochs-5000.pth')
model.load_state_dict(state_dict)
# Purely inference
# predicted_on_X_train = model(torch.Tensor(X_train).requires_grad_()).data.numpy()
# predicted_on_X_validation = model(torch.Tensor(X_validation).requires_grad_()).data.numpy()
# predicted_on_X_test = model(torch.Tensor(X_test).requires_grad_()).data.numpy()
with torch.no_grad():
    ## this turns off dropout for evaluation mode of model
    model.eval()
    predicted_on_X_train = model(torch.Tensor(X_train)).data.numpy()
    predicted_on_X_validation = model(torch.Tensor(X_validation)).data.numpy()
    predicted_on_X_test = model(torch.Tensor(X_test)).data.numpy()
    predicted_on_y_test = model(torch.Tensor(y_test)).data.numpy()
    ##predicted_on_X_test1 = model(torch.Tensor(X_test1)).data.numpy()
    ##predicted_on_X_test2 = model(torch.Tensor(X_test2)).data.numpy()
# print(predicted)

    end_time = time.time()
    print('end_time: ', end_time)
    print('time taken to train in sec: ', (end_time - start_time))

    ## make axis bold
    plt.rcParams.update({'font.size': 10})
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    mse_training_interval = 1  
    mse_validation_interval = 1
    running_loss = running_loss[::mse_training_interval]
    running_loss_index = [i for i in range(1, epochs+1, mse_training_interval)]
    running_loss_validation = running_loss_validation[::mse_validation_interval]
    running_loss_validation_index = [i for i in range(1, epochs+1, mse_validation_interval)]
    print('mse lengths: ', len(running_loss), len(running_loss_validation))
    # print('running_loss_index: ', running_loss_index)
    # print('running_loss_validation_index: ', running_loss_validation_index)

# In[17]:
print(y_train)
print(X_test)
print(scaler2.inverse_transform(X_test))
print(B.inverse_transform(X_test))

print(y_test)
print(scaler2.inverse_transform(y_test))
print(B.inverse_transform(y_test))

# In[18]:
print(scaler2.inverse_transform(y_train))     # WHY WE ARE CONVERTING y_TRAIN
# In[19]:
print(B.inverse_transform(predicted_on_X_train)[:,0])
# In[20]:
print(B.inverse_transform(y_train)[:,0])
# In[21]:
print(B.inverse_transform(X_train))
# In[22]:
print(predicted_on_X_train)
# In[23]:
print(scaler2.inverse_transform(predicted_on_X_test)) #WHY WE ARE CONVERTING predicted_on_X_test
print(scaler2.inverse_transform(predicted_on_y_test))
# In[24]:
##print(scaler2.inverse_transform(predicted_on_X_test1))
# In[25]:
##print(scaler2.inverse_transform(predicted_on_X_test2))
# In[26]:
print(B.inverse_transform(y_test))
# In[27]:
###############################################################################
#################   plotting graphs together - Reflection Spectra  ###################
###############################################################################
plt.figure(figsize=(17,9))
sns.set_style("white") 
###plt.suptitle('NanoAntenna - Reflection Spectra - (epochs - {})'.format(epochs), fontsize=15,
                ###color = 'k', fontweight='bold')     ## giving title on top of all subplots

plt.subplot(231)
plt.title('subplot: A') # No grid lines
ax = sns.lineplot(running_loss_index , running_loss, linewidth = 4, color = 'magenta', 
             label = str('mse_loss_train'))
sns.lineplot(running_loss_validation_index, running_loss_validation, linewidth = 4, color = 'blUE',
             label = str('mse_loss_validation'))
plt.legend(loc = 'best', fontsize = 15, frameon=False)
plt.title('MSE vs Epochs', fontsize = 24 , color = 'k', fontweight ='bold')
plt.xlabel('Epochs', fontsize = 21, color = 'k', fontweight='bold')
plt.ylabel('Mean Squared Error', fontsize = 21 )
plt.tick_params(axis = "both", labelsize = 15 )

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.show()

#Value save
running_loss_index = pd.Series(running_loss_index)
running_loss_index.to_csv('running_loss_index.csv', index=False)
running_loss = pd.Series(running_loss)
running_loss.to_csv('running_loss_train.csv', index=False)
running_loss_validation_index = pd.Series(running_loss_validation_index)
running_loss_validation_index.to_csv('running_loss_validation_index.csv', index=False)
running_loss_validation = pd.Series(running_loss_validation)
running_loss_validation.to_csv('running_loss_validation.csv', index=False)
# In[28]:
plt.figure(figsize=(17,9))
sns.set_style("whitegrid") 
plt.suptitle('NanoAntenna - Reflection Spectra - (epochs - {})'.format(epochs), fontsize=15,
         color = 'k', fontweight='bold')     ## giving title on top of all subplots

plt.subplot(232)
plt.title('subplot: B')
# Plot true data
plt.plot(B.inverse_transform(y_train)[:,0], 'mo', markersize = 8, markeredgewidth = 4,
         markerfacecolor = 'None', label = 'y_train')
# Plot predictions
plt.plot(B.inverse_transform(predicted_on_X_train)[:,0], 'bo', markersize = 8, markeredgewidth = 4,
         markerfacecolor = 'None', label = 'predicted_on_X_train')
# Legend and plot
plt.legend(loc = 'best', fontsize = 10)
plt.tick_params(axis = "both", labelsize=12)
plt.title('True vs Predicted 4(b)', fontsize=15, color = 'k', fontweight ='bold' )
plt.xlabel('True values on training', fontsize=15)
plt.ylabel('Predicted values', fontsize=15 )
plt.tick_params(axis = "both", labelsize=12)

# plt.figure()
plt.subplot(233)
plt.title('subplot: C')
# Plot true data
plt.plot(B.inverse_transform(y_validation)[:,0], 'mo', markersize = 8,markeredgewidth = 4,
         markerfacecolor = 'None',  label = 'y_validation')
# Plot predictions
plt.plot(B.inverse_transform(predicted_on_X_validation)[:,0], 'bo', markersize = 8, markeredgewidth = 4,
         markerfacecolor = 'None', label = 'predicted_on_X_validation')
# Legend and plot
plt.legend(loc = 'best', fontsize = 10)
plt.title('True vs Predicted 4(c)', fontsize = 15, color = 'k', fontweight='bold' )
plt.xlabel('True values on validation', fontsize = 15)
plt.ylabel('Predicted values', fontsize = 15 )
plt.tick_params(axis = "both", labelsize = 12)

# In[29]:
plt.figure(figsize= (9,9))
sns.set_style("white") 
# plt.figure()
# Plot true data Dataset 1
x = [400, 410, 420, 430, 440, 450, 460, 470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,
     620,630,640,650, 660,670,680,690,700,710,720,730,740,750,760,770,780,790,800,810,820,830,840,
     850,860,870,880,890,900,910,920,930,940,950,960,970,980,990,1000]

ya_test = (B.inverse_transform(y_test)[:,0]-min(B.inverse_transform(y_test)[:,0]))/\
    (max(B.inverse_transform(y_test)[:,0])-min(B.inverse_transform(y_test)[:,0]))
yp_test = (B.inverse_transform(predicted_on_y_test)[:,0]-min(B.inverse_transform(predicted_on_y_test)[:,0]))/\
    (max(B.inverse_transform(predicted_on_y_test)[:,0])-min(B.inverse_transform(predicted_on_y_test)[:,0]))

plt.plot(x, ya_test, 'magenta', linewidth = '6', label = 'Actual Reflection Spectra')
plt.plot(x, yp_test,'blue',linewidth = '6', label = 'Predicted Reflection Spectra')
# Legend and plot
plt.legend(loc="upper left", fontsize = 15)
plt.tick_params(axis = "both", labelsize = 25)
plt.title('True vs Predicted Reflection Spectra', fontsize = 25, color = 'k', fontweight ='bold' )
plt.xlabel('Wavelength (nm)', fontsize = 25, fontweight ='bold' )
plt.ylabel('Reflection Spectra', fontsize = 25, fontweight ='bold')

#Value save
xx = pd.Series(x)
xx.to_csv('Wavelength at 5000 epoch 5 Hd 50 Nodes.csv', index=False)
yy1 = pd.Series(ya_test)
yy1.to_csv('Original Reflection Spectra at 5000 epoch 5 Hd 50 Nodes.csv', index=False)
yy = pd.Series(yp_test)
yy.to_csv('Predicted Reflection Spectra at 5000 epoch 5 Hd 50 Nodes.csv', index=False) 
# In[29]:
plt.figure(figsize= (9,9))
sns.set_style("white") 
# Plot true data Dataset 2
x1 = [400, 410, 420, 430, 440, 450, 460, 470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,
     620,630,640,650, 660,670,680,690,700,710,720,730,740,750,760,770,780,790,800,810,820,830,840,
     850,860,870,880,890,900,910,920,930,940,950,960,970,980,990,1000]

y1a_test = (B.inverse_transform(y_test)[:,1]-min(B.inverse_transform(y_test)[:,1]))/\
    (max(B.inverse_transform(y_test)[:,1])-min(B.inverse_transform(y_test)[:,1]))

y1p_test = (B.inverse_transform(predicted_on_y_test)[:,1]-min(B.inverse_transform(predicted_on_y_test)[:,1]))/\
    (max(B.inverse_transform(predicted_on_y_test)[:,1])-min(B.inverse_transform(predicted_on_y_test)[:,1]))
    
    
plt.plot(x1, y1a_test, 'magenta', linewidth = '6',label = 'Actual Transmission Spectra')
plt.plot(x1, y1p_test, 'blue',linewidth = '6',label ='Predicted Transmission Spectra')
# Legend and plot
plt.legend(loc = 'best', fontsize = 15)
plt.tick_params(axis = "both", labelsize = 25)
plt.title('True vs Predicted Transmission Spectra', fontsize = 25, color = 'k', fontweight ='bold' )
plt.xlabel('Wavelength (nm)', fontsize = 25, fontweight ='bold' )
plt.ylabel('Transmission Spectra', fontsize = 25, fontweight ='bold' )

#Value save
xx = pd.Series(x1)
xx.to_csv('Wavelength at 5000 epoch 5 Hd 50 Nodes.csv', index=False)
yy1 = pd.Series(y1a_test)
yy1.to_csv('Original Transmission Spectra at 5000 epoch 5 Hd 50 Nodes.csv', index=False)
yy = pd.Series(y1p_test)
yy.to_csv('Predicted Transmission Spectra at 5000 epoch 5 Hd 50 Nodes.csv', index=False) 

# In[29]:
plt.figure(figsize= (9,9))
sns.set_style("white") 
# Plot true data Dataset 2
x2 = [400, 410, 420, 430, 440, 450, 460, 470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,
     620,630,640,650, 660,670,680,690,700,710,720,730,740,750,760,770,780,790,800,810,820,830,840,
     850,860,870,880,890,900,910,920,930,940,950,960,970,980,990,1000]

y2a_test = (B.inverse_transform(y_test)[:,2]-min(B.inverse_transform(y_test)[:,2]))/\
    (max(B.inverse_transform(y_test)[:,2])-min(B.inverse_transform(y_test)[:,2]))

y2p_test = (B.inverse_transform(predicted_on_y_test)[:,2]-min(B.inverse_transform(predicted_on_y_test)[:,2]))/\
    (max(B.inverse_transform(predicted_on_y_test)[:,2])-min(B.inverse_transform(predicted_on_y_test)[:,2]))
    
    
plt.plot(x2, y2a_test, 'magenta', linewidth = '6',label = 'Actual Transmission Spectra')
plt.plot(x2, y2p_test, 'blue',linewidth = '6',label ='Predicted Transmission Spectra')
# Legend and plot
plt.legend(loc = 'best', fontsize = 15)
plt.tick_params(axis = "both", labelsize = 25)
plt.title('True vs Predicted Absorption Spectra', fontsize = 25, color = 'k', fontweight ='bold' )
plt.xlabel('Wavelength (nm)', fontsize = 25, fontweight ='bold' )
plt.ylabel('Absorption Spectra', fontsize = 25, fontweight ='bold' )

#Value save
xx = pd.Series(x2)
xx.to_csv('Wavelength at 5000 epoch 5 Hd 50 Nodes.csv', index=False)
yy1 = pd.Series(y2a_test)
yy1.to_csv('Original Absorption Spectra at 5000 epoch 5 Hd 50 Nodes.csv', index=False)
yy = pd.Series(y2p_test)
yy.to_csv('Predicted Absorption Spectra at 5000 epoch 5 Hd 50 Nodes.csv', index=False) 
# In[30]:
plt.figure(figsize=(52,9))
sns.set_style("white") 

# plt.figure()
plt.subplot(235)
plt.title('subplot: E')
true_values = B.inverse_transform(y_test)[:,0]
predicted_values = B.inverse_transform(predicted_on_X_test)[:,0]
x_index = [i for i in range(len(true_values))] # 
yerr = error_values = predicted_values - true_values
ax = sns.barplot(x = x_index, y = true_values, yerr = error_values, palette = 'PuRd')
plt.tick_params(axis = "both", labelsize = 15)
plt.title('Error values', fontsize = 24, color = 'k', fontweight ='bold' )
plt.xlabel('x - index', fontsize = 21)
plt.ylabel('True values', fontsize = 21)

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.show()
# In[31]:


plt.figure(figsize= (7,7))
sns.set_style("white") 
# plt.figure()
true_values = (B.inverse_transform(y_test)[:,0]-min(B.inverse_transform(y_test)[:,0]))/\
    (max(B.inverse_transform(y_test)[:,0])-min(B.inverse_transform(y_test)[:,0]))
predicted_values = (B.inverse_transform(predicted_on_y_test)[:,0]-min(B.inverse_transform(predicted_on_y_test)[:,0]))/\
    (max(B.inverse_transform(predicted_on_y_test)[:,0])-min(B.inverse_transform(predicted_on_y_test)[:,0]))
error = ((predicted_values - true_values)/predicted_values)*100

plt.errorbar(x, y = predicted_values, yerr = error_values, color = 'blue',
             ecolor = 'lightgrey', elinewidth = 8, capsize = 0, ms = 18, linewidth = 4);

plt.errorbar(x, y = true_values, yerr = error_values, color = 'magenta',
             ecolor = 'lightgrey', elinewidth = 8, capsize = 0, ms = 18, linewidth = 4);

plt.legend(loc = 'best', fontsize = 21)
plt.tick_params(axis = "both", labelsize = 21)
plt.title('Error value', fontsize=34, color = 'k', fontweight ='bold' )
plt.xlabel('Wavelength (nm)', fontsize=31, color = 'k', fontweight ='bold')
plt.ylabel('Reflection Spectra', fontsize=31, color = 'k', fontweight ='bold')
# In[32]:

sns.set_style("white") 
# plt.figure()

xx = (B.inverse_transform(y_train)[:,0]-min(B.inverse_transform(y_train)[:,0]))/\
    (max(B.inverse_transform(y_train)[:,0])-min(B.inverse_transform(y_train)[:,0]))
yy = (B.inverse_transform(predicted_on_X_train)[:,0]-min(B.inverse_transform(predicted_on_X_train)[:,0]))/\
    (max(B.inverse_transform(predicted_on_X_train)[:,0])-min(B.inverse_transform(predicted_on_X_train)[:,0]))  


xx_validation = (B.inverse_transform(y_validation)[:,0]-min(B.inverse_transform(y_validation)[:,0]))/\
    (max(B.inverse_transform(y_validation)[:,0])-min(B.inverse_transform(y_validation)[:,0]))
yy_validation = (B.inverse_transform(predicted_on_X_validation)[:,0]-min(B.inverse_transform(predicted_on_X_validation)[:,0]))/\
    (max(B.inverse_transform(predicted_on_X_validation)[:,0])-min(B.inverse_transform(predicted_on_X_validation)[:,0]))

xx_test = (B.inverse_transform(y_test)[:,0]-min(B.inverse_transform(y_test)[:,0]))/\
    (max(B.inverse_transform(y_test)[:,0])-min(B.inverse_transform(y_test)[:,0]))
yy_test = (B.inverse_transform(predicted_on_y_test)[:,0]-min(B.inverse_transform(predicted_on_y_test)[:,0]))/\
    (max(B.inverse_transform(predicted_on_y_test)[:,0])-min(B.inverse_transform(predicted_on_y_test)[:,0]))

axs = sns.jointplot(xx, yy, color = "m", marker ='o', kind = "reg",  scatter_kws = {"s": 100}, label = 'Training Set',
                    height = 7, ratio = 4, space=0.1)

bxs = axs.ax_joint.scatter(xx_validation, yy_validation, marker ='o', s = 90, facecolors ='none', edgecolors='b', 
                     linewidths = 4, label = 'Validation Set')

axs.ax_joint.scatter(xx_test, yy_test, marker = "o", s = 90, facecolors='none', edgecolors='k', 
                     linewidths = 4 , label = 'Testing Set')

plt.legend(loc = 'best', fontsize=21,frameon=False)
plt.xlabel('True sensitivity (nm/RIU)', fontsize=25, color = 'k', fontweight='bold' )
plt.ylabel('Predicted sensitivity (nm/RIU)', fontsize=25, color = 'k', fontweight='bold'  )
plt.tick_params(axis="both", labelsize=21)
axs.ax_marg_x.set_xlim(0, 1)
axs.ax_marg_y.set_ylim(0, 1)
axs.fig.suptitle('Data point location plot ', fontsize = 24,  color = 'k', fontweight='bold' )

axs.fig.tight_layout()
axs.fig.subplots_adjust(top = .9) 
print()
print("o/p of test set:           \n", (B.inverse_transform(y_test)[:,0]))
print("predicted o/p of test set: \n", (B.inverse_transform(predicted_on_X_test)[:,0]))
print("mse_test_set: ", mean_squared_error(y_test, predicted_on_X_test))

#xx_test = pd.Series(xx_test)
#xx_test.to_csv('xx_test.csv', index=False)
#yy_test = pd.Series(yy_test)
#yy_test.to_csv('yy_test.csv', index=False)

'''
#Value save
xx = pd.Series(xx)
xx.to_csv('xx_sensitivity.csv', index=False)
yy = pd.Series(yy)
yy.to_csv('yy_sensitivity.csv', index=False)

xx_validation = pd.Series(xx_validation)
xx_validation.to_csv('xx_validation_sensitivity.csv', index=False)
yy_validation = pd.Series(yy_validation)
yy_validation.to_csv('yy_validation_sensitivity.csv', index=False)

xx_test = pd.Series(xx_test)
xx_test.to_csv('xx_test_sensitivity.csv', index=False)
yy_test = pd.Series(yy_test)
yy_test.to_csv('yy_test_sensitivity.csv', index=False)
'''
# In[33]:






