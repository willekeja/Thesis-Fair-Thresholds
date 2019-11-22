# -*- coding: utf-8 -*-
"""Copy of Copy of Copy of Copy of Copy of Copy of Copy of Copy of Copy of Copy of Copy of Copy of first_results_nn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ufvpy207vrtQ0-TSWbkDH6RU6cOBlSgX
"""

import preprocess 
import utils_new as ut
from vfae import VFAE
from post_process_nn import PP_NN
from adversarial import ADV_NN
from algorithms_new import Zafar
from algorithms_new import Kamishima
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.utils.data
import warnings
from ortools.linear_solver import pywraplp

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

credit       = pd.read_csv('german.data', delimiter=' ')
compas       = pd.read_csv('compas.csv')
adult        = pd.read_csv('Dataset.data', delimiter = ' ', header = None)

credit       = preprocess.pre_process_credit(credit)
compas       = preprocess.pre_process_compas(compas)
adult        = preprocess.pre_process_adult(adult)

adult_y_0 = adult.loc[adult.iloc[:,-1]==0,:]
adult_y_1 = adult.loc[adult.iloc[:,-1]==1,:]


adult_small = pd.concat((adult_y_1.sample(3000, replace = False, random_state = 0), adult_y_0.sample(3000, replace = False, random_state = 0)))
adult_small.name = 'adult_small'
adult_small.baseline = max((adult_small.label == 1).sum()/len(adult_small), 1-(adult_small.label == 1).sum()/len(adult_small))


#hyperparameters = {'hidden': np.arange(8,32)}
#for data_set, sensitive_variable in zip([adult_small,compas], [adult_small.columns.get_loc('Gender_Male'), compas.columns.get_loc('race_factor_Caucasian')]):    
  #ut.train_test_get_results_pp(data_set, sensitive_variable, hyperparameters, repeats = 5)
  #print('done')
hyperparameters = {'hidden': np.arange(8,64)}
for data_set, sensitive_variable in zip([adult_small, compas, credit], [adult_small.columns.get_loc('Gender_Male'), compas.columns.get_loc('race_factor_Caucasian'),  credit.columns.get_loc('age_old')]):
  ut.train_test_get_results_pp(data_set, sensitive_variable, hyperparameters, repeats = 5)
  print('done')
  