# -*- coding: utf-8 -*-
"""Untitled39.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xja4we1RZxsRFRhZUGdCh4pN5wF1QLNB
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
from algorithms_new import PP_Logistic


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

credit       = pd.read_csv('german.data', delimiter=' ')
compas       = pd.read_csv('compas.csv')
adult        = pd.read_csv('Dataset.data', delimiter = ' ', header = None)
reincidencia = pd.read_excel('reincidencia.xlsx')

credit       = preprocess.pre_process_credit(credit)
compas       = preprocess.pre_process_compas(compas)
adult        = preprocess.pre_process_adult(adult)
reincidencia = preprocess.pre_process_reincidencia(reincidencia)

adult_y_0 = adult.loc[adult.iloc[:,-1]==0,:]
adult_y_1 = adult.loc[adult.iloc[:,-1]==1,:]


adult_small = pd.concat((adult_y_1.sample(3000, replace = False, random_state = 0), adult_y_0.sample(3000, replace = False, random_state = 0)))
adult_small.name = 'adult_small'
adult_small.baseline = max((adult_small.label == 1).sum()/len(adult_small), 1-(adult_small.label == 1).sum()/len(adult_small))


from algorithms_new import PP_Logistic

for data_set, sensitive_variable in zip([adult], [adult.columns.get_loc('Gender_Male')]):
  ut.train_test_get_results_grid(data_set,sensitive_variable, PP_Logistic, {'fairness': np.linspace(1/0.1,1/0.99,100)}, repeats = range(1,3))

