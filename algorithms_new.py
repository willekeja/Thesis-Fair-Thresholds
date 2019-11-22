
import utils_new as ut
import pandas as pd
import numpy as np
import seaborn as sns
import math
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zafar
import torch
import torch.nn as nn
import torch.utils
from ortools.linear_solver import pywraplp
from sklearn.linear_model import LogisticRegression
import pandas as pd
from aif360.algorithms.inprocessing.prejudice_remover import PrejudiceRemover
from aif360.algorithms.inprocessing.meta_fair_classifier import MetaFairClassifier
from aif360.datasets import BinaryLabelDataset
from sklearn.model_selection import KFold


class Lipton(object):
  def __init__(self, data_set, train_index, validation_index, sensitive_variable):
    self.train_features      = data_set.iloc[:train_index,:-1]
    self.validation_features = data_set.iloc[train_index:validation_index,:-1]
    self.test_features       = data_set.iloc[validation_index:,:-1]

    self.train_labels      = data_set.iloc[:train_index,-1]
    self.validation_labels = data_set.iloc[train_index:validation_index,-1]
    self.test_labels       = data_set.iloc[validation_index:,-1]
    
    self.index_s0_train    = self.train_features.iloc[:,sensitive_variable] == 0
    self.index_s1_train    = self.train_features.iloc[:,sensitive_variable] == 1  
    self.index_s0_validate = self.validation_features.iloc[:,sensitive_variable] == 0
    self.index_s1_validate = self.validation_features.iloc[:,sensitive_variable] == 1  
    self.index_s0_test     = self.test_features.iloc[:,sensitive_variable] == 0
    self.index_s1_test     = self.test_features.iloc[:,sensitive_variable] == 1
    self.sensitive_variable = sensitive_variable
  
  def fit_predict(self, p):
    
    # pos rate b < pos rate a

  
    logfit = LogisticRegression().fit( self.train_features, self.train_labels)
    
    yhat_train = logfit.predict_proba(self.train_features)[:,1]
    
    df = pd.DataFrame({'yhat_train': yhat_train, 'y': self.train_labels})
    
    n_s_0 = len(df.loc[self.index_s0_train, :])
    n_s_1 = len(df.loc[self.index_s1_train, :])
    
    pos_rate_s0 = (yhat_train[self.index_s0_train]>0.5).sum().astype(float) / n_s_0
    pos_rate_s1 = (yhat_train[self.index_s1_train]>0.5).sum().astype(float) / n_s_1
    
    if pos_rate_s0 >= pos_rate_s1:
      b = 's_1'
      a = 's_0'
      n_b = n_s_1
      n_a = n_s_0
      
      index_b_train = self.index_s1_train
      index_a_train = self.index_s0_train
      index_b_valid = self.index_s1_validate
      
      index_a_valid = self.index_s0_validate
      index_b_test = self.index_s1_test
      index_a_test = self.index_s0_test
      
    else:
      b = 's_0'
      a = 's_1'
      n_b = n_s_0
      n_a = n_s_1
      index_b_train = self.index_s0_train
      index_a_train = self.index_s1_train
      
      index_b_valid = self.index_s0_validate
      index_a_valid = self.index_s1_validate
      
      index_b_test = self.index_s0_test
      index_a_test = self.index_s1_test
   
    df_a = df.loc[index_a_train,:].reset_index(drop = True)
    df_a_sorted = df_a.sort_values('yhat_train', ascending = False).reset_index(drop = True)
    
    df_b = df.loc[index_b_train,:].reset_index(drop = True)
    df_b_sorted = df_b.sort_values('yhat_train', ascending = False).reset_index(drop = True)
    
    group_a = df_a_sorted.loc[df_a_sorted.yhat_train <= 0.5, :].reset_index(drop = True)
    group_b = df_b_sorted.loc[df_b_sorted.yhat_train > 0.5, :].reset_index(drop = True)

    score_b = 1/(n_a - 2*n_a*group_a.yhat_train)
    score_a = p/(200*n_b * group_b.yhat_train - 100*n_b)
    
    group_a['score_a'] = score_a
    group_b['score_b'] = score_b

    group_a_sorted = group_a.sort_values('score_a', ascending = False).reset_index(drop = True)
    group_b_sorted = group_b.sort_values('score_b', ascending = False).reset_index(drop = True)

    group_b_pos = len(group_b) 
    group_a_pos = len(df_a_sorted.loc[df_a_sorted.yhat_train > 0.5, :])
    
    p_estimate = (group_b_pos /n_b)/ (group_a_pos/n_a)
    
    while p_estimate*100 < p:
      if group_b_sorted.score_b[0] >=  group_a_sorted.score_a[0]:
        group_b_pos += 1
      else:
        group_a_pos -= 1

      p_estimate = (group_b_pos /n_b)/ (group_a_pos/n_a)

    if group_b_sorted.score_b[0] >=  group_a_sorted.score_a[0]:
      group_b_pos += 1
    else:
      group_a_pos -= 1
   
    threshold_a = df_a_sorted.yhat_train[group_a_pos-1]

    threshold_b = df_b_sorted.yhat_train[group_b_pos-1]
    
    if a == 's_0':
      threshold_s_0 = threshold_a
      threshold_s_1 = threshold_b
    else:
      threshold_s_0 = threshold_b
      threshold_s_1 = threshold_a

      
    train_statistics = ut.get_statistics(y_hat = yhat_train,
                                      y     = self.train_labels,
                                      s     = self.train_features.iloc[:,self.sensitive_variable],
                                      threshold_s_0 = threshold_s_0,
                                      threshold_s_1 = threshold_s_1)

    y_hat_validate = logfit.predict_proba(self.validation_features)[:,1]
    
    validation_statistics = ut.get_statistics(y_hat = y_hat_validate,
                                  y     = self.validation_labels,
                                  s     = self.validation_features.iloc[:,self.sensitive_variable],
                                  threshold_s_0 = threshold_s_0,
                                  threshold_s_1 = threshold_s_1)
    
    y_hat_test = logfit.predict_proba(self.test_features)[:,1]
    
    test_statistics = ut.get_statistics(y_hat = y_hat_test,
                                  y     = self.test_labels,
                                  s     = self.test_features.iloc[:,self.sensitive_variable],
                                  threshold_s_0 = threshold_s_0,
                                  threshold_s_1 = threshold_s_1)
    
    hyperparameters = {'fairness' : p}

    return({'train_statistics': train_statistics, 'validation_statistics': validation_statistics, 'test_statistics': test_statistics, 'hyperparameters': hyperparameters})
    



class PP_Logistic(object):       
  def __init__(self, data_set, train_index, validation_index, sensitive_variable):
    self.train_features      = data_set.iloc[:train_index,:-1]
    self.validation_features = data_set.iloc[train_index:validation_index,:-1]
    self.test_features       = data_set.iloc[validation_index:,:-1]

    self.train_labels      = data_set.iloc[:train_index,-1]
    self.validation_labels = data_set.iloc[train_index:validation_index,-1]
    self.test_labels       = data_set.iloc[validation_index:,-1]
    
    self.sensitive_variable = sensitive_variable

  def fit_predict(self, fairness):
            
    # perform logistic regression 
    logfit = LogisticRegression().fit(self.train_features,self.train_labels)
   
    # get optimal thresholds for group a (s0) and group b (s1)

    y_hat_train = logfit.predict_proba(self.train_features)[:,1]
    predicted_probabilities_train_a = y_hat_train[self.train_features.iloc[:,self.sensitive_variable] == 0]
    predicted_probabilities_train_b = y_hat_train[self.train_features.iloc[:,self.sensitive_variable] == 1]
    
    threshold_a, threshold_b = ut.get_optimal_thresholds(predicted_probabilities_train_a, predicted_probabilities_train_b, (self.train_labels *2-1).to_numpy(), 
                                                         (self.train_features.iloc[:,self.sensitive_variable] == 0).to_numpy(), (self.train_features.iloc[:,self.sensitive_variable] == 1).to_numpy(), fairness)
    train_statistics = ut.get_statistics(y_hat = y_hat_train,
                                      y     = self.train_labels,
                                      s     = self.train_features.iloc[:,self.sensitive_variable],
                                      threshold_s_0 = threshold_a,
                                      threshold_s_1 = threshold_b)

    y_hat_validate = logfit.predict_proba(self.validation_features)[:,1]

    validation_statistics = ut.get_statistics(y_hat = y_hat_validate,
                                       y     = self.validation_labels,
                                       s     = self.validation_features.iloc[:,self.sensitive_variable],
                                       threshold_s_0 = threshold_a,
                                       threshold_s_1 = threshold_b)


    y_hat_test = logfit.predict_proba(self.test_features)[:,1]

    test_statistics = ut.get_statistics(y_hat = y_hat_test,
                                       y     = self.test_labels,
                                       s     = self.test_features.iloc[:,self.sensitive_variable],
                                       threshold_s_0 = threshold_a,
                                       threshold_s_1 = threshold_b)
    

    hyperparameters = {'fairness' : fairness}

    return({'train_statistics': train_statistics, 'validation_statistics': validation_statistics, 'test_statistics': test_statistics, 'hyperparameters': hyperparameters})
sigmoid = lambda x:  1/(1+np.exp(-x))

class Zafar(object):
  def __init__(self, data_set, train_index, validation_index, sensitive_variable):
            
    self.train_features      = data_set.iloc[:train_index,:-1]
    self.validation_features = data_set.iloc[train_index:validation_index,:-1]
    self.test_features       = data_set.iloc[validation_index:,:-1]

    self.train_labels      = data_set.iloc[:train_index,-1]*2-1
    self.validation_labels = data_set.iloc[train_index:validation_index,-1]*2-1
    self.test_labels       = data_set.iloc[validation_index:,-1]*2-1

    self.train_features      = zafar.add_intercept(self.train_features)
    self.validation_features = zafar.add_intercept(self.validation_features)
    self.test_features       = zafar.add_intercept(self.test_features)
    
    self.sensitive_variable  = sensitive_variable +1

  def fit_predict(self, fairness):
         
    params  = {'apply_fairness_constraints'    : 1,
               'apply_accuracy_constraint'     : 0,
               'sep_constraint'                : 0,
               'x'                             : self.train_features,
               'y'                             : self.train_labels, 
               'x_control'                     : {'s1': self.train_features[:,self.sensitive_variable]}, 
               'sensitive_attrs'               : ['s1'], 
               'sensitive_attrs_to_cov_thresh' : {'s1':fairness},
               'gamma'                         : None}       

         
    w = zafar.train_model(**params)
    
    train_predicts = np.matmul(self.train_features, w)
    
    train_statistics  = ut.get_statistics(y_hat = sigmoid(train_predicts), y = (self.train_labels+1)/2, s = self.train_features[:,self.sensitive_variable])
    
    validation_predicts = np.matmul(self.validation_features, w)
    
    validation_statistics  = ut.get_statistics(y_hat = sigmoid(validation_predicts), y = (self.validation_labels+1)/2, s = self.validation_features[:,self.sensitive_variable])
             
    test_predicts = np.matmul(self.test_features, w)
    
    test_statistics  = ut.get_statistics(y_hat = sigmoid(test_predicts), y = (self.test_labels+1)/2, s = self.test_features[:,self.sensitive_variable])
    
    hyperparameters = {'fairness': fairness}
    
    return({'train_statistics': train_statistics, 'validation_statistics': validation_statistics, 'test_statistics': test_statistics, 
             'hyperparameters': hyperparameters})


class aif360_classifier(object):
  def __init__(self, data_set, index_train, index_validate, sensitive_variable):

    self.sensitive_variable_string = list(data_set.columns)[sensitive_variable]

    self.s_train    = data_set.iloc[:index_train,sensitive_variable]
    self.s_validate = data_set.iloc[index_train:index_validate,sensitive_variable]
    self.s_test     = data_set.iloc[index_validate:,sensitive_variable]

    self.y_train    = data_set.iloc[:index_train,-1]
    self.y_validate = data_set.iloc[index_train:index_validate,-1]
    self.y_test     = data_set.iloc[index_validate:,-1]




    self.train = BinaryLabelDataset(df = data_set.iloc[:index_train,:],
                            label_names=['label'],
                            protected_attribute_names=[self.sensitive_variable_string],
                            favorable_label=1,
                            unfavorable_label=0)

    self.validate = BinaryLabelDataset(df = data_set.iloc[index_train:index_validate,:],
                            label_names=['label'],
                            protected_attribute_names=[self.sensitive_variable_string],
                            favorable_label=1,
                            unfavorable_label=0)



    self.test = BinaryLabelDataset(df = data_set.iloc[index_validate:,:] ,
                            label_names=['label'],
                            protected_attribute_names=[self.sensitive_variable_string],
                            favorable_label=1,
                            unfavorable_label=0)
    
class Kamishima(aif360_classifier):
  
  def fit_predict(self, fairness):
      
    model = PrejudiceRemover(class_attr = 'label', sensitive_attr = self.sensitive_variable_string, eta = fairness)

    model.fit(self.train)

    predicts_train    = np.squeeze(model.predict(self.train).labels)
    
    train_statistics  = ut.get_statistics(y_hat = predicts_train, y = self.y_train, s = self.s_train)
    
    predicts_validate = np.squeeze(model.predict(self.validate).labels)
      
    validation_statistics  = ut.get_statistics(y_hat = predicts_validate, y = self.y_validate, s = self.s_validate)
      
    predicts_test = np.squeeze(model.predict(self.test).labels)
     
    test_statistics  = ut.get_statistics(y_hat = predicts_test, y = self.y_test, s = self.s_test)
    
    hyperparameters = {'fairness' : fairness}
      
    return({'train_statistics': train_statistics, 'validation_statistics': validation_statistics, 'test_statistics': test_statistics, 'hyperparameters': hyperparameters})



