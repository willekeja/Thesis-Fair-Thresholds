import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
import ortools
import torch
import torch.nn as nn
import torch.utils.data
import utils_new as ut

sigmoid_inverse = lambda x : torch.log(x/(1-x))

class MLP(nn.Module):
  def __init__(self, D_in, hidden):
    super(MLP,self).__init__()
    self.MLP = nn.Sequential(nn.Linear(D_in, hidden),
                             nn.ReLU(),
                             nn.Linear(hidden, hidden),
                             nn.ReLU(),
                             nn.Linear(hidden, hidden),
                             nn.ReLU(),
                             nn.Linear(hidden, 1),
                             nn.Sigmoid())

  def forward(self, x):
    out = self.MLP(x)
    return(out)



class PP_NN(object):
  def __init__(self, data_set, index_train, index_validate, sensitive_variable):

    batch_size = 100
    data_set_train = ut.dataset_for_preprocessed_features_include_s(data_set.iloc[:index_train,:], sensitive_variable)  
    data_set_valid = ut.dataset_for_preprocessed_features_include_s(data_set.iloc[index_train:index_validate,:], sensitive_variable)
    data_set_train_and_valid = ut.dataset_for_preprocessed_features_include_s(data_set.iloc[:index_validate,:], sensitive_variable)  
    data_set_test  = ut.dataset_for_preprocessed_features_include_s(data_set.iloc[index_validate:,:], sensitive_variable)  
    self.dataloader_train = torch.utils.data.DataLoader(data_set_train, batch_size = batch_size, shuffle = True)
    self.dataloader_train_full = torch.utils.data.DataLoader(data_set_train, batch_size = len(data_set.iloc[:index_train,:]), shuffle = True)
    self.dataloader_valid = torch.utils.data.DataLoader(data_set_valid, batch_size = len(data_set.iloc[index_train:index_validate,:]), shuffle = False)    
    self.dataloader_train_and_valid = torch.utils.data.DataLoader(data_set_train_and_valid, batch_size = batch_size, shuffle = True)
    self.dataloader_train_and_valid_full = torch.utils.data.DataLoader(data_set_train_and_valid, batch_size = len(data_set.iloc[:index_validate,:]), shuffle = True)
    self.dataloader_test = torch.utils.data.DataLoader(data_set_test, batch_size = len(data_set.iloc[index_validate:,:]), shuffle = False)
    
    self.feature_size = data_set.shape[1]-1
  def get_best_model(self, num_trials, hyperparameters):

    def train(model, optimizer, batch):
      X, s, y = batch
      X, s, y = X.cuda(), s.cuda(), y.cuda()
      out     = model(X)
      loss    = criterion(out, y.unsqueeze(1)) 
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      return model, optimizer

    list_of_accuracies = []
    list_of_parameters = []
    list_of_models     = []
    criterion = nn.BCELoss()

    for trial in range(num_trials):
      hidden = np.asscalar(np.random.choice(hyperparameters['hidden'], 1))

      model = MLP(self.feature_size, hidden).cuda()

      optimizer = torch.optim.Adam(model.parameters())

      for i in range(15):
        for batch in self.dataloader_train:
          model, optimizer = train(model, optimizer, batch)
      batch = next(iter(self.dataloader_valid))
      X_valid, s_valid, y_valid = batch
      X_valid, s_valid, y_valid = X_valid.cuda(), s_valid.cuda(), y_valid.cuda()
      out     = model(X_valid)


      list_of_accuracies.append((((out.squeeze()>0.5).int() == y_valid.int()).sum().float()/len(X_valid)).detach().cpu().numpy())
      list_of_parameters.append(hidden)
      list_of_models.append(model)
      

    best_parameter = list_of_parameters[np.argmax(list_of_accuracies)]
    print(best_parameter) 
    #hidden = best_parameter
    model = list_of_models[np.argmax(list_of_accuracies)]
    self.model = model



  def get_thresholds_and_predict(self, fairness):

    batch = next(iter(self.dataloader_train_full))
    X_train, s_train, y_train = batch
    X_train, s_train, y_train = X_train.cuda(), s_train.cuda(), y_train.cuda()
    X_train.requires_grad = True
    out     = self.model(X_train).squeeze()
    predicted_probabilities_train_a = out[s_train == 0].detach().cpu().numpy()
    predicted_probabilities_train_b = out[s_train == 1].detach().cpu().numpy()
    
    threshold_a, threshold_b = ut.get_optimal_thresholds(predicted_probabilities_train_a, predicted_probabilities_train_b, y_train.cpu().numpy()*2-1, (s_train==0).cpu().numpy().astype(bool), (s_train==1).cpu().numpy().astype(bool), fairness)
    
    train_statistics = ut.get_statistics(y_hat = out.squeeze().detach().cpu().numpy(),
                                           y     = y_train.cpu().numpy(),
                                           s     = s_train.cpu().numpy(),
                                           threshold_s_0 = threshold_a,
                                           threshold_s_1 = threshold_b)
    out = sigmoid_inverse(out)
    out.sum().backward()
    X_pos_gradient = X_train.grad
    norm_gradient  = torch.sqrt(torch.sum((X_pos_gradient**2), axis= 1))
    threshold = torch.ones(X_train.size(0)).cuda()
    threshold[s_train == 0] = torch.tensor(threshold_a).cuda()
    threshold[s_train == 1] = torch.tensor(threshold_b).cuda()
    
    oben = torch.abs(sigmoid_inverse(threshold) - out).squeeze()
    db_to_boundary = (oben)/(norm_gradient+ np.finfo(np.float32).tiny)
    train_statistics['distances'] = db_to_boundary
       
      
    batch = next(iter(self.dataloader_valid))
    X_valid, s_valid, y_valid = batch
    X_valid, s_valid, y_valid = X_valid.cuda(), s_valid.cuda(), y_valid.cuda()
    out     = self.model(X_valid).squeeze()

    validation_statistics = ut.get_statistics(y_hat = out.squeeze().detach().cpu().numpy(),
                                           y     = y_valid.cpu().numpy(),
                                           s     = s_valid.cpu().numpy(),
                                           threshold_s_0 = threshold_a,
                                           threshold_s_1 = threshold_b)



    batch = next(iter(self.dataloader_test))
    X_test, s_test, y_test = batch
    X_test, s_test, y_test = X_test.cuda(), s_test.cuda(), y_test.cuda()
    out     = self.model(X_test).squeeze()

    test_statistics = ut.get_statistics(y_hat = out.squeeze().detach().cpu().numpy(),
                                           y     = y_test.cpu().numpy(),
                                           s     = s_test.cpu().numpy(),
                                           threshold_s_0 = threshold_a,
                                           threshold_s_1 = threshold_b)

    return({'train_statistics' : train_statistics, 'validation_statistics': validation_statistics, 'test_statistics': test_statistics})
