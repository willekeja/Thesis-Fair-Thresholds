
import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd
import numpy as np
import utils_new as ut

sigmoid_inverse = lambda x : torch.log(x/(1-x))


class label_classifier(nn.Module):
  
     def __init__(self, D_in, hidden):
         
       super(label_classifier, self).__init__()
          
       self.linear_1  = nn.Linear(D_in, hidden) 
       self.linear_2  = nn.Linear(hidden, hidden)
       self.linear_3  = nn.Linear(hidden, hidden)
       self.linear_4  = nn.Linear(hidden, 1)

 
     def forward(self,x):
       hidden   =  nn.ReLU()(self.linear_3(nn.ReLU()(self.linear_2(nn.ReLU()(self.linear_1(x))))))
       out      =  nn.Sigmoid()(self.linear_4(hidden))
       return(out, hidden)

      
class domain_classifier(nn.Module):
  
     def __init__(self, hidden):
      
       super(domain_classifier, self).__init__()
       self.linear_1 = nn.Linear(hidden, hidden)
       self.linear_2 = nn.Linear(hidden, 1)
        
     def forward(self, x):
       out      = nn.Sigmoid()(self.linear_2(nn.ReLU()(self.linear_1(x))))
       return(out)
    
class label_classifier_linear(nn.Module):
  
  def __init__(self, D_in):
         
    super(label_classifier_linear, self).__init__()

    self.linear_1  = nn.Linear(D_in, 1) 
 

 
  def forward(self,x):
    out      =  nn.Sigmoid()(self.linear_1(x))
    return(out)

      
class domain_classifier_for_linear(nn.Module):
  
  def __init__(self, hidden):
      
    super(domain_classifier_for_linear, self).__init__()
    self.linear_1 = nn.Linear(1, hidden)
    self.linear_2 = nn.Linear(hidden, 1)

  def forward(self, x):
    out      = nn.Sigmoid()(self.linear_2(nn.ReLU()(self.linear_1(x))))
    return(out)
      



class ADV_NN(object):
  def __init__(self, data_set, index_train, index_validate, sensitive_variable, hyperparameters):

    batch_size = 100
    data_set_train = ut.dataset_for_preprocessed_features_include_s(data_set.iloc[:index_train,:], sensitive_variable)  
    data_set_valid = ut.dataset_for_preprocessed_features_include_s(data_set.iloc[index_train:index_validate,:], sensitive_variable)
    data_set_train_and_valid = ut.dataset_for_preprocessed_features_include_s(data_set.iloc[:index_validate,:], sensitive_variable)  
    data_set_test  = ut.dataset_for_preprocessed_features_include_s(data_set.iloc[index_validate:,:], sensitive_variable)  
    self.dataloader_train = torch.utils.data.DataLoader(data_set_train, batch_size = batch_size, shuffle = True)
    self.dataloader_train_full = torch.utils.data.DataLoader(data_set_train, batch_size = len(data_set.iloc[:index_train,:]), shuffle = False)
    self.dataloader_valid = torch.utils.data.DataLoader(data_set_valid, batch_size = len(data_set.iloc[index_train:index_validate,:]), shuffle = False)    
    self.dataloader_train_and_valid = torch.utils.data.DataLoader(data_set_train_and_valid, batch_size = batch_size, shuffle = True)
    self.dataloader_train_and_valid_full = torch.utils.data.DataLoader(data_set_train_and_valid, batch_size = len(data_set.iloc[:index_validate,:]), shuffle = False)
    self.dataloader_test = torch.utils.data.DataLoader(data_set_test, batch_size = len(data_set.iloc[index_validate:,:]), shuffle = False)
    self.features_size = data_set.shape[1]-1
    self.hyperparameters = hyperparameters
    
  def fit_predict(self):
    
    criterion  = nn.BCELoss()

    hidden      = np.asscalar(np.random.choice(self.hyperparameters['hidden'],1))
    weight      = np.asscalar(np.random.uniform(self.hyperparameters['weight_lower'],self.hyperparameters['weight_upper'], 1))
    lr          = np.asscalar(np.random.uniform(0.0005,0.0001, 1))

    model_label = label_classifier(self.features_size , hidden).cuda()
    model_domain = domain_classifier(hidden).cuda()
    optimizer_label = torch.optim.Adam(model_label.parameters(), lr = lr)
    optimizer_domain = torch.optim.Adam(model_domain.parameters(), lr = lr)
    batch_counter = 0
    
    for epoch in range(50):
      for batch in self.dataloader_train:
        X, s, y = batch
        X, s, y = X.cuda(), s.cuda(), y.cuda()
        out_label, out_features  = model_label(X)
        out_domain       = model_domain(out_features)
        loss_domain      = criterion(out_domain, s.unsqueeze(1)) 
        loss_label       = criterion(out_label, y.unsqueeze(1))
        loss_domain.backward(retain_graph = True)
        for parameter in model_label.parameters():
          if parameter.grad is not None:
            parameter.grad *= - weight
        loss_label.backward() 
        if batch_counter % 2 == 0:
          optimizer_label.step()
        else:
          optimizer_domain.step()
        optimizer_label.zero_grad()
        optimizer_domain.zero_grad()
        batch_counter += 1
        
    batch = next(iter(self.dataloader_train_full))
    X_train, s_train, y_train = batch
    X_train, s_train, y_train = X_train.cuda(), s_train.cuda(), y_train.cuda()
    X_train.requires_grad = True

    out_label_train, out_features_train  = model_label(X_train)
    
    train_statistics = ut.get_statistics(y_hat = out_label_train.squeeze().detach().cpu().numpy(),
                                      y     = y_train.cpu().numpy(),
                                      s     = s_train.cpu().numpy())
    out_label_train = sigmoid_inverse(out_label_train)
    out_label_train.sum().backward()
    X_pos_gradient = X_train.grad
    norm_gradient  = torch.sqrt(torch.sum(X_pos_gradient**2, axis= 1))
    oben = torch.abs(out_label_train).squeeze()
    db_to_boundary = (oben)/(norm_gradient+ np.finfo(np.float32).tiny)
    train_statistics['distances'] = db_to_boundary
       
    batch = next(iter(self.dataloader_valid))
    X_valid, s_valid, y_valid = batch
    X_valid, s_valid, y_valid = X_valid.cuda(), s_valid.cuda(), y_valid.cuda()
    out_label_valid, out_features_valid  = model_label(X_valid)
    
    validation_statistics = ut.get_statistics(y_hat = out_label_valid.squeeze().detach().cpu().numpy(),
                                           y     = y_valid.cpu().numpy(),
                                           s     = s_valid.cpu().numpy())
      
    batch = next(iter(self.dataloader_test))
    X_test, s_test, y_test = batch
    X_test, s_test, y_test = X_test.cuda(), s_test.cuda(), y_test.cuda()
    out_label_test, out_features_test  = model_label(X_test)
    test_statistics = ut.get_statistics(y_hat = out_label_test.squeeze().detach().cpu().numpy(),
                                     y     = y_test.cpu().numpy(),
                                     s     = s_test.cpu().numpy())
    hyperparameters = {'hidden': hidden, 'weight': weight, 'lr': lr}
    return({'train_statistics': train_statistics, 'validation_statistics': validation_statistics, 
            'test_statistics': test_statistics, 'hyperparameters': hyperparameters})
  
  
class ADV_Logistic(object):
  def __init__(self, data_set, index_train, index_validate, sensitive_variable, hyperparameters):

    batch_size = 100
    data_set_train = ut.dataset_for_preprocessed_features_include_s(data_set.iloc[:index_train,:], sensitive_variable)  
    data_set_valid = ut.dataset_for_preprocessed_features_include_s(data_set.iloc[index_train:index_validate,:], sensitive_variable)
    data_set_train_and_valid = ut.dataset_for_preprocessed_features_include_s(data_set.iloc[:index_validate,:], sensitive_variable)  
    data_set_test  = ut.dataset_for_preprocessed_features_include_s(data_set.iloc[index_validate:,:], sensitive_variable)  
    self.dataloader_train = torch.utils.data.DataLoader(data_set_train, batch_size = batch_size, shuffle = True)
    self.dataloader_train_full = torch.utils.data.DataLoader(data_set_train, batch_size = len(data_set.iloc[:index_train,:]), shuffle = False)
    self.dataloader_valid = torch.utils.data.DataLoader(data_set_valid, batch_size = len(data_set.iloc[index_train:index_validate,:]), shuffle = False)    
    self.dataloader_train_and_valid = torch.utils.data.DataLoader(data_set_train_and_valid, batch_size = batch_size, shuffle = True)
    self.dataloader_train_and_valid_full = torch.utils.data.DataLoader(data_set_train_and_valid, batch_size = len(data_set.iloc[:index_validate,:]), shuffle = False)
    self.dataloader_test = torch.utils.data.DataLoader(data_set_test, batch_size = len(data_set.iloc[index_validate:,:]), shuffle = False)
    self.features_size = data_set.shape[1]-1

    self.hyperparameters = hyperparameters
  def fit_predict(self):
    
    criterion   = nn.BCELoss()
    hidden      = np.asscalar(np.random.choice(self.hyperparameters['hidden'],1))
    weight      = np.asscalar(np.random.uniform(self.hyperparameters['weight_lower'],self.hyperparameters['weight_upper'], 1))
    lr          = np.asscalar(np.random.uniform(0.0005,0.0001, 1))

    model_label = label_classifier_linear(self.features_size).cuda()
    model_domain = domain_classifier_for_linear(hidden).cuda()
    optimizer_label = torch.optim.Adam(model_label.parameters(), lr = lr)
    optimizer_domain = torch.optim.Adam(model_domain.parameters(), lr = lr)
    batch_counter = 0
    
    for epoch in range(50):
      for batch in self.dataloader_train:
        X, s, y = batch
        X, s, y = X.cuda(), s.cuda(), y.cuda()
        out_label        = model_label(X)
        out_domain       = model_domain(out_label)
        loss_domain      = criterion(out_domain, s.unsqueeze(1)) 
        loss_label       = criterion(out_label, y.unsqueeze(1))
        loss_domain.backward(retain_graph = True)
        for parameter in model_label.parameters():
          if parameter.grad is not None:
            parameter.grad *= - weight
        loss_label.backward() 
        if batch_counter % 2 == 0:
          optimizer_label.step()
        else:
          optimizer_domain.step()
        optimizer_label.zero_grad()
        optimizer_domain.zero_grad()
        batch_counter += 1
        
    batch = next(iter(self.dataloader_train_full))
    X_train, s_train, y_train = batch
    X_train, s_train, y_train = X_train.cuda(), s_train.cuda(), y_train.cuda()
    out_label_train           = model_label(X_train)
    
    train_statistics = ut.get_statistics(y_hat = out_label_train.squeeze().detach().cpu().numpy(),
                                      y     = y_train.cpu().numpy(),
                                      s     = s_train.cpu().numpy())
       
    batch = next(iter(self.dataloader_valid))
    X_valid, s_valid, y_valid = batch
    X_valid, s_valid, y_valid = X_valid.cuda(), s_valid.cuda(), y_valid.cuda()
    out_label_valid           = model_label(X_valid)
    
    validation_statistics = ut.get_statistics(y_hat = out_label_valid.squeeze().detach().cpu().numpy(),
                                           y     = y_valid.cpu().numpy(),
                                           s     = s_valid.cpu().numpy())
      
    batch = next(iter(self.dataloader_test))
    X_test, s_test, y_test = batch
    X_test, s_test, y_test = X_test.cuda(), s_test.cuda(), y_test.cuda()
    out_label_test         = model_label(X_test)
    test_statistics = ut.get_statistics(y_hat = out_label_test.squeeze().detach().cpu().numpy(),
                                     y     = y_test.cpu().numpy(),
                                     s     = s_test.cpu().numpy())
    hyperparameters = {'hidden': hidden, 'weight': weight, 'lr': lr}
    return({'train_statistics': train_statistics, 'validation_statistics': validation_statistics, 
            'test_statistics': test_statistics, 'hyperparameters': hyperparameters})

  
  

  
  