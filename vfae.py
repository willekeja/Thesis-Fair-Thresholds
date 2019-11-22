
import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd
import numpy as np
import utils_new as ut

sigmoid_inverse = lambda x : torch.log(x/(1-x))

class dataset_for_preprocessed_vfae(torch.utils.data.Dataset):
  def __init__(self, data_set, sensitive_variable):
    self.tensor_data = torch.from_numpy(data_set.values.astype(np.float32))
    self.sensitive_variable = sensitive_variable
  def __len__(self):
    return len(self.tensor_data)

  def __getitem__(self, idx):
    feature_cols       = [x for x in range(self.tensor_data.size()[1]) if x != self.tensor_data.size()[1]-1 and  x!= self.sensitive_variable]
    features           = self.tensor_data[idx, feature_cols]
    sensitive_variable = self.tensor_data[idx, self.sensitive_variable]
    labels             = self.tensor_data[idx, -1]
    return features, sensitive_variable, labels



class dist(nn.Module):
  def __init__(self, in_dim, hidden, z):
    super(dist, self).__init__()
    self.hidden = nn.Sequential(nn.Linear(in_dim, hidden),
                                nn.ReLU())      
    self.mu_l     = nn.Linear(hidden, z)
    self.logsigma_l = nn.Linear(hidden, z)

                                  
  def forward(self, inputs):
    hidden = self.hidden(inputs)  
    mu     = self.mu_l(hidden)
    logsigma  = self.logsigma_l(hidden)
    return(mu, logsigma)
  
  
def reparameterize(mu, logsigma):
  eps              = torch.empty_like(logsigma).normal_() 
  eps              = eps.mul((logsigma*0.5)).exp().add(mu)
  return(eps)  



class vfae(nn.Module):
  def __init__(self, d_x, hidden, z):
    super(vfae, self).__init__()
    self.q_z_1   = dist(d_x+1, hidden, z)
    self.q_y     = nn.Sequential(nn.Linear(z, 1),
                                 nn.Sigmoid())
    self.q_z_2   = dist(z+1, hidden, z)
    self.p_z_1   = dist(z+1, hidden, z)
    self.p_x     = nn.Sequential(nn.Linear(z +1, hidden),
                                 nn.ReLU(),
                                 nn.Linear(hidden, d_x),
                                 nn.Sigmoid())


  def forward(self, x,s,y):
    if self.training:
      mu_q_z_1, logsigma_q_z_1 = self.q_z_1(torch.cat((x,s.unsqueeze(1)),1))
      q_z_1_sample  = reparameterize(mu_q_z_1, logsigma_q_z_1)
      y_recon       = self.q_y(q_z_1_sample)                             
      mu_q_z_2, logsigma_q_z_2 =self.q_z_2(torch.cat((q_z_1_sample,y_recon),1))
      q_z_2_sample  = reparameterize(mu_q_z_2, logsigma_q_z_2)
      mu_p_z_1, logsigma_p_z_1 = self.p_z_1(torch.cat((q_z_2_sample,y),1))
      p_z_1_sample  = reparameterize(mu_p_z_1, logsigma_p_z_1)
      x_recon       = self.p_x(torch.cat((p_z_1_sample,s.unsqueeze(1)),1))

      return({'mu_q_z_1': mu_q_z_1, 'logsigma_q_z_1': logsigma_q_z_1, 'y_recon': y_recon, 'mu_q_z_2': mu_q_z_2 , 'logsigma_q_z_2': logsigma_q_z_2, 
            'mu_p_z_1':mu_p_z_1, 'logsigma_p_z_1': logsigma_p_z_1, 'x_recon': x_recon, 'q_z_1_sample': q_z_1_sample})  
    else:
      mu_q_z_1, logsigma_q_z_1 = self.q_z_1(torch.cat((x,s.unsqueeze(1)),1))
      q_z_1_sample  = reparameterize(mu_q_z_1, logsigma_q_z_1)
      y_recon       = self.q_y(q_z_1_sample)
      return y_recon, q_z_1_sample
# mmd taken from here: https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd
  
def dkl_1_2(mu_1, sigma_1, mu_2, sigma_2):
  return(1/2 * (torch.sum(sigma_2.log(), dim=0) - torch.sum(sigma_1.log(), dim = 0) - mu_1.size(0) + torch.sum((sigma_1/sigma_2), dim = 0) + torch.sum(((mu_1 - mu_2)**2 / sigma_2), dim = 0)))



    
def loss_function(x_recon, y_recon, mu_q_z_1, logsigma_q_z_1, mu_q_z_2, logsigma_q_z_2, mu_p_z_1, logsigma_p_z_1, x, y, s0, s1, l1_penalty, l2_penalty, mmd_penalty, variational_alpha = 1, **kwargs):  
  recon_loss_x = nn.MSELoss(reduction = 'sum')(x_recon, x) 
  kldivergence_l_2 = dkl_1_2(mu_q_z_2, logsigma_q_z_2.exp(), torch.zeros(mu_q_z_2.size(0), mu_q_z_2.size(1)).cuda(), torch.ones(mu_q_z_2.size(0), mu_q_z_2.size(1)).cuda()).sum()
  kldivergence_l_1 = dkl_1_2(mu_q_z_1, logsigma_q_z_1.exp(), mu_p_z_1, logsigma_p_z_1.exp()).sum()
  mmd = compute_mmd(s0, s1)    
  recon_loss_y = nn.BCELoss(reduction = 'sum')(y_recon, y)
  return  1 * recon_loss_x +  recon_loss_y  + kldivergence_l_1 +  kldivergence_l_2 + mmd_penalty * mmd


 
class VFAE(object):
  def __init__(self, data_set, index_train, index_validate, sensitive_variable, hyperparameters):
    
    self.hyperparameters = hyperparameters
    self.batch_size = 100
    data_set_train = dataset_for_preprocessed_vfae(data_set.iloc[:index_train,:], sensitive_variable)  
    data_set_valid = dataset_for_preprocessed_vfae(data_set.iloc[index_train:index_validate,:], sensitive_variable)
    data_set_train_and_valid = dataset_for_preprocessed_vfae(data_set.iloc[:index_validate,:], sensitive_variable)  
    data_set_test  = dataset_for_preprocessed_vfae(data_set.iloc[index_validate:,:], sensitive_variable)  
    self.dataloader_train = torch.utils.data.DataLoader(data_set_train, batch_size = self.batch_size, shuffle = True)
    self.dataloader_train_full = torch.utils.data.DataLoader(data_set_train, batch_size = len(data_set.iloc[:index_train,:]), shuffle = True)
    self.dataloader_valid = torch.utils.data.DataLoader(data_set_valid, batch_size = len(data_set.iloc[index_train:index_validate,:]), shuffle = False)    
    self.dataloader_train_and_valid = torch.utils.data.DataLoader(data_set_train_and_valid, batch_size = self.batch_size, shuffle = True)
    self.dataloader_train_and_valid_full = torch.utils.data.DataLoader(data_set_train_and_valid, batch_size = len(data_set.iloc[:index_validate,:]), shuffle = True)
    self.dataloader_test = torch.utils.data.DataLoader(data_set_test, batch_size = len(data_set.iloc[index_validate:,:]), shuffle = False)

    self.x_shape = data_set.shape[1]-2
  def fit_predict(self):
    
    batch_size  = self.batch_size    
   
    hidden      = np.asscalar(np.random.choice(self.hyperparameters['hidden'],1))
    z           = np.asscalar(np.random.choice(self.hyperparameters['z'],1))
    weight      = np.asscalar(np.random.uniform(self.hyperparameters['weight_lower'],self.hyperparameters['weight_upper'], 1))
    lr          = np.asscalar(np.random.uniform(0.004, 0.009, 1))
    model = vfae(self.x_shape, hidden, z).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(self.hyperparameters['epochs']):
      for batch in self.dataloader_train:

        X, s, y = batch
        X, s, y = X.cuda(), s.cuda(), y.cuda()
        returns =     model(X,s,y.unsqueeze(1))
        loss    = loss_function(x = X, y= y, s0 = returns['q_z_1_sample'][s==0], s1 = returns['q_z_1_sample'][s==1], l1_penalty = 1, l2_penalty = 1, mmd_penalty = weight  * batch_size, variational_alpha = 1, **returns)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    model.eval()
    batch = next(iter(self.dataloader_train_full))
    X_train, s_train, y_train = batch
    X_train, s_train, y_train = X_train.cuda(), s_train.cuda(), y_train.cuda()
    X_train.requires_grad = True
    s_train.requires_grad = True
    y_recon_train, z_1_train  = model(X_train,s_train,y_train)
    train_statistics = ut.get_statistics(y_hat = y_recon_train.squeeze().detach().cpu().numpy(),
                                      y     = y_train.cpu().numpy(),
                                      s  = s_train.detach().cpu().numpy())
   
    y_recon_train = sigmoid_inverse(y_recon_train)
    y_recon_train.sum().backward()
    X_pos_gradient = X_train.grad
    s_pos_gradient = s_train.grad
    pos_gradient   = torch.cat((X_pos_gradient, s_pos_gradient.unsqueeze(1)), 1)
    print('new')
    norm_gradient  = torch.sqrt(torch.sum((pos_gradient**2), axis= 1))
    oben = torch.abs(y_recon_train).squeeze()
    db_to_boundary = oben /(norm_gradient+ np.finfo(np.float32).tiny)
    train_statistics['distances'] = db_to_boundary


    batch = next(iter(self.dataloader_valid))
    X_valid, s_valid, y_valid = batch
    X_valid, s_valid, y_valid = X_valid.cuda(), s_valid.cuda(), y_valid.cuda()
    y_recon_valid, z_1_valid = model(X_valid,s_valid,y_valid)
    
    validation_statistics = ut.get_statistics(y_hat = y_recon_valid.squeeze().detach().cpu().numpy(),
                                           y     = y_valid.cpu().numpy(),
                                           s     = s_valid.cpu().numpy())
    
    batch = next(iter(self.dataloader_test))
    X_test, s_test, y_test = batch
    X_test, s_test, y_test = X_test.cuda(), s_test.cuda(), y_test.cuda()
    y_recon_test, z_1_test = model(X_test,s_test,y_test)
        
    test_statistics       = ut.get_statistics(y_hat = y_recon_test.squeeze().detach().cpu().numpy(),
                                           y     = y_test.cpu().numpy(),
                                           s     = s_test.cpu().numpy())
  
    hyperparameters       = {'hidden': hidden, 'z': z, 'weight': weight, 'lr': lr}
    
    return({'train_statistics'       : train_statistics,
            'validation_statistics' : validation_statistics, 
            'test_statistics'       : test_statistics, 
            'hyperparameters'       : hyperparameters})
  
