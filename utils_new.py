import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
import ortools
import os
import torch
import torch.nn as nn
import torch.utils.data
from post_process_nn import PP_NN
def train_test_get_results_grid(data_set, sensitive_variable, classifier, hyperparameters, repeats):

  for repeat in repeats:
    data = data_set.sample(frac=1, replace=False, random_state = repeat)
    index_train    = np.floor(data_set.shape[0]*0.6).astype(int)
    index_validate = np.floor(data_set.shape[0]*0.8).astype(int)
    results_dict = {}
    for i,fairness in enumerate(hyperparameters['fairness']):
      print(fairness)
      model = classifier(data, index_train, index_validate, sensitive_variable)   
      results_dict[i] = model.fit_predict(fairness)
      print('accuracy', results_dict[i]['test_statistics']['accuracy'], 'fairness', results_dict[i]['test_statistics']['fairness'])
    try:
      os.makedirs(f'final_result_dictionaries/{model.__class__.__name__}/{data_set.name}/')
    except FileExistsError:
      pass
    torch.save({'data_set': data_set.name, 'random_state': repeat, 'algorithm': model.__class__.__name__, 'results': results_dict, 'hyperparameters': hyperparameters}, 
               f'final_result_dictionaries/{model.__class__.__name__}/{data_set.name}/{repeat}.tar')
    
def train_test_get_results(data_set, sensitive_variable, classifier, hyperparameters, repeats):
  for repeat in repeats:
    data = data_set.sample(frac=1, replace=False, random_state = repeat)
    index_train    = np.floor(data_set.shape[0]*0.6).astype(int)
    index_validate = np.floor(data_set.shape[0]*0.8).astype(int)
    print(data_set.name)

    results_dict = {}
    for i in range(100):
      model = classifier(data, index_train, index_validate, sensitive_variable, hyperparameters)   
      results_dict[i] = model.fit_predict()
      print('accuracy', results_dict[i]['validation_statistics']['accuracy'], 'fairness', results_dict[i]['validation_statistics']['fairness'], flush = True)
    try:
      os.makedirs(f'final_result_dictionaries/{model.__class__.__name__}/{data_set.name}/')
    except FileExistsError:
      pass
    torch.save({'data_set': data_set.name, 'random_state': repeat, 'algorithm': model.__class__.__name__, 'hyperparameters': hyperparameters, 'results': results_dict}, 
               f'final_result_dictionaries/{model.__class__.__name__}/{data_set.name}/{repeat}.tar')


def get_statistics(y_hat, y, s, threshold_s_0 = 0.5, threshold_s_1 = 0.5): 

  n_s_0 = (1-s).sum()
  n_s_1 = s.sum()

  assert n_s_0 + n_s_1 == len(s)

  assert min(y_hat) >= 0
  assert -max(y_hat) >= -1

  y_hat_s_0 = y_hat[s==0]
  y_hat_s_1 = y_hat[s==1]

  y_hat_class_s_0 = (y_hat_s_0 > threshold_s_0).astype(int)
  y_hat_class_s_1 = (y_hat_s_1 > threshold_s_1).astype(int)
  correct_s_0 = y_hat_class_s_0 == y[s==0]
  correct_s_1 = y_hat_class_s_1 == y[s==1]

  accuracy = (correct_s_0.sum() + correct_s_1.sum()).astype(float) / len(y)

  s_0_rate = (y_hat_class_s_0.sum() +  np.finfo(np.float32).tiny)/ n_s_0 
  s_1_rate = (y_hat_class_s_1.sum()+  np.finfo(np.float32).tiny)/ n_s_1

  fairness = (min(s_0_rate, s_1_rate) + np.finfo(np.float32).tiny) / (max(s_0_rate, s_1_rate)+np.finfo(np.float32).tiny)


  return({'y': y, 'y_hat': y_hat, 's': s, 'accuracy': accuracy, 'fairness': fairness, 'y_hat_s_0': y_hat_s_0, 'y_hat_s_1': y_hat_s_1, 's_0_rate': s_0_rate, 's_1_rate': s_1_rate, 'y_hat_class_s_0' : y_hat_class_s_0, 'y_hat_class_s_1' : y_hat_class_s_1, 'threshold_s_0': threshold_s_0, 'threshold_s_1': threshold_s_1})

def get_optimal_thresholds(predicted_probabilities_train_a, predicted_probabilities_train_b, train_labels, index_s0_train, index_s1_train, fairness):

  train_labels_a   = train_labels[index_s0_train]

  train_labels_b   = train_labels[index_s1_train]

  df_a = pd.DataFrame(data = {'predicts': predicted_probabilities_train_a, 'labels': train_labels_a})

  df_b = pd.DataFrame(data = {'predicts': predicted_probabilities_train_b, 'labels': train_labels_b})

  df_a_sorted = df_a.sort_values('predicts', ascending = False)

  df_b_sorted = df_b.sort_values('predicts', ascending = False)

  solver = pywraplp.Solver('simple_mip_program',
                 pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
  objective = solver.Objective()

  objective.SetMaximization()

  c1 = df_a_sorted['labels']

  c2 = df_b_sorted['labels']

  c = np.concatenate((c1, c2))

  num_when_bug = -17777

  pick = [num_when_bug] * len(c)

  for i in range(len(c)):

    pick[i] = solver.IntVar(0.0, 1.0, f'{i}')

  for i in range(len(c)):

    objective.SetCoefficient(pick[i], np.asscalar(c[i]))

  constraints = [num_when_bug] * len(c)



  for i in range(len(c1)-1):

    constraints[i] = solver.Constraint(-0.1, solver.infinity())

    constraints[i].SetCoefficient(pick[i], 1)

    constraints[i].SetCoefficient(pick[i+1], -1)

  for i in range(len(c1), len(c)-1):

    constraints[i-1] = solver.Constraint(-0.1, solver.infinity())

    constraints[i-1].SetCoefficient(pick[i], 1)

    constraints[i-1].SetCoefficient(pick[i+1], -1)


  constraints[-2] = solver.Constraint(-solver.infinity(), 0)

  for i in range(len(c1)):

    constraints[-2].SetCoefficient(pick[i], 1/len(c1))

  for i in range(len(c1), len(c)):

    constraints[-2].SetCoefficient(pick[i], fairness * (-1/len(c2)))

  constraints[-1] = solver.Constraint(-solver.infinity(), 0)

  for i in range(len(c1)):

    constraints[-1].SetCoefficient(pick[i], fairness * (-1/len(c1)))

  for i in range(len(c1), len(c)):

    constraints[-1].SetCoefficient(pick[i], 1/len(c2))

  solver.Solve()

  result_status = solver.Solve()

  assert result_status == pywraplp.Solver.OPTIMAL

  assert solver.VerifySolution(1e-10, True)

  solutions_list = [i.solution_value() for i in pick]

  firstzero_a = next(i for i,x in enumerate(solutions_list[:len(c1)]) if x == 0)

  firstzero_b = next(i for i,x in enumerate(solutions_list[len(c1):]) if x == 0)

  threshold_a = df_a_sorted['predicts'].iloc[firstzero_a]

  threshold_b = df_b_sorted['predicts'].iloc[firstzero_b]

  return(threshold_a, threshold_b)

class dataset_for_preprocessed_features_include_s(torch.utils.data.Dataset):
  def __init__(self, data_set, sensitive_variable):
    self.tensor_data = torch.from_numpy(data_set.values.astype(np.float32))
    self.sensitive_variable = sensitive_variable
  def __len__(self):
    return len(self.tensor_data)

  def __getitem__(self, idx):
    feature_cols       = [x for x in range(self.tensor_data.size()[1]) if x != self.tensor_data.size()[1]-1]
    features           = self.tensor_data[idx, feature_cols]
    sensitive_variable = self.tensor_data[idx, self.sensitive_variable]
    labels             = self.tensor_data[idx, -1]
    return features, sensitive_variable, labels
  


def train_test_get_results_pp(data_set, sensitive_variable, hyperparameters, repeats):
  data_set_name  = data_set.name

  for repeat in range(0,repeats):
    data_set = data_set.sample(frac=1, replace=False, random_state = repeat)
    index_train    = np.floor(data_set.shape[0]*0.6).astype(int)
    index_validate = np.floor(data_set.shape[0]*0.8).astype(int)

    model = PP_NN(data_set, index_train, index_validate, sensitive_variable)
    model.get_best_model(20, hyperparameters)
    results_dict = {}
    for i,p in enumerate(np.linspace(0.01,0.99,100)):
     
      results_dict[i] = model.get_thresholds_and_predict(1/p)
      print('accuracy', results_dict[i]['test_statistics']['accuracy'], 'fairness', results_dict[i]['test_statistics']['fairness'])
    try:
      os.makedirs(f'final_result_dictionaries/{model.__class__.__name__}/{data_set_name}/')
    except FileExistsError:
      pass
    torch.save({'data_set': data_set_name, 'random_state': repeat, 'algorithm': model.__class__.__name__, 'hyperparameters': hyperparameters, 'results': results_dict}, 
               f'final_result_dictionaries/{model.__class__.__name__}/{data_set_name}/{repeat}.tar')