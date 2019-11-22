
import numpy as np
from scipy.optimize import minimize # for loss func minimization
from copy import deepcopy

import os,sys
import traceback
import numpy as np
from random import seed, shuffle
#from cvxpy import *
import dccp
from dccp.problem import is_dccp


def log_logistic(X):

	""" This function is used from scikit-learn source code. Source link below """

	"""Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.
	This implementation is numerically stable because it splits positive and
	negative values::
	    -log(1 + exp(-x_i))     if x_i > 0
	    x_i - log(1 + exp(x_i)) if x_i <= 0
	Parameters
	----------
	X: array-like, shape (M, N)
	    Argument to the logistic function
	Returns
	-------
	out: array, shape (M, N)
	    Log of the logistic function evaluated at every point in x
	Notes
	-----
	Source code at:
	https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
	-----
	See the blog post describing this implementation:
	http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
	"""
	if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
	out = np.empty_like(X) # same dimensions and data types

	idx = X>0
	out[idx] = -np.log(1.0 + np.exp(-X[idx]))
	out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
	return out

def logistic_loss(w, X, y, return_arr=None):
	"""Computes the logistic loss.
	This function is used from scikit-learn source code
	Parameters
	----------
	w : ndarray, shape (n_features,) or (n_features + 1,)
	    Coefficient vector.
	X : {array-like, sparse matrix}, shape (n_samples, n_features)
	    Training data.
	y : ndarray, shape (n_samples,)
	    Array of labels.
	"""
	

	yz = y * np.dot(X,w)
	# Logistic loss is the negative of the log of the logistic function.
	if return_arr == True:
		  out = -(log_logistic(yz))
	else:
		  out = -np.sum(log_logistic(yz))
	return out


def train_model(x, y, x_control, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma=0,loss_function = logistic_loss, apply_fairness_constraints = 1, apply_accuracy_constraint = 0, sep_constraint = 0):

    """
    Function that trains the model subject to various fairness constraints.
    If no constraints are given, then simply trains an unaltered classifier.
    Example usage in: "synthetic_data_demo/decision_boundary_demo.py"
    ----
    Inputs:
    X: (n) x (d+1) numpy array -- n = number of examples, d = number of features, one feature is the intercept
    y: 1-d numpy array (n entries)
    x_control: dictionary of the type {"s": [...]}, key "s" is the
    sensitive feature name, and the value is a 1-d list with n elements holding the sensitive feature values
    loss_function: the loss function that we want to optimize -- for now we have implementation of 
    logistic loss, but other functions like hinge loss can also be added
    apply_fairness_constraints: optimize accuracy subject to fairness constraint (0/1 values)
    apply_accuracy_constraint: optimize fairness subject to accuracy constraint (0/1 values)
    sep_constraint: apply the fine grained accuracy constraint
        for details, see Section 3.3 of arxiv.org/abs/1507.05259v3
        For examples on how to apply these constraints, see "synthetic_data_demo/decision_boundary_demo.py"
    Note: both apply_fairness_constraints and apply_accuracy_constraint cannot be 1 at the same time
    sensitive_attrs: ["s1", "s2", ...], list of sensitive features for which to apply fairness constraint, 
    all of these sensitive features should have a corresponding array in x_control
    sensitive_attrs_to_cov_thresh: the covariance threshold that the classifier should achieve
    (this is only needed when apply_fairness_constraints=1, not needed for the other two constraints)
    gamma: controls the loss in accuracy we are willing to incur when using apply_accuracy_constraint and sep_constraint
    ----
    Outputs:
    w: the learned weight vector for the classifier
    """


    assert((apply_accuracy_constraint == 1 and apply_fairness_constraints == 1) == False) # both constraints cannot be applied at the same time

    max_iter = 100000 # maximum number of iterations for the minimization algorithm

    if apply_fairness_constraints == 0:
      constraints = []
    else:
      constraints = get_constraint_list_cov(x, y, x_control, sensitive_attrs, sensitive_attrs_to_cov_thresh)      

    if apply_accuracy_constraint == 0: #its not the reverse problem, just train w with cross cov constraints
        f_args=(x, y)
        w = minimize(fun = loss_function,
            x0 = np.random.rand(x.shape[1],),
            args = f_args,
            method = 'SLSQP',
            options = {"maxiter":max_iter},
            constraints = constraints
            )
       
    else:

        # train on just the loss function
        w = minimize(fun = loss_function,
            x0 = np.random.rand(x.shape[1],),
            args = (x, y),
            method = 'SLSQP',
            options = {"maxiter":max_iter},
            constraints = []
            )

        old_w = deepcopy(w.x)
        

        def constraint_gamma_all(w, x, y,  initial_loss_arr):
            
            gamma_arr = np.ones_like(y) * gamma # set gamma for everyone
            new_loss = loss_function(w, x, y)
            old_loss = sum(initial_loss_arr)
            return ((1.0 + gamma) * old_loss) - new_loss

        def constraint_protected_people(w,x,y): # dont confuse the protected here with the sensitive feature protected/non-protected values -- protected here means that these points should not be misclassified to negative class
            return np.dot(w, x.T) # if this is positive, the constraint is satisfied
        def constraint_unprotected_people(w,ind,old_loss,x,y):
            
            new_loss = loss_function(w, np.array([x]), np.array(y))
            return ((1.0 + gamma) * old_loss) - new_loss

        constraints = []
        predicted_labels = np.sign(np.dot(w.x, x.T))
        unconstrained_loss_arr = loss_function(w.x, x, y, return_arr=True)

        if sep_constraint == True: # separate gemma for different people
            for i in range(0, len(predicted_labels)):
                if predicted_labels[i] == 1.0 and x_control[sensitive_attrs[0]][i] == 1.0: # for now we are assuming just one sensitive attr for reverse constraint, later, extend the code to take into account multiple sensitive attrs
                    c = ({'type': 'ineq', 'fun': constraint_protected_people, 'args':(x[i], y[i])}) # this constraint makes sure that these people stay in the positive class even in the modified classifier             
                    constraints.append(c)
                else:
                    c = ({'type': 'ineq', 'fun': constraint_unprotected_people, 'args':(i, unconstrained_loss_arr[i], x[i], y[i])})                
                    constraints.append(c)
        else: # same gamma for everyone
            c = ({'type': 'ineq', 'fun': constraint_gamma_all, 'args':(x,y,unconstrained_loss_arr)})
            constraints.append(c)

        def cross_cov_abs_optm_func(weight_vec, x_in, x_control_in_arr):
            cross_cov = (x_control_in_arr - np.mean(x_control_in_arr)) * np.dot(weight_vec, x_in.T)
            return float(abs(sum(cross_cov))) / float(x_in.shape[0])


        w = minimize(fun = cross_cov_abs_optm_func,
            x0 = old_w,
            args = (x, x_control[sensitive_attrs[0]]),
            method = 'SLSQP',
            options = {"maxiter":100000},
            constraints = constraints
            )

    try:
        assert(w.success == True)
    except:
        print("Optimization problem did not converge.. Check the solution returned by the optimizer.")
        print("Returned solution is:")
        print(w)



    return w.x






def add_intercept(x):

    """ Add intercept to the data before linear classification """
    m,n = x.shape
    intercept = np.ones(m).reshape(m, 1) # the constant b
    return np.concatenate((intercept, x), axis = 1)


def get_line_coordinates(w, x1, x2):
    y1 = (-w[0] - (w[1] * x1)) / w[2]
    y2 = (-w[0] - (w[1] * x2)) / w[2]    
    return y1,y2
  
def get_constraint_list_cov(x_train, y_train, x_control_train, sensitive_attrs, sensitive_attrs_to_cov_thresh):

    """
    get the list of constraints to be fed to the minimizer
    """

    constraints = []


    for attr in sensitive_attrs:


        attr_arr = x_control_train[attr]                
        thresh = sensitive_attrs_to_cov_thresh[attr]
        c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov, 'args':(x_train, y_train, attr_arr,thresh, False)})
        constraints.append(c)
        

    return constraints
  
def test_sensitive_attr_constraint_cov(model, x_arr, y_arr_dist_boundary, x_control, thresh, verbose):

    
    """
    The covariance is computed b/w the sensitive attr val and the distance from the boundary
    If the model is None, we assume that the y_arr_dist_boundary contains the distace from the decision boundary
    If the model is not None, we just compute a dot product or model and x_arr
    for the case of SVM, we pass the distace from bounday becase the intercept in internalized for the class
    and we have compute the distance using the project function
    this function will return -1 if the constraint specified by thresh parameter is not satifsified
    otherwise it will reutrn +1
    if the return value is >=0, then the constraint is satisfied
    """

    


    assert(x_arr.shape[0] == x_control.shape[0])
    if len(x_control.shape) > 1: # make sure we just have one column in the array
        assert(x_control.shape[1] == 1)
    
    arr = []
    if model is None:
        arr = y_arr_dist_boundary # simply the output labels
    else:
        arr = np.dot(model, x_arr.T) # the product with the weight vector -- the sign of this is the output label
    
    arr = np.array(arr, dtype=np.float64)


    cov = np.dot(x_control - np.mean(x_control), arr ) / float(len(x_control))

        
    ans = thresh - abs(cov) # will be <0 if the covariance is greater than thresh -- that is, the condition is not satisfied
    # ans = thresh - cov # will be <0 if the covariance is greater than thresh -- that is, the condition is not satisfied
    if verbose is True:
        print("Covariance is"), cov
        print("Diff is:"), ans
        print
    return ans
  
def zafar_logreg(train_set, train_labels, test_set, test_labels, sensitive_variable, fairness = 0.0):
    train_set, test_set = add_intercept(train_set),  add_intercept(test_set)# adds intercept as first column
    #train_set, test_set = np.delete(train_set, sensitive_variable, 1), np.delete(test_set, sensitive_variable, 1)
    params = {'apply_fairness_constraints'    : 1,
	            'apply_accuracy_constraint'     : 0,
	            'sep_constraint'                : 0,
              'x'                             : train_set[:,:-1],
              'y'                             : train_labels, 
              'x_control'                     : {'s1': train_set[:,sensitive_variable]}, 
              'sensitive_attrs'               : ['s1'], 
              'sensitive_attrs_to_cov_thresh' : {'s1':fairness},
              'gamma'                         : 0.5}
    
    w = train_model(**params)
        
    labels = np.sign(np.matmul(test_set[:,:-1], w))
            
    labels_a = labels[test_set[:,sensitive_variable] == 0 ]        

    labels_b = labels[test_set[:,sensitive_variable] == 1 ]

    len_a_test = len(test_set[test_set[:,sensitive_variable] == 0 ])

    len_b_test = len(test_set[test_set[:,sensitive_variable] == 1 ])
    
    positive_rate_a = (labels_a == 1).sum()/len_a_test
        
    positive_rate_b = (labels_b == 1).sum()/len_b_test
    
    richtig_a = (labels_a == (test_labels[test_set[:,sensitive_variable] == 0])).sum()
     
    alle_a = (test_set[:,sensitive_variable]== 0).sum()
    
    richtig_b = (labels_b == (test_labels[test_set[:,sensitive_variable] == 1])).sum()

    alle_b = (test_set[:,sensitive_variable]== 1).sum()

    rate_a = richtig_a/alle_a

    rate_b = richtig_b/alle_b
    
    return({'accuracy': (richtig_a+richtig_b) / (alle_a + alle_b), 'fairness' : (positive_rate_a/positive_rate_b), 'rate_a' : positive_rate_a, 'rate_b' : positive_rate_b})


