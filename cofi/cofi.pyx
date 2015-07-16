#cython: embedsignature=True, boundscheck=True

import numpy as np
from numpy.random import randn
from scipy.optimize import fmin_cg
from sklearn.base import BaseEstimator
from sklearn.utils.validation import NotFittedError


def roll_params(X, Theta):
    return np.concatenate((X.ravel(), Theta.ravel()))


def unroll_params(params, num_users, num_items, num_features):
    X = params[:num_items*num_features].reshape(num_items, num_features)
    Theta = params[num_items*num_features:].reshape(num_users, num_features)
    return X, Theta


def cost(params,
         long [:] item,
         long [:] user,
         double [:] rating,
         int num_ratings,
         double lm,
         int num_users, 
         int num_items, 
         int num_features=10):

    """Regularized cost function for collaborative filtering"""
    
    cdef:
        int i, j, k
        double delta, J = 0.
        double [:,:] X, Theta
        
    X, Theta = unroll_params(params, num_users, num_items, num_features)
    
    for r in range(num_ratings):
        
        i = item[r]
        j = user[r]
        delta = -rating[r]

        for k in range(num_features):
            delta += Theta[j,k]*X[i,k]

        J += 0.5*delta**2
        
    # regularization
    
    for i in range(num_items):
        for k in range(num_features):
            J += 0.5*lm*X[i,k]**2
            
    for j in range(num_users):
        for k in range(num_features):
            J += 0.5*lm*Theta[j,k]**2
    
    return J


def cost_grad(params,
              long [:] item,
              long [:] user,
              double [:] rating,
              int num_ratings,
              double lm,
              int num_users, 
              int num_items, 
              int num_features=10):

    """Gradient of regularized cost function for collaborative filtering"""
    
    cdef:
        int i, j, k
        double delta
        double [:,:] X, Theta, X_grad, Theta_grad
        
    X, Theta = unroll_params(params, num_users, num_items, num_features)
    X_grad = np.zeros_like(X)
    Theta_grad = np.zeros_like(Theta)
    
    for r in range(num_ratings):
        
        i = item[r]
        j = user[r]
        delta = -rating[r]

        for k in range(num_features):
            delta += Theta[j,k]*X[i,k]

        for k in range(num_features):
            X_grad[i,k] += delta*Theta[j,k]
            Theta_grad[j,k] += delta*X[i,k]
            
    
    # regularization
    
    for i in range(num_items):
        for k in range(num_features):
            X_grad[i,k] += lm*X[i,k]
            
    for j in range(num_users):
        for k in range(num_features):
            Theta_grad[j,k] += lm*Theta[j,k]
    
    grad = np.concatenate((np.asarray(X_grad).ravel(), 
                           np.asarray(Theta_grad).ravel()))
    
    return grad


def _predict(double [:,:] X, 
             double [:,:] Theta, 
             int num_features,
             long [:] item, 
             long [:] user,
             int num_ratings):

    cdef double [:] ypred = np.zeros(num_ratings)

    for r in range(num_ratings):
        for k in range(num_features):
            ypred[r] += X[item[r],k]*Theta[user[r],k]

    return np.asarray(ypred)


class CollaborativeFiltering(BaseEstimator):
    
    def __init__(self, num_users, num_items,
                 num_features=10, lm=0, maxiter=100):

        self.num_users = num_users
        self.num_items = num_items
        self.num_features = num_features
        self.lm = lm 
        self.maxiter = maxiter
        

    def fit(self, X, y, progress=False, **kw):

        # initialize parameters randomly
        num_params = (self.num_users + self.num_items)*self.num_features
        params_initial = randn(num_params)

        # arguments to pass to cost function
        args = (X[:,0], X[:,1], y, len(y),
                self.lm, self.num_users,
                self.num_items, self.num_features)

        # set parameters for minimization routine

        params = dict(disp=False)
        params.update(kw)

        if progress:
            params['callback'] = StatusPrinter(cost, args)

        # minimize cost using conjugate gradient

        params_optimal = fmin_cg(cost, params_initial, 
                                 cost_grad, args,
                                 maxiter=self.maxiter, **params)
        
        self.X_, self.Theta_ = unroll_params(params_optimal, 
                                             self.num_users, 
                                             self.num_items, 
                                             self.num_features)

        return self

        
    def predict(self, X):

        if not hasattr(self, 'X_'):
            raise NotFittedError()

        return _predict(self.X_, self.Theta_, self.num_features,
                        X[:,0], X[:,1], X.shape[0])



class StatusPrinter:
    
    """Print number of iterations and cost"""
    
    def __init__(self, cost_func, args):
        self.i = 0
        self.cost_func = cost_func
        self.args = args

    def _print_status(self, x):
        cost = self.cost_func(x, *self.args)
        print('iteration {}, cost = {}'.format(self.i, cost))
        
    def __call__(self, x):
        self.i += 1

        if self.i < 10:
            self._print_status(x)
        elif self.i < 100 and self.i % 10 == 0:
            self._print_status(x)
        elif self.i % 100 == 0:
            self._print_status(x)
