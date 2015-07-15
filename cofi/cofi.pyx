#cython: boundscheck=False

import numpy as np

def cofi_cost_sparse(params, 
                     int [:] item, 
                     int [:] user,
                     double [:] rating,
                     int num_ratings,
                     double lm,
                     int num_users, 
                     int num_items, 
                     int num_features=100):

    """Regularized cost function for collaborative filtering"""
    
    cdef:
        int i, j, k
        double delta, J = 0.
        double [:,:] X, Theta
        
    # unroll parameters
    X = params[:num_items*num_features].reshape(num_items, num_features)
    Theta = params[num_items*num_features:].reshape(num_users, num_features)
    
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


def cofi_cost_grad_sparse(params, 
                          int [:] item, 
                          int [:] user,
                          double [:] rating,
                          int num_ratings,
                          double lm,
                          int num_users, 
                          int num_items, 
                          int num_features=100):

    """Gradient of regularized cost function for collaborative filtering"""
    
    cdef:
        int i, j, k
        double delta
        double [:,:] X, Theta, X_grad, Theta_grad
        
    # unroll parameters
    X = params[:num_items*num_features].reshape(num_items, num_features)
    Theta = params[num_items*num_features:].reshape(num_users, num_features)
    
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
