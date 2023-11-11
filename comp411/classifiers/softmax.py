import numpy as np
from random import shuffle
import builtins

def softmax_loss_naive(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg_l2: (float) regularization strength for L2 regularization
    - reg_l1: (float) default: 0. regularization strength for L1 regularization 
                to be used in Elastic Net Reg. if supplied, this function uses Elastic
                Net Regularization.

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0.:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.      #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    for i in range(N):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        scores = np.exp(scores)
        scores /= np.sum(scores)
        loss += -np.log(scores[y[i]])
        for j in range(W.shape[1]):
            if j == y[i]:
                dW[:, j] += (scores[j] - 1) * X[i]
            else:
                dW[:, j] += scores[j] * X[i]

    if regtype == 'ElasticNet':
        loss /= N
        loss += reg_l2 * np.sum(W * W) + reg_l1 * np.sum(np.abs(W))
        dW /= N
        dW += 2 * reg_l2 * W + reg_l1 * np.sign(W)

    elif regtype == 'L2':
        loss /= N
        loss += reg_l2 * np.sum(W * W)
        dW /= N
        dW += 2 * reg_l2 * W
        
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg_l2, reg_l1 = 0):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    if reg_l1 == 0:
        regtype = 'L2'
    else:
        regtype = 'ElasticNet'
    
    ##############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.   #
    # Store the loss in loss and the gradient in dW. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the         #
    # regularization! If regtype is set as 'L2' just implement L2 Regularization #
    # else implement both L2 and L1.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    scores = X.dot(W)
    scores -= np.max(scores, axis=1)[:, np.newaxis]
    scores = np.exp(scores)
    scores /= np.sum(scores, axis=1)[:, np.newaxis]
    loss = -np.sum(np.log(scores[np.arange(N), y]))
    loss /= N
    if regtype == 'ElasticNet':
        loss += reg_l2 * np.sum(W * W) + reg_l1 * np.sum(np.abs(W))
        scores[np.arange(N), y] -= 1
        dW = X.T.dot(scores)
        dW /= N
        dW += 2 * reg_l2 * W + reg_l1 * np.sign(W)
    elif regtype == 'L2':
        loss += reg_l2 * np.sum(W * W)
        scores[np.arange(N), y] -= 1
        dW = X.T.dot(scores)
        dW /= N
        dW += 2 * reg_l2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
