from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FourLayerNet(object):
    """
    A four-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be
        affine - relu - affine - relu - affine - relu - affine softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-2,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layers
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the four-layer net. Weights   #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2' and so on..              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        w1 = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        b1 = np.zeros(hidden_dim)
        w2 = np.random.normal(0, weight_scale, (hidden_dim, hidden_dim))
        b2 = np.zeros(hidden_dim)
        self.params['W1'] = w1
        self.params['b1'] = b1
        self.params['W2'] = w2
        self.params['b2'] = b2
        w3 = np.random.normal(0, weight_scale, (hidden_dim, hidden_dim))
        b3 = np.zeros(hidden_dim)
        self.params['W3'] = w3
        self.params['b3'] = b3
        w4 = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        b4 = np.zeros(num_classes)
        self.params['W4'] = w4
        self.params['b4'] = b4

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the four-layer net, computing the   #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        w1 = self.params['W1']
        b1 = self.params['b1']
        w2 = self.params['W2']
        b2 = self.params['b2']
        w3 = self.params['W3']
        b3 = self.params['b3']
        w4 = self.params['W4']
        b4 = self.params['b4']

        out1, cache1 = affine_relu_forward(X, w1, b1)
        out2, cache2 = affine_relu_forward(out1, w2, b2)
        out3, cache3 = affine_relu_forward(out2, w3, b3)
        scores, cache4 = affine_forward(out3, w4, b4)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the four-layer net. Store the loss #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dout2 = softmax_loss(scores, y)
        loss += self.reg * (np.sum(w1 * w1) + np.sum(w2 * w2) + np.sum(w3 * w3) + np.sum(w4 * w4))/2
        dout1, dw4, db4 = affine_backward(dout2, cache4)
        dout0, dw3, db3 = affine_relu_backward(dout1, cache3)
        dout, dw2, db2 = affine_relu_backward(dout0, cache2)
        dx, dw1, db1 = affine_relu_backward(dout, cache1)
        grads['W1'] = dw1 + self.reg * w1
        grads['b1'] = db1
        grads['W2'] = dw2 + self.reg * w2
        grads['b2'] = db2
        grads['W3'] = dw3 + self.reg * w3
        grads['b3'] = db3
        grads['W4'] = dw4 + self.reg * w4
        grads['b4'] = db4



        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
