import numpy as np
from random import shuffle

def add2Column(matrix, index, column):
  matrix[:, index:index + 1] += column

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    indicator_f = 0
    for j in range(num_classes):
      if j == y[i]:
        continue

      margin = scores[j] - correct_class_score + 1  # note delta = 1

      if margin > 0:
        loss += margin

        indicator_f += 1
        add2Column(dW, j, X[i][:, None])

    add2Column(dW, y[i], -(indicator_f * X[i][:, None]))

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  dW_reg = reg * 2 * W
  dW += dW_reg
  dW /= num_train


  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW

def L_i_vectorized(x, y, W):
  delta = 1.0
  scores = x.dot(W)
  margins = np.maximum(0, scores - scores[y] + delta)
  margins[y] = 0
  # print(margins.shape)

  loss_i = np.sum(margins)
  return loss_i

def svm_loss_vectorized(W, X, y, reg):
  """
   Structured SVM loss function, vectorized implementation.

   Inputs and outputs are the same as svm_loss_naive.
   """
  loss = 0.0
  dW = np.zeros(W.shape)  # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  # for i in range(num_train):
  #   loss += L_i_vectorized(X[i], y[i], W)


  print(X.shape, W.shape, y.shape)

  delta = 1.0
  scores = X.dot(W)
  print(scores.shape)



  margins = np.maximum(0, scores - scores[y] + delta)
  margins[y] = 0

  loss = np.sum(margins)

  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  return loss, dW

def _svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    loss += L_i_vectorized(X[i], y[i], W)

  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  return loss, dW
