import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  num_train = y.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)

    scores_e = np.exp(scores)
    scores_e_sum = np.sum(scores_e)
    score_correct_e = scores_e[y[i]]

    l_i = -np.log(score_correct_e / scores_e_sum)
    loss += l_i

    for j in range(num_classes):
      if j == y[i]:
        dw_j = (-1) * (scores_e_sum - score_correct_e) / scores_e_sum * X[i]
      else:
        dw_j = scores_e[j] / scores_e_sum * X[i]

      dW[:, j] += dw_j

  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += 2 * reg * W

  return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = y.shape[0]
  num_classes = W.shape[1]

  train_indx = list(range(num_train))

  scores = X.dot(W)
  scores -= np.max(scores, axis=1)[:, None]

  scores_e = np.exp(scores)
  score_correct_e = scores_e[train_indx, y]
  scores_e_sum = np.sum(scores_e, axis=1)

  #print(scores_e.shape, score_correct_e.shape, scores_e_sum.shape)

  loss = np.sum(-np.log(score_correct_e / scores_e_sum))

  loss /= num_train
  loss += reg * np.sum(W * W)

  # grad


  dW /= num_train
  dW += 2 * reg * W

  return loss, dW

