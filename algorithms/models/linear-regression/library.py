from sklearn.linear_model import LinearRegression
import numpy as np

np.random.seed(1)

class LinearRegressionWrapper:
  ''' Provides a wrapper for the LinearRegression class provided by scikit-learn '''
  def __init__(self):
    ''' Initialises the wrapper instance '''
    pass

  def fit(self, X, y):
    '''
    Fits the training data X to the output values y to obtain model parameters:

    Parameters:
    -----------
      X (numpy.ndarray): a numpy array of training items of shape (n_items, 1).
      y (numpy.ndarray): a numpy array of output values corresponding to the training items of shape (n_items, 1).

    Returns:
    --------
      self: the current instance.
    '''
    self._lin_reg = LinearRegression()
    self._lin_reg.fit(X, y)

    self.intercept_ = self._lin_reg.intercept_
    self.coef_ = self._lin_reg.coef_
    return self

  def predict(self, X_new):
    '''
    Predicts and returns the output values for items in the testing dataset.

    Parameters:
    -----------
      X_new (numpy.ndarray): a numpy array of testing data items of shape (n_items, 1).

    Returns:
    --------
      numpy.ndarray: a numpy array of predicted output values of shape (n_items, 1).
    '''
    preds = self._lin_reg.predict(X_new)
    return preds

def main():
  # Create dataset with equation y = 5 + 2x + Gaussian noise
  n_items = 100
  X = 2 * np.random.rand(n_items, 1)
  y = 5 + 2 * X + np.random.randn(n_items, 1)

  # Fit X to y using the linear regression wrapper
  lin_reg = LinearRegressionWrapper().fit(X, y)

  # Predict outputs for test data
  X_new = np.array([[0], [1]])
  preds = lin_reg.predict(X_new)
  print(preds)

if __name__ == '__main__':
  main()

