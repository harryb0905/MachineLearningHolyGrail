import numpy as np
import matplotlib.pyplot as plt
import random
plt.style.use('ggplot')

np.random.seed(1)

class LinearRegression:
  ''' An OOP implementation of the linear regression algorithm '''
  def __init__(self):
    ''' Initialises the linear regression class variables '''
    self.theta_hat_ = None
    self.intercept_ = None
    self.coef_ = None
    self._n_params = 0

  def fit(self, X, y):
    '''
    Runs the linear regression algorithm by estimating model parameters using a closed-form
    solution known as the Normal Equation.

    Parameters:
    -----------
      X (numpy.ndarray): a numpy array of input data values of shape (n_items, 1).
      y (numpy.ndarray): a numpy array of output values of shape (n_items, 1).
    '''
    self.X, self.y = X, y
    m = self.X.shape[0]

    # Add x0 = 1 to each instance and store n_params
    X_b = np.c_[np.ones((m, 1)), X] 
    self._n_params = X_b.shape[1]

    # Compute theta hat using the Normal equation
    self.theta_hat_ = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    # Store estimated parameters
    self.intercept_ = self.theta_hat_[0].item()
    self.coef_ = self.theta_hat_[1:][0]

    # Can also use SVD
    # theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, self.y, rcond=1e-6)
    # theta_best_svd = np.linalg.pinv(X_b).dot(self.y)

    # Or gradient descent
    eta = 0.1  # learning rate
    n_iterations = 1000

    theta = np.random.randn(2,1)  # random initialization

    for iteration in range(n_iterations):
        gradients = (2/m) * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients

    print(f'theta: {theta}')

  def predict(self, X_new, plot_preds=True):
    '''
    Predicts outputs based on the provided data items.

    Parameters:
    -----------
      X_new (numpy.ndarray): a numpy array of input data values of shape (n_items, 1).
      plot_preds (boolean): True if predictions are plotted, False otherwise.

    Returns:
    --------
      numpy.ndarray: a numpy array of y values of shape (n_items, 1).
    '''
    n_items = X_new.shape[0]

    # Compute predictions based on estimated parameters
    X_new_b = np.c_[np.ones((n_items, 1)), X_new] # add x0 = 1 to each instance
    y_predict = X_new_b.dot(self.theta_hat_)

    # Plot predicted values
    if plot_preds:
      self._plot_preds(X_new, y_predict)

    print(f'Prediction: {y_predict}')
    return y_predict

  def _plot_preds(self, X_new, y_predict):
    '''
    Plots the predicted line equation overlaying the true data values.

    Parameters:
    -----------
      X_new (numpy.ndarray): a numpy array of input data values of shape (n_items, 1).
      y_predict (numpy.ndarray): a numpy array of output data values of shape (n_items, 1).
    '''
    coef = round(self.coef_[0], 2)
    sign = '-' if coef < 0 else '+'
    estimated_eq = f'y = {round(self.intercept_, 2)} {sign} {coef}$X_1$'

    # Plot model predictions
    plt.plot(X_new, y_predict, "r-", linewidth=2, label='Predicted')
    plt.plot(self.X, self.y, "b.", label='Actual')
    plt.axis([0, np.max(self.X), 0, np.max(self.y)+0.05*np.max(self.y)])
    plt.title(estimated_eq)
    plt.xlabel('$X_1$')
    plt.ylabel('y')
    plt.legend(loc="upper left", fontsize=12)
    plt.show()

def main():
  # Generate data: y = 5 + 2x_1 + Gaussian noise
  n_items = 200
  X = 2 * np.random.rand(n_items, 1)
  y = 5 + 2 * X + np.random.randn(n_items, 1) 

  # Fit linear regression model and obtain estimated parameters
  lr = LinearRegression()
  lr.fit(X, y)
  print(lr.intercept_, lr.coef_)

  # Use model to make predictions on unseen data items
  X_new = np.array([[0], [2]])
  lr.predict(X_new)

if __name__ == '__main__':
  main()
