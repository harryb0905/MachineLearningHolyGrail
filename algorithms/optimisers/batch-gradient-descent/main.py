import numpy as np

# Batch Gradient Descent for Linear Regression
def batch_gd(X, y, eta=0.1, n_iterations=1000):
  m = X.shape[0]

  theta = np.random.randn(2,1)

  for iteration in range(n_iterations):
    gradients = 2/m * X.T.dot(X.dot(theta) - y)
    theta = theta - eta * gradients

  return theta

def main():
  # Create dataset
  n_items = 100
  X = 2 * np.random.rand(n_items, 1)
  y = 5 + 2 * X + np.random.randn(n_items, 1) 
  X_b = np.c_[np.ones((n_items, 1)), X] 

  # Run batch gradient descent to obtain parameters
  theta = batch_gd(X_b, y)
  print(f'Theta: {theta}')

if __name__ == '__main__':
  main()