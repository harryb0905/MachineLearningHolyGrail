from sklearn.linear_model import LinearRegression, SGDRegressor
import numpy as np

class LinearRegressionWrapper:
  def __init__(self, algorithm="SGD"):
    self._alg_funcs = {
      "normal": self._normal,
      "SVD": self._svd,
      "SGD": self._sgd
    }

    self._model = None

    if algorithm in self._alg_funcs:
      self._algorithm = algorithm
    else:
      return f"Algorithm {algorithm} is not supported. (Supported algorithms: {', '.join(self._alg_funcs.keys())})"

  def fit(self, X, y, **kwargs):
    self._alg_funcs[self._algorithm](X, y, **kwargs)

  def _normal(self, X, y, **kwargs):
    X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    self.intercept_, self.coef_ = theta_best[0].item(), theta_best[1].item()

  def _svd(self, X, y, **kwargs):
    self._model = LinearRegression()
    self._model.fit(X, y)
    self.intercept_, self.coef_ = self._model.intercept_.item(), self._model.coef_.item()

  def _sgd(self, X, y, **kwargs):
    self._model = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
    self._model.fit(X, y.ravel())
    self.intercept_, self.coef_ = self._model.intercept_.item(), self._model.coef_.item()

  def predict(self, X_new):
    X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new] # add x0 = 1 to each instance
    y_predict = X_new_b.dot(np.array([[self.intercept_], [self.coef_]]))
    return y_predict

def main():
  np.random.seed(42)

  # Generate fake training and test data
  X = 5 * np.random.rand(100, 1)
  y = 8 + 10 * X + np.random.randn(100, 1)
  X_new = np.array([[0],[1],[2],[3]])

  # Run linear regression with chosen algorithm and obtain predictions
  algorithms = ["normal", "SVD", "SGD"]
  for algo in algorithms:
    lin_reg = LinearRegressionWrapper(algorithm=algo)
    lin_reg.fit(X, y)
    print(f"Linear regression results (with {algo} algorithm):\nIntercept: {lin_reg.intercept_}, Coefficients: {lin_reg.coef_}")
    preds = lin_reg.predict(X_new)
    print(f"Predictions: {preds}")

if __name__ == "__main__":
  main()
