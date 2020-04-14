from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

class LinearRegressionWrapper:
  def __init__(self, algorithm="SGD", poly_degree=None):
    self._alg_funcs = {
      "normal": self._normal,
      "SVD": self._svd,
      "SGD": self._sgd
    }

    self._poly_features = None
    if poly_degree is not None:
      self._poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)

    self._model = None

    if algorithm in self._alg_funcs:
      self._algorithm = algorithm
    else:
      return f"Algorithm {algorithm} is not supported. (Supported algorithms: {', '.join(self._alg_funcs.keys())})"

  def fit(self, X, y, **kwargs):
    # Run polynomial regression if specified
    if self._poly_features is not None:
      X = self._poly_features.fit_transform(X)

      if self._algorithm == "normal":
        print("[Warning] Cannot run polynomial regression using the Normal equation with a degree higher than 1 - running linear regression of degree 1")
    self._alg_funcs[self._algorithm](X, y, **kwargs)

  def _normal(self, X, y, **kwargs):
    X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    self.intercept_, self.coef_ = theta_best[0].item(), theta_best[1].item()

  def _svd(self, X, y, **kwargs):
    self._model = LinearRegression()
    self._model.fit(X, y)
    print(self._model.coef_)
    self.intercept_, self.coef_ = self._model.intercept_.item(), self._model.coef_[0]

  def _sgd(self, X, y, **kwargs):
    self._model = SGDRegressor(max_iter=1000, tol=1e-3, penalty="l2", eta0=0.1)
    self._model.fit(X, y.ravel())
    self.intercept_, self.coef_ = self._model.intercept_.item(), self._model.coef_[0]

  def predict(self, X_new):
    if self._algorithm == "normal":
      X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new] # add x0 = 1 to each instance
      y_predict = X_new_b.dot(np.array([[self.intercept_], [self.coef_]]))
      return y_predict

    # Run polynomial regression if specified
    if self._poly_features is not None:
      X_new_b = self._poly_features.fit_transform(X_new)

    print(X_new_b)
    return self._model.predict(X_new_b)
  
def main():
  np.random.seed(42)

  # Linear data: generate fake training and test data
  X = 5 * np.random.rand(100, 1)
  y = 8 + 10 * X + np.random.randn(100, 1)

  # Non-linear data: generate fake training and test data
  m = 100
  X = 5 * np.random.randn(m, 1) - 5
  y = 0.5 * X**2 + X + 4 + np.random.randn(m, 1)
  X_new = np.array([[0],[1],[2],[3]])

  # Run linear regression with chosen algorithm and obtain predictions
  algorithms = ["normal", "SVD", "SGD"]
  for algo in algorithms:
    lin_reg = LinearRegressionWrapper(algorithm=algo, poly_degree=2)
    lin_reg.fit(X, y)
    print(f"Linear regression results (with {algo} algorithm):\nIntercept: {lin_reg.intercept_}, Coefficients: {lin_reg.coef_}")
    preds = lin_reg.predict(X_new)
    print(f"Predictions: {preds}")

if __name__ == "__main__":
  main()
