from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import numpy as np

# For reproducability
np.random.seed(42) 

class LinearRegressionWrapper:
  def __init__(self, algorithm="SGD", poly_degree=1):
    self._alg_funcs = {
      "normal": self._normal,
      "SVD": self._svd,
      "SGD": self._sgd
    }
    
    self._model = None
    self._poly_degree = poly_degree

    if algorithm in self._alg_funcs:
      self._algorithm = algorithm
    else:
      return f"Algorithm {algorithm} is not supported. (Supported algorithms: {', '.join(self._alg_funcs.keys())})"

  def fit(self, X, y, **kwargs):
    # Run polynomial regression if specified
    if self._poly_degree > 1 and self._algorithm == "normal":
      # Update degree of polynomial to 1 when using normal equation
      print("[Warning] Cannot run polynomial regression using the Normal equation with a degree higher than 1 - running linear regression of degree 1")
      self._poly_features = PolynomialFeatures(degree=1, include_bias=False)
      self._poly_degree = 1
    else:
      self._poly_features = PolynomialFeatures(degree=self._poly_degree, include_bias=False)

    # Run linear regression algorithm
    self._alg_funcs[self._algorithm](X, y, **kwargs)

  def _normal(self, X, y, **kwargs):
    X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    self.intercept_, self.coef_ = theta_best[0].item(), theta_best[1].item()

  def _svd(self, X, y, **kwargs):
    # Fit LinearRegression model (no scaling required)
    self._clf = Pipeline([
      ('poly_features', self._poly_features), # may be redundant step if degree=1 is used
      ('classifier', LinearRegression())
    ]).fit(X, y)

    # Extract computed intercept and coefficient values
    self.intercept_, self.coef_ = self._clf['classifier'].intercept_.item(), self._clf['classifier'].coef_[0]

  def _sgd(self, X, y, **kwargs):
    clf = SGDRegressor(max_iter=1000, tol=1e-3, penalty="l2", eta0=0.1)

    if self._poly_degree > 1:
      # Scale features to make SGD converge faster
      self._clf = Pipeline([
        ('poly_features', self._poly_features),
        ('std_scaler', StandardScaler()),
        ('classifier', clf)
      ])
      self._clf.fit(X, y.ravel())

      # Extract computed intercept and coefficient values
      self.intercept_, self.coef_ = self._clf['classifier'].intercept_.item(), self._clf['classifier'].coef_[0]
    else:
      self._clf = clf
      self._clf.fit(X, y.ravel())

       # Extract computed intercept and coefficient values
      self.intercept_, self.coef_ = self._clf.intercept_.item(), self._clf.coef_[0]

  def predict(self, X_new):
    if self._algorithm == "normal":
      X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new] # add x0 = 1 to each instance
      y_predict = X_new_b.dot(np.array([[self.intercept_], [self.coef_]]))
      return y_predict

    # Obtain predictions using fitted model
    return self._clf.predict(X_new)
  
def main():
  # Linear data: generate fake training data
  m = 100
  X = 5 * np.random.rand(m, 1)
  y = 8 + 10 * X + np.random.randn(m, 1)
  poly_degree = 1

  # Non-linear data: generate fake training data
  # m = 100
  # X = 6 * np.random.rand(m, 1) - 3
  # y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
  # poly_degree = 2

  # Test data
  X_new = np.array([[0],[1],[2],[3]])

  # Run linear regression with chosen algorithm and obtain predictions
  algorithms = ["normal", "SVD", "SGD"]
  for algo in algorithms:
    lin_reg = LinearRegressionWrapper(algorithm=algo, poly_degree=poly_degree)
    lin_reg.fit(X, y)
    print(f"Linear regression results (with {algo} algorithm):\nIntercept: {lin_reg.intercept_}, Coefficients: {lin_reg.coef_}")
    preds = lin_reg.predict(X_new)
    print(f"Predictions: {preds}")

if __name__ == "__main__":
  main()
