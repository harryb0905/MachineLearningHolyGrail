from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

class LogisticRegressionWrapper:
  def __init__(self):
    pass
    
  def fit(self, X_train, y_train, k=10):
    lin_reg = LogisticRegression(random_state=42)
    lin_reg_scores = cross_val_score(lin_reg, X_train, y_train, cv=k)
    return {
      f'CVMean': lin_reg_scores.mean()
    }