from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class RandomForestWrapper:
  def __init__(self):
    pass
    
  def fit(self, X_train, y_train, k=10):
    forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=k)
    return {
      f'CVMean': forest_scores.mean()
    }