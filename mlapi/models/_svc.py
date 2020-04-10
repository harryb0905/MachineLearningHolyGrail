from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

class SVCWrapper:
  def __init__(self):
    pass
    
  def fit(self, X_train, y_train, k=10):
    svc_clf = SVC(gamma="auto", random_state=42)
    svc_scores = cross_val_score(svc_clf, X_train, y_train, cv=k)
    return {
      f'CVMean': svc_scores.mean()
    }