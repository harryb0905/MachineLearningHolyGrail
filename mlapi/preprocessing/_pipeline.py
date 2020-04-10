from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

class DefaultPipeline:
  ''' 
  Creates a default preprocessing pipeline using scikit-learn.
  Numerical attributes are scaled using standardization.
  Categorical attributes are transformed to a numerical representation using one-hot encoding.
  Missing values for numerical attributes are imputed using the median value of that attribute.
  Missing values for categorical attributes are imputed using the model value of that attribute.
  '''
  def __init__(self):
    pass

  def run(self, X_train_df, num_attribs=None, cat_attribs=None):
    # Get numerical and categorical attributes
    if num_attribs is None:
      num_attribs = [i for i in X_train_df.columns if X_train_df.dtypes[i] in ['int64', 'float64']]
    if cat_attribs is None:
      cat_attribs = [i for i in X_train_df.columns if X_train_df.dtypes[i] == 'object']

    num_pipeline = Pipeline([
      ("select_numeric", DataFrameSelector(num_attribs)),
      ('imputer', SimpleImputer(strategy="median")),
      ('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
      ('selector', DataFrameSelector(cat_attribs)),
      ('imputer', SimpleImputer(strategy="most_frequent")),
      ('cat_encoder', OneHotEncoder(sparse=False)),
    ])

    full_pipeline = ColumnTransformer([
      ("num", num_pipeline, num_attribs),
      ("cat", cat_pipeline, cat_attribs),
    ])

    return full_pipeline.fit_transform(X_train_df)
