from . import LogisticRegressionWrapper, SVCWrapper,RandomForestWrapper

SUPPORTED_MODELS = {
  'LogisticRegression': LogisticRegressionWrapper,
  'SVC': SVCWrapper,
  'RandomForest': RandomForestWrapper
}