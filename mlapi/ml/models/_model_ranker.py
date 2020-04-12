import time
import random
import os
import sys

# Use PYTHONPATH instead
# BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_PATH)

import fileopts as fo
from ml.preprocessing import DefaultPipeline
import ml.models

# Number of folds for k-fold cross-validation
K = 10

class ModelRanker:
  def __init__(self):
    pass

  def compile(self, X_train_df, y_train, num_attribs, cat_attribs, model_names, order_by):
    # Preprocess data
    X_train_df_preprocessed = DefaultPipeline().run(X_train_df, num_attribs, cat_attribs)

    ranked_models = []

    # Run all models and evaluate using k-fold CV
    for model_name in model_names:
      print(f"Training {model_name} model...")
      model = SUPPORTED_MODELS[model_name]()
      train_results = model.fit(X_train_df_preprocessed, y_train, K)
      print(f"{K}-fold cross-validation mean: {train_results[order_by]}")
      ranked_models.append((model_name, train_results))

    # Rank models by specified score
    ranked_models.sort(key=lambda x: x[1][order_by], reverse=True)
    print(f"Ranked results: {ranked_models}")

def main():
  # Load titanic dataset
  train_df = fo.read_file_as_df("datasets/titanic/train.csv")
  test_df = fo.read_file_as_df("datasets/titanic/test.csv")

  # Drop unwanted columns and keep labels separate
  X_train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1)
  y_train = train_df['Survived']
  X_test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

  # Choose numerical/categorical attributes (if not chosen here, all will be used)
  num_attribs = ["Age", "SibSp", "Parch", "Fare"]
  cat_attribs = ["Pclass", "Sex", "Embarked"]

  ranker = ModelRanker().compile(
    X_train_df=X_train_df,
    y_train=y_train,
    num_attribs=num_attribs,
    cat_attribs=cat_attribs,
    model_names=['LogisticRegression', 'RandomForest', 'SVC'],
    order_by="CVMean"
  )

if __name__ == '__main__':
  main()


