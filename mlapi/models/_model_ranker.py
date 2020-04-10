from progress.bar import FillingSquaresBar
import time
import random
import os
import sys

# Use PYTHONPATH instead
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import fileopts as fo

SUPPORTED_MODEL_NAMES = [
  'LinearRegression',
  'LogisticRegression',
  'RandomForest'
]

class ModelRanker:
  def __init__(self):
    pass

  def compile(self, model_names):
    for model_name in model_names:
      bar = FillingSquaresBar(f"Training {model_name} model", max=10)
      for i in range(10):
          # Do some work
          bar.next()
          time.sleep(random.randint(0, 2))
      bar.finish()

def main():

  train_data = fo.read_file_as_df("datasets/titanic/train.csv")
  test_data = fo.read_file_as_df("datasets/titanic/test.csv")

  ranker = ModelRanker()
  ranker.compile(
    model_names=['LinearRegression', 'RandomForest']
  )

if __name__ == '__main__':
  main()

