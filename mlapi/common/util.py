from flask_restful import reqparse, abort, Api, Resource
import joblib

from ml import models

## API helpers
def abort_if_model_doesnt_exist(model_name):
  if model_name not in models.SUPPORTED_MODELS:
    abort(404, message=f"Model {model_name} doesn't exist")

## Model helpers
# Loads a pickled model from disk
def load_model(filepath):
  with open(filepath, 'rb') as f:
    return joblib.load(f)
  return None