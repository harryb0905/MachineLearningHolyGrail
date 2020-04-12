from flask import Flask, jsonify
from flask_restful import reqparse, abort, Api, Resource
import joblib
import numpy as np
from ml import models
# from model import NLPModel

app = Flask(__name__)
api = Api(app)

# model = NLPModel()

# Loads a pickled model from disk
def load_model(filepath):
  with open(filepath, 'rb') as f:
    model = joblib.load(f) # model.clf?

def abort_if_model_doesnt_exist(model_name):
  if model_name not in models.SUPPORTED_MODELS:
    abort(404, message=f"Model {model_name} doesn't exist")

# Argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

# Returns list of all currently supported models
class Models(Resource):
  def get(self):
    return [*models.SUPPORTED_MODELS]

# Runs a specified model
class Model(Resource):
  def get(self, model_name):
    abort_if_model_doesnt_exist(model_name)
    print(model_name)

# Setup the Api resource routing here - route the URL to the resource
api.add_resource(Models, '/models')
api.add_resource(Model, '/models/<model_name>')

if __name__ == '__main__':
  app.run(debug=True)

