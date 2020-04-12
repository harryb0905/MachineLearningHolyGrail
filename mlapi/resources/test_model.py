from flask import jsonify
from flask_restful import Resource, fields, marshal_with, request
import pandas as pd
import json
import pickle
from common.util import *

# importing models (from https://hackernoon.com/machine-learning-w22g322x)
with open('ml/datasets/boston/model/model.pkl', 'rb') as f:
  classifier = pickle.load(f)
 
with open('ml/datasets/boston/model/model_columns.pkl', 'rb') as f:
  model_columns = pickle.load(f)

# Runs a test model on the boston house price dataset
class TestModel(Resource):
  def __init__(self):
    self.reqparse = reqparse.RequestParser()
    self.reqparse.add_argument('inputs', type=list, location='json')

  def post(self):
    self.args = self.reqparse.parse_args()
    inputs = self.args['inputs']
    df = pd.DataFrame(inputs)

    # Construct query and obtain prediction
    query_ = pd.get_dummies(df)
    prediction = list(classifier.predict(df))
    print(prediction)

    return jsonify({
      "prediction": str(prediction)
    })



