from flask_restful import Resource
from common.util import *

# from model import NLPModel
# model = NLPModel()

# Runs a specified model
class Model(Resource):
  def get(self, model_name):
    abort_if_model_doesnt_exist(model_name)
    return {
      'model': model_name
    }