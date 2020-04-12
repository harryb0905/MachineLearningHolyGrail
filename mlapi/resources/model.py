from flask_restful import Resource, fields, marshal_with
from common.util import *

# from model import NLPModel
# model = NLPModel()

model_fields = {
  'model_name': fields.String
}

# Runs a specified model
class Model(Resource):
  @marshal_with(model_fields)
  def get(self, model_name):
    abort_if_model_doesnt_exist(model_name)
    return {
      'model_name': model_name
    }, 200