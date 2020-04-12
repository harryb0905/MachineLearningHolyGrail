from flask_restful import Resource
from ml.models import SUPPORTED_MODELS

# Returns list of all currently supported models
class Models(Resource):
  def __init__(self, **kwargs):
    pass

  def get(self):
    return [*SUPPORTED_MODELS]