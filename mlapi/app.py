from flask import Flask, Blueprint
from flask_restful import reqparse, Api
import resources

# Initialise application
app = Flask(__name__)
api_bp = Blueprint('api', __name__)
api = Api(api_bp)

# Argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

# Setup the Api resource routing here - route the URL to the resource
api.add_resource(resources.Models, '/models')
api.add_resource(resources.Model, '/models/<string:model_name>')
api.add_resource(resources.TestModel, '/models/test_model')
app.register_blueprint(api_bp)

if __name__ == '__main__':
  app.run(debug=True)