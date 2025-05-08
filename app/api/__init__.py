"""
API blueprints for fake news classification system.
"""
from flask import Blueprint

# Create blueprints
dataset_bp = Blueprint('dataset', __name__, url_prefix='/api/dataset')
preprocessing_bp = Blueprint('preprocessing', __name__, url_prefix='/api/preprocessing')
model_bp = Blueprint('model', __name__, url_prefix='/api/model')
visualization_bp = Blueprint('visualization', __name__, url_prefix='/api/visualization')

# Import routes to register them with blueprints
from app.api import dataset_routes, preprocessing_routes, model_routes, visualization_routes