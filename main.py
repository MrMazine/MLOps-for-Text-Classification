"""
Main entry point for the fake news classification MLOps system.
"""
import os
import logging
from flask import Flask, render_template, redirect, url_for

from app import create_app

# Create Flask application
app = create_app()

@app.route('/')
def index():
    """Render the main page of the application."""
    return render_template('index.html')

@app.route('/datasets')
def datasets():
    """Render the datasets page."""
    return render_template('datasets.html')

@app.route('/preprocessing')
def preprocessing():
    """Render the preprocessing page."""
    return render_template('preprocessing.html')

@app.route('/models')
def models():
    """Render the models page."""
    return render_template('models.html')

@app.route('/visualization')
def visualization():
    """Render the visualization page."""
    return render_template('visualization.html')

@app.route('/setup_test_data')
def setup_test():
    """Setup test data for quick demonstration."""
    # Redirect to the dataset_routes.setup_test_data endpoint
    return redirect(url_for('dataset.setup_test_data'))

if __name__ == '__main__':
    # Run the application
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)