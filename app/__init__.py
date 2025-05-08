"""
Fake News Classification MLOps System.

This package contains the Flask application and core functionality for
a fake news classification system with Data Version Control (DVC)
and memory-efficient data processing.
"""
import os
import logging
from flask import Flask
from app.utils.database import init_app as init_db

def create_app(test_config=None):
    """
    Factory function to create and configure the Flask application.
    
    Args:
        test_config: Configuration dictionary for testing
        
    Returns:
        Configured Flask application
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(f"logs/fake_news_classification_{logging.Formatter().converter().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized.")

    # Create and configure the app
    app = Flask(__name__, 
                static_folder='static',
                template_folder='templates')
    
    # Setup a secret key, required by sessions
    app.secret_key = os.environ.get("FLASK_SECRET_KEY") or "fake_news_detection_secret_key"
    
    # Load configuration
    if test_config is None:
        # Load the config.yaml file in non-testing mode
        from app.utils.config import Config
        config = Config()
        app.config.from_mapping(
            CONFIG=config,
        )
        logger.info("Loaded configuration from config.yaml")
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)
        logger.info("Loaded test configuration")
    
    # Initialize database
    db = init_db(app)
    
    # Ensure the instance folder exists
    try:
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('data/external', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    except OSError:
        pass

    # Register blueprints
    from app.api import dataset_bp, preprocessing_bp, model_bp, visualization_bp
    app.register_blueprint(dataset_bp)
    app.register_blueprint(preprocessing_bp)
    app.register_blueprint(model_bp)
    app.register_blueprint(visualization_bp)
    
    # Add close_db function to app teardown
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        """Close the database session at the end of the request."""
        db.session.remove()
    
    return app