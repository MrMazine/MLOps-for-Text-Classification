#!/usr/bin/env python3
"""
Setup script for the Fake News Classification MLOps system.
This script initializes the project environment, database, and DVC.
"""
import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger("setup")

def run_command(command, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def create_directories():
    """Create necessary directories for the project."""
    dirs = [
        "data/raw",
        "data/processed",
        "data/external",
        "logs",
        "models/svm",
        "models/naive_bayes",
        "models/transformers",
        "notebooks",
        "config/env",
        "config/templates"
    ]
    
    for directory in dirs:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    return True

def initialize_dvc():
    """Initialize DVC repository."""
    logger.info("Initializing DVC...")
    
    # Check if DVC is already initialized
    if Path(".dvc").exists():
        logger.info("DVC already initialized.")
        return True
    
    # Initialize DVC
    success, stdout, stderr = run_command("dvc init")
    if not success:
        logger.error(f"Failed to initialize DVC: {stderr}")
        return False
    
    # Configure DVC cache
    cache_dir = Path(".dvc/cache")
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("DVC initialized successfully.")
    return True

def setup_database():
    """Set up the database."""
    logger.info("Setting up database...")
    
    # Check if DATABASE_URL is set
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.warning("DATABASE_URL environment variable not set.")
        logger.info("Using default SQLite database.")
        os.environ["DATABASE_URL"] = "sqlite:///fakenews.db"
    
    # Run database initialization script
    try:
        logger.info("Importing database modules...")
        from app.utils.database import db
        from app import create_app
        
        app = create_app()
        with app.app_context():
            logger.info("Creating database tables...")
            db.create_all()
            
        logger.info("Database setup complete.")
        return True
    except Exception as e:
        logger.error(f"Failed to set up database: {str(e)}")
        return False

def setup_environment():
    """Set up environment variables."""
    logger.info("Setting up environment variables...")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        logger.info("Creating .env file...")
        with open(env_file, "w") as f:
            f.write("# Environment variables for Fake News Classification\n")
            f.write("FLASK_APP=main.py\n")
            f.write("FLASK_ENV=development\n")
            f.write("FLASK_DEBUG=1\n")
            f.write("FLASK_SECRET_KEY=fake_news_detection_dev_key\n")
            f.write("# DATABASE_URL=postgresql://user:password@localhost/fakenews\n")
    
    logger.info("Environment setup complete.")
    return True

def download_nltk_resources():
    """Download required NLTK resources."""
    logger.info("Downloading NLTK resources...")
    
    try:
        import nltk
        resources = ["punkt", "stopwords", "wordnet"]
        for resource in resources:
            try:
                nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
                logger.info(f"NLTK resource '{resource}' already downloaded.")
            except LookupError:
                logger.info(f"Downloading NLTK resource '{resource}'...")
                nltk.download(resource)
        
        logger.info("NLTK resources downloaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to download NLTK resources: {str(e)}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    required_version = (3, 10)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        logger.warning(f"Python {required_version[0]}.{required_version[1]} or higher is required."
                      f" You are using Python {current_version[0]}.{current_version[1]}.")
        return False
    
    return True

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup script for Fake News Classification")
    parser.add_argument("--skip-dvc", action="store_true", help="Skip DVC initialization")
    parser.add_argument("--skip-db", action="store_true", help="Skip database setup")
    parser.add_argument("--skip-nltk", action="store_true", help="Skip NLTK resource download")
    args = parser.parse_args()
    
    logger.info("Starting setup...")
    
    # Check Python version
    if not check_python_version():
        logger.warning("Setup will continue, but you may encounter issues.")
    
    # Create directories
    if not create_directories():
        logger.error("Failed to create directories.")
        return 1
    
    # Setup environment
    if not setup_environment():
        logger.error("Failed to set up environment.")
        return 1
    
    # Initialize DVC
    if not args.skip_dvc and not initialize_dvc():
        logger.error("Failed to initialize DVC.")
        return 1
    
    # Download NLTK resources
    if not args.skip_nltk and not download_nltk_resources():
        logger.error("Failed to download NLTK resources.")
        return 1
    
    # Setup database
    if not args.skip_db and not setup_database():
        logger.error("Failed to set up database.")
        return 1
    
    logger.info("Setup completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())