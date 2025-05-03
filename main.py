import os
import logging
from flask import Flask, render_template, request, jsonify, send_file
import yaml
import tempfile
import shutil
import time
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

from src.config import Config
from src.data.data_loader import DataLoader
from src.data.preprocessing import TextPreprocessor
from src.data.data_validator import DataValidator
from dvc_helpers.dvc_setup import DVCHandler
from dvc_helpers.remote_storage import RemoteStorageManager
from src.utils.logging_utils import setup_logging
from src.utils.decorators import timing_decorator
from models import Base, Dataset, DatasetVersion, DatasetStats, ProcessingTask, RemoteStorage

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-dev-secret")

# Load configuration
config = Config()

# Configure database
database_url = os.environ.get("DATABASE_URL")
if database_url:
    engine = create_engine(database_url, pool_pre_ping=True)
    db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
    
    # Drop all tables and recreate them to ensure schema is up to date
    # (This is fine for development but should be handled by migrations in production)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables dropped and recreated with current schema")
else:
    logger.error("DATABASE_URL environment variable not set")
    db_session = None

# Initialize components
dvc_handler = DVCHandler(config.dvc_repo_path)
remote_storage = RemoteStorageManager(config)
data_validator = DataValidator()

@app.teardown_appcontext
def shutdown_session(exception=None):
    """Close the database session at the end of the request."""
    if db_session is not None:
        db_session.remove()

@app.route('/')
def index():
    """Render the main page of the application."""
    try:
        # Get DVC information
        dvc_info = dvc_handler.get_dvc_info()
        
        # Get all datasets (local and linked)
        try:
            # First try using database
            if db_session is not None:
                db_datasets = db_session.query(Dataset).all()
                datasets = []
                
                for dataset in db_datasets:
                    # For local datasets, get additional info from DVC
                    if dataset.path and not dataset.url:
                        dvc_datasets = dvc_handler.list_datasets()
                        dvc_dataset = next((d for d in dvc_datasets if d['path'] == dataset.path), None)
                        if dvc_dataset:
                            dataset_info = dvc_dataset
                            dataset_info['name'] = dataset.name
                            dataset_info['description'] = dataset.description
                            dataset_info['url'] = None
                            datasets.append(dataset_info)
                    # For external links
                    elif dataset.url:
                        datasets.append({
                            'name': dataset.name,
                            'path': dataset.path,
                            'url': dataset.url,
                            'description': dataset.description,
                            'created_at': dataset.created_at.strftime('%Y-%m-%d %H:%M:%S')
                        })
            else:
                # Fallback to DVC datasets if database is not available
                datasets = dvc_handler.list_datasets()
        except Exception as db_error:
            logger.error(f"Error getting datasets from database: {str(db_error)}")
            # Fallback to DVC datasets
            datasets = dvc_handler.list_datasets()
        
        return render_template('index.html', 
                              dvc_info=dvc_info,
                              datasets=datasets)
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return render_template('index.html', error=str(e))

@app.route('/upload_dataset', methods=['POST'])
@timing_decorator
def upload_dataset():
    """Upload and version a new dataset."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        dataset_name = request.form.get('dataset_name', 'unnamed_dataset')
        description = request.form.get('description', '')
        dataset_url = request.form.get('dataset_url', '')  # Added dataset URL
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        # Create temporary file path
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        file.save(temp_file_path)
        
        # Validate the dataset
        is_valid, validation_msg = data_validator.validate_dataset(temp_file_path)
        if not is_valid:
            shutil.rmtree(temp_dir)
            return jsonify({"error": f"Invalid dataset: {validation_msg}"}), 400
        
        # Copy file to raw data directory
        dataset_path = os.path.join(config.raw_data_path, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)
        dest_path = os.path.join(dataset_path, file.filename)
        shutil.copy(temp_file_path, dest_path)
        
        # Add and version the dataset with DVC
        commit_message = f"Added new dataset: {dataset_name}"
        success = dvc_handler.add_and_commit_dataset(dest_path, commit_message)
        
        if success and db_session is not None:
            # Get file size
            file_size = os.path.getsize(dest_path)
            
            # Get file type
            _, file_ext = os.path.splitext(file.filename)
            file_type = file_ext.lower()[1:] if file_ext else 'unknown'
            
            try:
                # Store dataset info in database
                dataset = Dataset(
                    name=dataset_name,
                    path=dest_path,
                    url=dataset_url,  # Added URL
                    file_type=file_type,
                    size_bytes=file_size,
                    description=description
                )
                db_session.add(dataset)
                
                # Get commit hash from the latest version
                versions = dvc_handler.get_dataset_versions(dest_path)
                if versions and len(versions) > 0:
                    latest_version = versions[0]
                    dataset_version = DatasetVersion(
                        dataset_id=dataset.id,
                        commit_hash=latest_version['hash'],
                        message=commit_message,
                        author=latest_version['author'],
                        is_current=True
                    )
                    db_session.add(dataset_version)
                
                # Commit the transaction
                db_session.commit()
                logger.info(f"Dataset {dataset_name} saved to database")
            except Exception as db_error:
                if db_session is not None:
                    db_session.rollback()
                logger.error(f"Database error: {str(db_error)}")
                # Continue with the process even if database fails
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return jsonify({"success": True, "message": f"Dataset {dataset_name} uploaded and versioned successfully"}), 200
    
    except Exception as e:
        if db_session is not None:
            db_session.rollback()
        logger.error(f"Error in upload_dataset: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/preprocess_dataset', methods=['POST'])
@timing_decorator
def preprocess_dataset():
    """Preprocess a dataset and version the processed data."""
    try:
        data = request.json
        dataset_path = data.get('dataset_path')
        preprocessing_steps = data.get('preprocessing_steps', [])
        
        if not dataset_path:
            return jsonify({"error": "Dataset path is required"}), 400
        
        # Initialize preprocessor with the selected steps
        preprocessor = TextPreprocessor(preprocessing_steps)
        
        # Load the dataset
        data_loader = DataLoader(dataset_path)
        
        # Generate processed dataset path
        dataset_name = os.path.basename(dataset_path)
        processed_path = os.path.join(config.processed_data_path, f"processed_{dataset_name}")
        
        # Process the dataset
        preprocessor.preprocess_and_save(data_loader, processed_path)
        
        # Version the processed dataset
        dvc_handler.add_and_commit_dataset(processed_path, f"Preprocessed dataset: {dataset_name}")
        
        return jsonify({
            "success": True, 
            "message": f"Dataset preprocessed and versioned successfully",
            "processed_path": processed_path
        }), 200
    
    except Exception as e:
        logger.error(f"Error in preprocess_dataset: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/dataset_versions', methods=['GET'])
def dataset_versions():
    """Get all versions of a specific dataset."""
    try:
        dataset_path = request.args.get('dataset_path')
        if not dataset_path:
            return jsonify({"error": "Dataset path is required"}), 400
        
        versions = dvc_handler.get_dataset_versions(dataset_path)
        return jsonify({"versions": versions}), 200
    
    except Exception as e:
        logger.error(f"Error in dataset_versions: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/checkout_version', methods=['POST'])
@timing_decorator
def checkout_version():
    """Checkout a specific version of a dataset."""
    try:
        data = request.json
        dataset_path = data.get('dataset_path')
        version = data.get('version')
        
        if not dataset_path or not version:
            return jsonify({"error": "Dataset path and version are required"}), 400
        
        success = dvc_handler.checkout_version(dataset_path, version)
        if success:
            return jsonify({
                "success": True,
                "message": f"Successfully checked out version {version} of dataset"
            }), 200
        else:
            return jsonify({"error": "Failed to checkout version"}), 500
    
    except Exception as e:
        logger.error(f"Error in checkout_version: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/sync_remote', methods=['POST'])
@timing_decorator
def sync_with_remote():
    """Sync local DVC repository with remote storage."""
    try:
        action = request.json.get('action', 'push')  # 'push' or 'pull'
        
        if action == 'push':
            result = remote_storage.push_to_remote()
        else:
            result = remote_storage.pull_from_remote()
        
        if result['success']:
            return jsonify({
                "success": True,
                "message": result['message']
            }), 200
        else:
            return jsonify({"error": result['message']}), 500
    
    except Exception as e:
        logger.error(f"Error in sync_with_remote: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/dataset_stats', methods=['GET'])
def dataset_stats():
    """Get statistics for a dataset."""
    try:
        dataset_path = request.args.get('dataset_path')
        if not dataset_path:
            return jsonify({"error": "Dataset path is required"}), 400
        
        # Load dataset with memory-efficient generator
        data_loader = DataLoader(dataset_path)
        
        # Calculate statistics
        stats = data_loader.get_statistics()
        
        return jsonify({"stats": stats}), 200
    
    except Exception as e:
        logger.error(f"Error in dataset_stats: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/download_dataset', methods=['GET'])
def download_dataset():
    """Download a dataset."""
    try:
        dataset_path = request.args.get('dataset_path')
        if not dataset_path or not os.path.exists(dataset_path):
            return jsonify({"error": "Invalid dataset path"}), 400
        
        # Create a temporary zip file of the dataset
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "dataset.zip")
        
        # Create zip file
        shutil.make_archive(zip_path[:-4], 'zip', os.path.dirname(dataset_path), os.path.basename(dataset_path))
        
        # Send the file
        return send_file(zip_path, as_attachment=True, download_name=f"{os.path.basename(dataset_path)}.zip")
    
    except Exception as e:
        logger.error(f"Error in download_dataset: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/add_dataset_link', methods=['POST'])
@timing_decorator
def add_dataset_link():
    """Add a link to an external dataset."""
    try:
        data = request.json
        dataset_name = data.get('dataset_name')
        dataset_url = data.get('dataset_url')
        description = data.get('description', '')
        
        if not dataset_name or not dataset_url:
            return jsonify({"error": "Dataset name and URL are required"}), 400
        
        if db_session is not None:
            try:
                # Store dataset info in database
                dataset = Dataset(
                    name=dataset_name,
                    path="",  # Empty path since it's an external link
                    url=dataset_url,
                    file_type="external",
                    description=description
                )
                db_session.add(dataset)
                db_session.commit()
                logger.info(f"Dataset link for {dataset_name} saved to database")
                
                return jsonify({
                    "success": True,
                    "message": f"Dataset link for {dataset_name} added successfully"
                }), 200
            except Exception as db_error:
                if db_session is not None:
                    db_session.rollback()
                logger.error(f"Database error: {str(db_error)}")
                return jsonify({"error": f"Database error: {str(db_error)}"}), 500
        else:
            return jsonify({"error": "Database connection not available"}), 500
    
    except Exception as e:
        logger.error(f"Error in add_dataset_link: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/list_datasets', methods=['GET'])
def list_datasets():
    """List all datasets including local and external links."""
    try:
        datasets = []
        
        # Get datasets from DVC
        local_datasets = dvc_handler.list_datasets()
        
        # Get datasets from database
        if db_session is not None:
            db_datasets = db_session.query(Dataset).all()
            for dataset in db_datasets:
                datasets.append({
                    "id": dataset.id,
                    "name": dataset.name,
                    "path": dataset.path,
                    "url": dataset.url,
                    "file_type": dataset.file_type,
                    "size_bytes": dataset.size_bytes,
                    "description": dataset.description,
                    "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
                })
        else:
            # If no database, just use DVC datasets
            datasets = local_datasets
        
        return jsonify({"datasets": datasets}), 200
    
    except Exception as e:
        logger.error(f"Error in list_datasets: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure data directories exist
    os.makedirs(config.raw_data_path, exist_ok=True)
    os.makedirs(config.processed_data_path, exist_ok=True)
    
    # Initialize DVC repository if it doesn't exist
    if not dvc_handler.is_dvc_initialized():
        dvc_handler.initialize_dvc()
        logger.info("DVC repository initialized")
    
    # Run the Flask application
    # Add a setup route to initialize test data
    @app.route('/setup_test_data')
    def setup_test_data():
        """Set up test data for quick testing. Uses spam.csv from the root directory."""
        try:
            source_path = 'spam.csv'
            if not os.path.exists(source_path):
                return jsonify({"error": "Test data file spam.csv not found in the root directory"}), 404
            
            # Create test data directory
            test_data_dir = os.path.join(config.raw_data_path, 'spam_test')
            os.makedirs(test_data_dir, exist_ok=True)
            
            # Copy the spam.csv file to the test directory
            dest_path = os.path.join(test_data_dir, 'spam.csv')
            shutil.copy(source_path, dest_path)
            logger.info(f"Copied test data to {dest_path}")
            
            # Add external URL dataset link
            if db_session is not None:
                # Store dataset info in database for Kaggle Spam dataset
                dataset = Dataset(
                    name="Kaggle SMS Spam Collection",
                    path="",  # Empty path since it's an external link
                    url="https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset",
                    file_type="external",
                    description="The SMS Spam Collection is a public set of SMS labeled messages that have been collected for mobile phone spam research."
                )
                db_session.add(dataset)
                db_session.commit()
                logger.info("Added external Kaggle dataset link to database")
            
            return jsonify({
                "success": True,
                "message": "Test data has been set up successfully! Check the 'Local Datasets' and 'Linked Datasets' tabs.",
                "local_path": dest_path,
                "external_added": True if db_session is not None else False
            }), 200
        
        except Exception as e:
            logger.error(f"Error setting up test data: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    app.run(host='0.0.0.0', port=5000, debug=True)
