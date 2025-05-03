import os
import logging
from flask import Flask, render_template, request, jsonify, send_file
import yaml
import tempfile
import shutil
import time

from src.config import Config
from src.data.data_loader import DataLoader
from src.data.preprocessing import TextPreprocessor
from src.data.data_validator import DataValidator
from dvc_helpers.dvc_setup import DVCHandler
from dvc_helpers.remote_storage import RemoteStorageManager
from src.utils.logging_utils import setup_logging
from src.utils.decorators import timing_decorator

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-dev-secret")

# Load configuration
config = Config()

# Initialize components
dvc_handler = DVCHandler(config.dvc_repo_path)
remote_storage = RemoteStorageManager(config)
data_validator = DataValidator()

@app.route('/')
def index():
    """Render the main page of the application."""
    try:
        # Get DVC information
        dvc_info = dvc_handler.get_dvc_info()
        # Get available datasets
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
        dvc_handler.add_and_commit_dataset(dest_path, f"Added new dataset: {dataset_name}")
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return jsonify({"success": True, "message": f"Dataset {dataset_name} uploaded and versioned successfully"}), 200
    
    except Exception as e:
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

if __name__ == '__main__':
    # Ensure data directories exist
    os.makedirs(config.raw_data_path, exist_ok=True)
    os.makedirs(config.processed_data_path, exist_ok=True)
    
    # Initialize DVC repository if it doesn't exist
    if not dvc_handler.is_dvc_initialized():
        dvc_handler.initialize_dvc()
        logger.info("DVC repository initialized")
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)
