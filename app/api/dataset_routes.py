"""
Dataset API routes for fake news classification system.
"""
import os
import json
import logging
import shutil
from datetime import datetime
from flask import request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
from app.utils.database import db
from app.models.database import NewsDataset, DatasetVersion, DatasetStats
from app.utils.dvc_handler import DVCHandler
from app.api import dataset_bp

logger = logging.getLogger(__name__)

@dataset_bp.route('/upload', methods=['POST'])
def upload_dataset():
    """Upload and version a new news dataset."""
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400
        
    if file:
        # Get form data
        name = request.form.get('name', file.filename)
        description = request.form.get('description', '')
        category = request.form.get('category', 'general')
        
        # Secure the filename and create directory structure
        filename = secure_filename(file.filename)
        dataset_dir = os.path.join(current_app.config['CONFIG']['data']['raw_dir'], 
                                  secure_filename(name.lower().replace(' ', '_')))
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save the file
        file_path = os.path.join(dataset_dir, filename)
        file.save(file_path)
        logger.info(f"Saved uploaded file to {file_path}")
        
        # Get basic file stats
        file_size = os.path.getsize(file_path)
        file_type = os.path.splitext(filename)[1].lstrip('.')
        
        # Create dataset record in database
        dataset = NewsDataset(
            name=name,
            path=file_path,
            file_type=file_type,
            created_at=datetime.utcnow(),
            size_bytes=file_size,
            description=description,
            category=category
        )
        db.session.add(dataset)
        db.session.commit()
        
        # Initialize DVC and add dataset
        dvc_handler = DVCHandler(os.getcwd())
        if not dvc_handler.is_dvc_initialized():
            dvc_handler.initialize_dvc()
            
        # Add dataset to DVC and create version
        commit_message = f"Added dataset: {name}"
        success = dvc_handler.add_and_commit_dataset(file_path, commit_message)
        
        if success:
            # Get the commit hash from DVC
            versions = dvc_handler.get_dataset_versions(file_path)
            commit_hash = versions[0]['commit_hash'] if versions else "initial"
            
            # Create dataset version record
            version = DatasetVersion(
                dataset_id=dataset.id,
                commit_hash=commit_hash,
                message=commit_message,
                author=request.form.get('author', 'system'),
                date_created=datetime.utcnow(),
                is_current=True
            )
            db.session.add(version)
            db.session.commit()
            
            return jsonify({
                "success": True,
                "message": "Dataset uploaded and versioned successfully",
                "dataset_id": dataset.id,
                "version_id": version.id
            })
        else:
            return jsonify({
                "success": True,
                "message": "Dataset uploaded but not versioned",
                "dataset_id": dataset.id,
                "warning": "DVC versioning failed"
            })
    
    return jsonify({"success": False, "message": "File upload failed"}), 500

@dataset_bp.route('/link', methods=['POST'])
def add_dataset_link():
    """Add a link to an external news dataset."""
    data = request.json
    
    if not data or 'url' not in data or 'name' not in data:
        return jsonify({"success": False, "message": "Missing required fields"}), 400
        
    # Create dataset record in database
    dataset = NewsDataset(
        name=data['name'],
        path="",  # No local path for linked datasets
        url=data['url'],
        file_type=data.get('file_type', ''),
        created_at=datetime.utcnow(),
        size_bytes=0,  # Unknown size for external datasets
        description=data.get('description', ''),
        category=data.get('category', 'general')
    )
    db.session.add(dataset)
    db.session.commit()
    
    logger.info(f"Added external dataset link to database: {data['url']}")
    
    return jsonify({
        "success": True,
        "message": "External dataset link added successfully",
        "dataset_id": dataset.id
    })

@dataset_bp.route('/list', methods=['GET'])
def list_datasets():
    """List all news datasets including local and external links."""
    # Get filter parameters
    dataset_type = request.args.get('type', 'all')  # 'local', 'external', or 'all'
    category = request.args.get('category', None)
    
    # Build query
    query = db.session.query(NewsDataset)
    
    if dataset_type == 'local':
        query = query.filter(NewsDataset.path != "")
    elif dataset_type == 'external':
        query = query.filter(NewsDataset.url != None)
        
    if category:
        query = query.filter(NewsDataset.category == category)
        
    # Execute query and format results
    datasets = []
    for dataset in query.all():
        datasets.append({
            "id": dataset.id,
            "name": dataset.name,
            "path": dataset.path,
            "url": dataset.url,
            "file_type": dataset.file_type,
            "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
            "record_count": dataset.record_count,
            "size_bytes": dataset.size_bytes,
            "description": dataset.description,
            "category": dataset.category
        })
    
    return jsonify({
        "success": True,
        "count": len(datasets),
        "datasets": datasets
    })

@dataset_bp.route('/<int:dataset_id>/versions', methods=['GET'])
def dataset_versions(dataset_id):
    """Get all versions of a specific news dataset."""
    dataset = db.session.query(NewsDataset).get(dataset_id)
    if not dataset:
        return jsonify({"success": False, "message": "Dataset not found"}), 404
        
    versions = []
    for version in dataset.versions:
        versions.append({
            "id": version.id,
            "commit_hash": version.commit_hash,
            "message": version.message,
            "author": version.author,
            "date_created": version.date_created.isoformat() if version.date_created else None,
            "is_current": version.is_current
        })
        
    return jsonify({
        "success": True,
        "dataset": {
            "id": dataset.id,
            "name": dataset.name
        },
        "versions": versions
    })

@dataset_bp.route('/<int:dataset_id>/checkout/<int:version_id>', methods=['POST'])
def checkout_version(dataset_id, version_id):
    """Checkout a specific version of a news dataset."""
    dataset = db.session.query(NewsDataset).get(dataset_id)
    version = db.session.query(DatasetVersion).get(version_id)
    
    if not dataset or not version or version.dataset_id != dataset.id:
        return jsonify({"success": False, "message": "Dataset or version not found"}), 404
        
    # Update current version flag
    for v in dataset.versions:
        v.is_current = (v.id == version_id)
    db.session.commit()
    
    # If this is a local dataset, try to checkout the version
    if dataset.path:
        dvc_handler = DVCHandler(os.getcwd())
        success = dvc_handler.checkout_version(dataset.path, version.commit_hash)
        
        if success:
            message = "Dataset version checked out successfully"
        else:
            message = "Dataset version marked as current, but checkout failed"
            
        return jsonify({
            "success": True,
            "message": message,
            "dataset_id": dataset.id,
            "version_id": version.id
        })
    else:
        return jsonify({
            "success": True,
            "message": "External dataset version marked as current",
            "dataset_id": dataset.id,
            "version_id": version.id
        })

@dataset_bp.route('/<int:dataset_id>/stats', methods=['GET'])
def dataset_stats(dataset_id):
    """Get statistics for a news dataset."""
    dataset = db.session.query(NewsDataset).get(dataset_id)
    if not dataset:
        return jsonify({"success": False, "message": "Dataset not found"}), 404
        
    # Get the most recent stats if available
    stats = db.session.query(DatasetStats).filter(
        DatasetStats.dataset_id == dataset_id
    ).order_by(DatasetStats.created_at.desc()).first()
    
    if stats:
        stats_data = {
            "id": stats.id,
            "record_count": stats.record_count,
            "field_count": stats.field_count,
            "avg_title_length": stats.avg_title_length,
            "avg_article_length": stats.avg_article_length,
            "min_article_length": stats.min_article_length,
            "max_article_length": stats.max_article_length,
            "created_at": stats.created_at.isoformat() if stats.created_at else None
        }
        
        # Add parsed JSON fields if they exist
        for field in ['top_entities', 'sentiment_distribution', 'topic_distribution']:
            value = getattr(stats, field)
            if value:
                try:
                    stats_data[field] = json.loads(value)
                except:
                    stats_data[field] = value
    else:
        stats_data = None
        
    return jsonify({
        "success": True,
        "dataset": {
            "id": dataset.id,
            "name": dataset.name
        },
        "stats": stats_data
    })

@dataset_bp.route('/<int:dataset_id>/download', methods=['GET'])
def download_dataset(dataset_id):
    """Download a news dataset."""
    dataset = db.session.query(NewsDataset).get(dataset_id)
    if not dataset:
        return jsonify({"success": False, "message": "Dataset not found"}), 404
        
    if not dataset.path or not os.path.exists(dataset.path):
        return jsonify({"success": False, "message": "Dataset file not found"}), 404
        
    return send_file(dataset.path, as_attachment=True, 
                    download_name=os.path.basename(dataset.path))

@dataset_bp.route('/<int:dataset_id>/delete', methods=['DELETE'])
def delete_dataset(dataset_id):
    """Delete a news dataset."""
    dataset = db.session.query(NewsDataset).get(dataset_id)
    if not dataset:
        return jsonify({"success": False, "message": "Dataset not found"}), 404
        
    # If this is a local dataset, delete the file
    if dataset.path and os.path.exists(dataset.path):
        try:
            if os.path.isdir(os.path.dirname(dataset.path)):
                shutil.rmtree(os.path.dirname(dataset.path))
            else:
                os.remove(dataset.path)
        except Exception as e:
            logger.error(f"Failed to delete dataset file: {str(e)}")
    
    # Delete the database record
    name = dataset.name
    db.session.delete(dataset)
    db.session.commit()
    
    return jsonify({
        "success": True,
        "message": f"Dataset '{name}' deleted successfully"
    })

@dataset_bp.route('/test_setup', methods=['GET'])
def setup_test_data():
    """Set up test data for fake news detection using provided sample files."""
    try:
        # Create data directories if they don't exist
        data_dirs = [
            os.path.join('data', 'raw', 'fake_news_test'),
            os.path.join('data', 'raw', 'true_news_test')
        ]
        
        for dir_path in data_dirs:
            os.makedirs(dir_path, exist_ok=True)
            
        # Copy the fake news dataset
        fake_news_source = os.path.join('attached_assets', 'Fake - Copy.csv')
        fake_news_dest = os.path.join('data', 'raw', 'fake_news_test', 'fake_news.csv')
        
        # Copy the true news dataset
        true_news_source = os.path.join('attached_assets', 'True - Copy.csv')
        true_news_dest = os.path.join('data', 'raw', 'true_news_test', 'true_news.csv')
        
        shutil.copy(fake_news_source, fake_news_dest)
        shutil.copy(true_news_source, true_news_dest)
        
        logger.info(f"Copied test data to {fake_news_dest} and {true_news_dest}")
        
        # Add datasets to database
        fake_news_dataset = NewsDataset(
            name="Fake News Test Dataset",
            path=fake_news_dest,
            file_type="csv",
            created_at=datetime.utcnow(),
            size_bytes=os.path.getsize(fake_news_dest),
            description="Sample fake news dataset for testing",
            category="fake"
        )
        
        true_news_dataset = NewsDataset(
            name="True News Test Dataset",
            path=true_news_dest,
            file_type="csv",
            created_at=datetime.utcnow(),
            size_bytes=os.path.getsize(true_news_dest),
            description="Sample true news dataset for testing",
            category="true"
        )
        
        db.session.add(fake_news_dataset)
        db.session.add(true_news_dataset)
        db.session.commit()
        
        # Add external dataset link
        external_dataset = NewsDataset(
            name="Kaggle Fake News Dataset",
            path="",
            url="https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset",
            file_type="csv",
            created_at=datetime.utcnow(),
            description="Large dataset of fake and real news from Kaggle",
            category="external"
        )
        
        db.session.add(external_dataset)
        db.session.commit()
        
        logger.info("Added external Kaggle dataset link to database")
        
        # Add to DVC
        dvc_handler = DVCHandler(os.getcwd())
        fake_success = dvc_handler.add_and_commit_dataset(
            fake_news_dest, 
            "Added fake news test dataset"
        )
        true_success = dvc_handler.add_and_commit_dataset(
            true_news_dest, 
            "Added true news test dataset"
        )
        
        if not fake_success or not true_success:
            logger.warning("Could not add test dataset to DVC tracking")
            
        # Create initial versions
        fake_versions = dvc_handler.get_dataset_versions(fake_news_dest)
        true_versions = dvc_handler.get_dataset_versions(true_news_dest)
        
        fake_commit = fake_versions[0]['commit_hash'] if fake_versions else "initial"
        true_commit = true_versions[0]['commit_hash'] if true_versions else "initial"
        
        fake_version = DatasetVersion(
            dataset_id=fake_news_dataset.id,
            commit_hash=fake_commit,
            message="Initial version of fake news test dataset",
            author="system",
            date_created=datetime.utcnow(),
            is_current=True
        )
        
        true_version = DatasetVersion(
            dataset_id=true_news_dataset.id,
            commit_hash=true_commit,
            message="Initial version of true news test dataset",
            author="system",
            date_created=datetime.utcnow(),
            is_current=True
        )
        
        db.session.add(fake_version)
        db.session.add(true_version)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Test data has been set up successfully! Check the 'Local Datasets' and 'Linked Datasets' tabs.",
            "local_path": fake_news_dest,
            "external_added": True
        })
        
    except Exception as e:
        logger.error(f"Error setting up test data: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Failed to set up test data: {str(e)}"
        }), 500