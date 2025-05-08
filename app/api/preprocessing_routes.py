"""
Preprocessing API routes for fake news classification system.
"""
import os
import json
import logging
from datetime import datetime
from flask import request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.utils.database import db
from app.models.database import NewsDataset, DatasetVersion, ProcessingTask
from app.utils.dvc_handler import DVCHandler
from app.api import preprocessing_bp

# Import preprocessing functions
from app.utils.text_preprocessing import TextPreprocessor
from app.utils.feature_extraction import FeatureExtractor

logger = logging.getLogger(__name__)

@preprocessing_bp.route('/preprocess', methods=['POST'])
def preprocess_dataset():
    """Preprocess a news dataset and version the processed data."""
    data = request.json
    
    if not data or 'dataset_id' not in data:
        return jsonify({"success": False, "message": "Missing dataset ID"}), 400
        
    # Get dataset
    dataset_id = data['dataset_id']
    dataset = db.session.query(NewsDataset).get(dataset_id)
    
    if not dataset:
        return jsonify({"success": False, "message": "Dataset not found"}), 404
        
    if not dataset.path or not os.path.exists(dataset.path):
        return jsonify({"success": False, "message": "Dataset file not found"}), 404
        
    # Get current version
    current_version = None
    for version in dataset.versions:
        if version.is_current:
            current_version = version
            break
            
    if not current_version:
        return jsonify({"success": False, "message": "No current version found for dataset"}), 404
    
    # Set up preprocessing parameters
    config = current_app.config['CONFIG']['preprocessing']
    params = {
        'remove_stopwords': data.get('remove_stopwords', config['remove_stopwords']),
        'lowercase': data.get('lowercase', config['lowercase']),
        'remove_punctuation': data.get('remove_punctuation', config['remove_punctuation']),
        'remove_numbers': data.get('remove_numbers', config['remove_numbers']),
        'lemmatize': data.get('lemmatize', config['lemmatize']),
        'stemming': data.get('stemming', config['stemming']),
    }
    
    # Create processing task
    task = ProcessingTask(
        dataset_version_id=current_version.id,
        task_type='preprocessing',
        parameters=json.dumps(params),
        input_path=dataset.path,
        status='running',
        start_time=datetime.utcnow()
    )
    db.session.add(task)
    db.session.commit()
    
    try:
        # Determine output path
        output_dir = os.path.join(
            current_app.config['CONFIG']['data']['processed_dir'],
            secure_filename(dataset.name.lower().replace(' ', '_'))
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(dataset.path))[0]
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{base_name}_processed_{timestamp}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor(params)
        
        # Preprocess the dataset
        success, stats = preprocessor.preprocess_and_save(
            dataset.path,
            output_path,
            text_column=data.get('text_column', 'text')
        )
        
        if not success:
            task.status = 'failed'
            task.end_time = datetime.utcnow()
            task.error_message = "Preprocessing failed"
            db.session.commit()
            
            return jsonify({
                "success": False,
                "message": "Preprocessing failed",
                "task_id": task.id
            }), 500
            
        # Update task with output path and completion
        task.output_path = output_path
        task.status = 'completed'
        task.end_time = datetime.utcnow()
        task.execution_time = (task.end_time - task.start_time).total_seconds()
        db.session.commit()
        
        # Add processed dataset to DVC
        dvc_handler = DVCHandler(os.getcwd())
        commit_message = f"Preprocessed dataset: {dataset.name}"
        dvc_success = dvc_handler.add_and_commit_dataset(output_path, commit_message)
        
        # Create a new dataset entry for the processed dataset
        processed_dataset = NewsDataset(
            name=f"{dataset.name} (Processed)",
            path=output_path,
            file_type=os.path.splitext(output_filename)[1].lstrip('.'),
            created_at=datetime.utcnow(),
            size_bytes=os.path.getsize(output_path),
            description=f"Preprocessed version of {dataset.name}",
            category=dataset.category,
            record_count=stats.get('record_count', 0)
        )
        db.session.add(processed_dataset)
        db.session.commit()
        
        # Create dataset version
        versions = dvc_handler.get_dataset_versions(output_path)
        commit_hash = versions[0]['commit_hash'] if versions else "initial"
        
        version = DatasetVersion(
            dataset_id=processed_dataset.id,
            commit_hash=commit_hash,
            message=commit_message,
            author=data.get('author', 'system'),
            date_created=datetime.utcnow(),
            is_current=True
        )
        db.session.add(version)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Dataset preprocessed successfully",
            "task_id": task.id,
            "dataset_id": processed_dataset.id,
            "version_id": version.id,
            "stats": stats
        })
        
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {str(e)}")
        
        task.status = 'failed'
        task.end_time = datetime.utcnow()
        task.error_message = str(e)
        db.session.commit()
        
        return jsonify({
            "success": False,
            "message": f"Preprocessing failed: {str(e)}",
            "task_id": task.id
        }), 500

@preprocessing_bp.route('/extract-features', methods=['POST'])
def extract_features():
    """Extract features from a preprocessed news dataset."""
    data = request.json
    
    if not data or 'dataset_id' not in data:
        return jsonify({"success": False, "message": "Missing dataset ID"}), 400
        
    # Get dataset
    dataset_id = data['dataset_id']
    dataset = db.session.query(NewsDataset).get(dataset_id)
    
    if not dataset:
        return jsonify({"success": False, "message": "Dataset not found"}), 404
        
    if not dataset.path or not os.path.exists(dataset.path):
        return jsonify({"success": False, "message": "Dataset file not found"}), 404
        
    # Get current version
    current_version = None
    for version in dataset.versions:
        if version.is_current:
            current_version = version
            break
            
    if not current_version:
        return jsonify({"success": False, "message": "No current version found for dataset"}), 404
    
    # Set up feature extraction parameters
    config = current_app.config['CONFIG']['features']
    params = {
        'use_tfidf': data.get('use_tfidf', config['use_tfidf']),
        'use_word_embeddings': data.get('use_word_embeddings', config['use_word_embeddings']),
        'embedding_dim': data.get('embedding_dim', config['embedding_dim']),
        'max_features': data.get('max_features', current_app.config['CONFIG']['preprocessing']['max_features']),
        'max_sequence_length': data.get('max_sequence_length', config['max_sequence_length'])
    }
    
    # Create processing task
    task = ProcessingTask(
        dataset_version_id=current_version.id,
        task_type='feature_extraction',
        parameters=json.dumps(params),
        input_path=dataset.path,
        status='running',
        start_time=datetime.utcnow()
    )
    db.session.add(task)
    db.session.commit()
    
    try:
        # Determine output path
        output_dir = os.path.join(
            current_app.config['CONFIG']['data']['processed_dir'],
            secure_filename(dataset.name.lower().replace(' ', '_'))
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(dataset.path))[0]
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Different file types based on extraction method
        if params['use_tfidf']:
            output_filename = f"{base_name}_tfidf_{timestamp}.npz"
        elif params['use_word_embeddings']:
            output_filename = f"{base_name}_embeddings_{timestamp}.npz"
        else:
            output_filename = f"{base_name}_features_{timestamp}.npz"
            
        output_path = os.path.join(output_dir, output_filename)
        
        # Initialize feature extractor
        extractor = FeatureExtractor(params)
        
        # Extract features
        success, stats = extractor.extract_and_save(
            dataset.path,
            output_path,
            text_column=data.get('text_column', 'text'),
            label_column=data.get('label_column', 'label')
        )
        
        if not success:
            task.status = 'failed'
            task.end_time = datetime.utcnow()
            task.error_message = "Feature extraction failed"
            db.session.commit()
            
            return jsonify({
                "success": False,
                "message": "Feature extraction failed",
                "task_id": task.id
            }), 500
            
        # Update task with output path and completion
        task.output_path = output_path
        task.status = 'completed'
        task.end_time = datetime.utcnow()
        task.execution_time = (task.end_time - task.start_time).total_seconds()
        db.session.commit()
        
        # Add feature dataset to DVC
        dvc_handler = DVCHandler(os.getcwd())
        commit_message = f"Extracted features from: {dataset.name}"
        dvc_success = dvc_handler.add_and_commit_dataset(output_path, commit_message)
        
        # Create a new dataset entry for the feature dataset
        feature_dataset = NewsDataset(
            name=f"{dataset.name} (Features)",
            path=output_path,
            file_type=os.path.splitext(output_filename)[1].lstrip('.'),
            created_at=datetime.utcnow(),
            size_bytes=os.path.getsize(output_path),
            description=f"Extracted features from {dataset.name}",
            category=dataset.category,
            record_count=stats.get('record_count', 0)
        )
        db.session.add(feature_dataset)
        db.session.commit()
        
        # Create dataset version
        versions = dvc_handler.get_dataset_versions(output_path)
        commit_hash = versions[0]['commit_hash'] if versions else "initial"
        
        version = DatasetVersion(
            dataset_id=feature_dataset.id,
            commit_hash=commit_hash,
            message=commit_message,
            author=data.get('author', 'system'),
            date_created=datetime.utcnow(),
            is_current=True
        )
        db.session.add(version)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Features extracted successfully",
            "task_id": task.id,
            "dataset_id": feature_dataset.id,
            "version_id": version.id,
            "stats": stats
        })
        
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        
        task.status = 'failed'
        task.end_time = datetime.utcnow()
        task.error_message = str(e)
        db.session.commit()
        
        return jsonify({
            "success": False,
            "message": f"Feature extraction failed: {str(e)}",
            "task_id": task.id
        }), 500