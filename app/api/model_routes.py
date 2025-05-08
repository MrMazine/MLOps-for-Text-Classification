"""
Model API routes for fake news classification system.
"""
import os
import json
import logging
import pickle
import numpy as np
from datetime import datetime
from flask import request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.utils.database import db
from app.models.database import NewsDataset, DatasetVersion, ProcessingTask, Model
from app.api import model_bp

logger = logging.getLogger(__name__)

@model_bp.route('/train', methods=['POST'])
def train_model():
    """Train a fake news classification model."""
    data = request.json
    
    if not data or 'dataset_id' not in data or 'model_type' not in data:
        return jsonify({"success": False, "message": "Missing required parameters"}), 400
        
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
    
    # Get model parameters
    model_type = data['model_type']
    model_params = data.get('parameters', {})
    
    # Create processing task
    task = ProcessingTask(
        dataset_version_id=current_version.id,
        task_type='model_training',
        parameters=json.dumps({
            'model_type': model_type,
            'parameters': model_params
        }),
        input_path=dataset.path,
        status='running',
        start_time=datetime.utcnow()
    )
    db.session.add(task)
    db.session.commit()
    
    try:
        # Import appropriate model class based on model type
        model_class = None
        
        if model_type == 'svm':
            from sklearn.svm import SVC
            model_class = SVC
        elif model_type == 'naive_bayes':
            from sklearn.naive_bayes import MultinomialNB
            model_class = MultinomialNB
        elif model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            model_class = LogisticRegression
        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model_class = RandomForestClassifier
        elif model_type in ['lstm', 'transformer']:
            # These would require more complex implementations with deep learning
            return jsonify({
                "success": False,
                "message": f"Deep learning model type '{model_type}' is not implemented in this route",
                "task_id": task.id
            }), 501
        else:
            return jsonify({
                "success": False,
                "message": f"Unknown model type: {model_type}",
                "task_id": task.id
            }), 400
            
        # Determine if this is a feature file or a raw dataset
        file_ext = os.path.splitext(dataset.path)[1].lower()
        
        if file_ext == '.npz':
            # Load features directly
            data = np.load(dataset.path)
            X = data['features']
            if 'labels' in data:
                y = data['labels']
            else:
                raise ValueError("No labels found in feature file")
        else:
            # For raw datasets, need to extract features first
            return jsonify({
                "success": False,
                "message": "Raw datasets need to be processed with feature extraction first",
                "task_id": task.id
            }), 400
            
        # Create and train model
        clf = model_class(**model_params)
        clf.fit(X, y)
        
        # Save model
        models_dir = os.path.join('models', secure_filename(model_type))
        os.makedirs(models_dir, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{model_type}_{timestamp}.pkl"
        model_path = os.path.join(models_dir, model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)
            
        # Save metadata separately
        metadata = {
            'model_type': model_type,
            'parameters': model_params,
            'feature_type': str(data.get('feature_type', 'unknown')),
            'training_dataset': dataset.name,
            'training_dataset_id': dataset.id,
            'training_date': datetime.utcnow().isoformat(),
            'classes': data['label_classes'].tolist() if 'label_classes' in data else None
        }
        
        metadata_path = f"{model_path}.meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Update task with output path and completion
        task.output_path = model_path
        task.status = 'completed'
        task.end_time = datetime.utcnow()
        task.execution_time = (task.end_time - task.start_time).total_seconds()
        db.session.commit()
        
        # Create model entry in database
        model = Model(
            name=data.get('name', f"{model_type} model"),
            version=timestamp,
            algorithm=model_type,
            path=model_path,
            training_dataset_id=current_version.id,
            parameters=json.dumps(model_params),
            metrics=json.dumps({}),  # Will be updated in evaluation
            created_at=datetime.utcnow()
        )
        db.session.add(model)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Model trained and saved successfully",
            "task_id": task.id,
            "model_id": model.id,
            "model_path": model_path
        })
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        
        task.status = 'failed'
        task.end_time = datetime.utcnow()
        task.error_message = str(e)
        db.session.commit()
        
        return jsonify({
            "success": False,
            "message": f"Model training failed: {str(e)}",
            "task_id": task.id
        }), 500

@model_bp.route('/evaluate', methods=['POST'])
def evaluate_model():
    """Evaluate a trained fake news classification model."""
    data = request.json
    
    if not data or 'model_id' not in data or 'dataset_id' not in data:
        return jsonify({"success": False, "message": "Missing required parameters"}), 400
        
    # Get model
    model_id = data['model_id']
    model_record = db.session.query(Model).get(model_id)
    
    if not model_record:
        return jsonify({"success": False, "message": "Model not found"}), 404
        
    if not model_record.path or not os.path.exists(model_record.path):
        return jsonify({"success": False, "message": "Model file not found"}), 404
        
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
    
    # Create processing task
    task = ProcessingTask(
        dataset_version_id=current_version.id,
        task_type='model_evaluation',
        parameters=json.dumps({
            'model_id': model_id,
            'model_type': model_record.algorithm
        }),
        input_path=dataset.path,
        status='running',
        start_time=datetime.utcnow()
    )
    db.session.add(task)
    db.session.commit()
    
    try:
        # Load model
        with open(model_record.path, 'rb') as f:
            model = pickle.load(f)
            
        # Determine if this is a feature file or a raw dataset
        file_ext = os.path.splitext(dataset.path)[1].lower()
        
        if file_ext == '.npz':
            # Load features directly
            data = np.load(dataset.path)
            X = data['features']
            if 'labels' in data:
                y = data['labels']
            else:
                raise ValueError("No labels found in feature file")
        else:
            # For raw datasets, need to extract features first
            return jsonify({
                "success": False,
                "message": "Raw datasets need to be processed with feature extraction first",
                "task_id": task.id
            }), 400
            
        # Evaluate model
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        y_pred = model.predict(X)
        
        # Calculate metrics
        accuracy = float(accuracy_score(y, y_pred))
        precision = float(precision_score(y, y_pred, average='weighted', zero_division=0))
        recall = float(recall_score(y, y_pred, average='weighted', zero_division=0))
        f1 = float(f1_score(y, y_pred, average='weighted', zero_division=0))
        confusion = confusion_matrix(y, y_pred).tolist()
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion
        }
        
        # Update task with completion
        task.status = 'completed'
        task.end_time = datetime.utcnow()
        task.execution_time = (task.end_time - task.start_time).total_seconds()
        task.accuracy = accuracy
        task.precision = precision
        task.recall = recall
        task.f1_score = f1
        db.session.commit()
        
        # Update model metrics
        model_record.metrics = json.dumps(metrics)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Model evaluated successfully",
            "task_id": task.id,
            "model_id": model_id,
            "metrics": metrics
        })
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        
        task.status = 'failed'
        task.end_time = datetime.utcnow()
        task.error_message = str(e)
        db.session.commit()
        
        return jsonify({
            "success": False,
            "message": f"Model evaluation failed: {str(e)}",
            "task_id": task.id
        }), 500

@model_bp.route('/predict', methods=['POST'])
def predict():
    """Make predictions with a trained fake news classification model."""
    data = request.json
    
    if not data or 'model_id' not in data or 'text' not in data:
        return jsonify({"success": False, "message": "Missing required parameters"}), 400
        
    # Get model
    model_id = data['model_id']
    model_record = db.session.query(Model).get(model_id)
    
    if not model_record:
        return jsonify({"success": False, "message": "Model not found"}), 404
        
    if not model_record.path or not os.path.exists(model_record.path):
        return jsonify({"success": False, "message": "Model file not found"}), 404
        
    try:
        # Load model
        with open(model_record.path, 'rb') as f:
            model = pickle.load(f)
            
        # Load model metadata
        metadata_path = f"{model_record.path}.meta.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {'model_type': model_record.algorithm}
            
        # Process text based on model type
        text = data['text']
        
        if isinstance(text, list):
            texts = text
        else:
            texts = [text]
            
        # Preprocess text
        from app.utils.text_preprocessing import TextPreprocessor
        preprocessor = TextPreprocessor()
        preprocessed_texts = preprocessor.preprocess_batch(texts)
        
        # Extract features based on metadata
        feature_type = metadata.get('feature_type', 'tfidf')
        
        if feature_type == 'tfidf':
            from sklearn.feature_extraction.text import TfidfVectorizer
            # Need to load the vectorizer used during training
            # For now, create a new one (not ideal, but a placeholder)
            vectorizer = TfidfVectorizer(max_features=5000)
            vectorizer.fit(preprocessed_texts)
            X = vectorizer.transform(preprocessed_texts)
        else:
            # Simple word count vectorizer as fallback
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(max_features=5000)
            vectorizer.fit(preprocessed_texts)
            X = vectorizer.transform(preprocessed_texts)
            
        # Make predictions
        y_pred = model.predict(X)
        
        # Determine class labels
        classes = metadata.get('classes', ['fake', 'real'])
        if len(classes) == 2:
            class_names = ['fake', 'real']
        else:
            class_names = [f"class_{i}" for i in range(len(classes))]
            
        # Get prediction probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X)
            probabilities = [proba.tolist() for proba in probas]
            
        # Format results
        results = []
        for i, pred in enumerate(y_pred):
            result = {
                'text': texts[i],
                'prediction': int(pred),
                'label': class_names[pred] if pred < len(class_names) else f"class_{pred}"
            }
            
            if probabilities:
                result['probabilities'] = probabilities[i]
                
            results.append(result)
            
        return jsonify({
            "success": True,
            "predictions": results,
            "model_info": {
                "id": model_record.id,
                "name": model_record.name,
                "algorithm": model_record.algorithm
            }
        })
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        
        return jsonify({
            "success": False,
            "message": f"Prediction failed: {str(e)}"
        }), 500

@model_bp.route('/list', methods=['GET'])
def list_models():
    """List all trained fake news classification models."""
    models = db.session.query(Model).all()
    
    model_list = []
    for model in models:
        # Parse metrics if available
        metrics = {}
        if model.metrics:
            try:
                metrics = json.loads(model.metrics)
            except:
                pass
                
        model_list.append({
            "id": model.id,
            "name": model.name,
            "version": model.version,
            "algorithm": model.algorithm,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "accuracy": metrics.get('accuracy', 0),
            "f1_score": metrics.get('f1_score', 0)
        })
        
    return jsonify({
        "success": True,
        "count": len(model_list),
        "models": model_list
    })