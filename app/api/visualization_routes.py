"""
Visualization API routes for fake news classification system.
"""
import os
import json
import logging
import numpy as np
from datetime import datetime
from flask import request, jsonify, current_app
from app.utils.database import db
from app.models.database import NewsDataset, DatasetStats, Model
from app.api import visualization_bp

logger = logging.getLogger(__name__)

@visualization_bp.route('/dataset-stats/<int:dataset_id>', methods=['GET'])
def get_dataset_stats(dataset_id):
    """Get visualization data for a news dataset."""
    dataset = db.session.query(NewsDataset).get(dataset_id)
    if not dataset:
        return jsonify({"success": False, "message": "Dataset not found"}), 404
        
    # Get the most recent stats if available
    stats = db.session.query(DatasetStats).filter(
        DatasetStats.dataset_id == dataset_id
    ).order_by(DatasetStats.created_at.desc()).first()
    
    if not stats:
        return jsonify({
            "success": False,
            "message": "No statistics available for this dataset"
        }), 404
        
    # Basic dataset info
    result = {
        "dataset": {
            "id": dataset.id,
            "name": dataset.name,
            "record_count": stats.record_count or 0,
            "category": dataset.category
        },
        "text_length": {
            "average": stats.avg_article_length or 0,
            "minimum": stats.min_article_length or 0,
            "maximum": stats.max_article_length or 0
        }
    }
    
    # Add sentiment distribution if available
    if stats.sentiment_distribution:
        try:
            sentiment_data = json.loads(stats.sentiment_distribution)
            result["sentiment"] = sentiment_data
        except:
            pass
            
    # Add topic distribution if available
    if stats.topic_distribution:
        try:
            topic_data = json.loads(stats.topic_distribution)
            result["topics"] = topic_data
        except:
            pass
            
    # Add entity data if available
    if stats.top_entities:
        try:
            entity_data = json.loads(stats.top_entities)
            result["entities"] = entity_data
        except:
            pass
            
    return jsonify({
        "success": True,
        "data": result
    })

@visualization_bp.route('/model-comparison', methods=['GET'])
def compare_models():
    """Compare performance of different fake news classification models."""
    models = db.session.query(Model).all()
    
    if not models:
        return jsonify({
            "success": False,
            "message": "No models available for comparison"
        }), 404
        
    # Group models by algorithm
    algorithms = {}
    
    for model in models:
        algorithm = model.algorithm
        
        if algorithm not in algorithms:
            algorithms[algorithm] = []
            
        # Parse metrics
        metrics = {}
        if model.metrics:
            try:
                metrics = json.loads(model.metrics)
            except:
                pass
                
        algorithms[algorithm].append({
            "id": model.id,
            "name": model.name,
            "version": model.version,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "accuracy": metrics.get('accuracy', 0),
            "precision": metrics.get('precision', 0),
            "recall": metrics.get('recall', 0),
            "f1_score": metrics.get('f1_score', 0)
        })
        
    # Calculate average metrics by algorithm
    comparison = []
    
    for algorithm, models in algorithms.items():
        avg_accuracy = sum(m['accuracy'] for m in models) / len(models)
        avg_precision = sum(m['precision'] for m in models) / len(models)
        avg_recall = sum(m['recall'] for m in models) / len(models)
        avg_f1 = sum(m['f1_score'] for m in models) / len(models)
        
        comparison.append({
            "algorithm": algorithm,
            "model_count": len(models),
            "metrics": {
                "avg_accuracy": avg_accuracy,
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "avg_f1_score": avg_f1
            },
            "models": models
        })
        
    return jsonify({
        "success": True,
        "comparison": comparison
    })

@visualization_bp.route('/confusion-matrix/<int:model_id>', methods=['GET'])
def get_confusion_matrix(model_id):
    """Get confusion matrix for a specific model."""
    model = db.session.query(Model).get(model_id)
    
    if not model:
        return jsonify({
            "success": False,
            "message": "Model not found"
        }), 404
        
    # Parse metrics
    if not model.metrics:
        return jsonify({
            "success": False,
            "message": "No metrics available for this model"
        }), 404
        
    try:
        metrics = json.loads(model.metrics)
    except:
        return jsonify({
            "success": False,
            "message": "Failed to parse model metrics"
        }), 500
        
    if 'confusion_matrix' not in metrics:
        return jsonify({
            "success": False,
            "message": "No confusion matrix available for this model"
        }), 404
        
    # Get confusion matrix
    confusion_matrix = metrics['confusion_matrix']
    
    # Get class labels - for fake news typically binary (fake/real)
    classes = metrics.get('classes', ['fake', 'real'])
    
    # If classes not in metrics, try to load from model metadata
    if not classes and model.path:
        metadata_path = f"{model.path}.meta.json"
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    classes = metadata.get('classes', ['fake', 'real'])
            except:
                pass
                
    return jsonify({
        "success": True,
        "confusion_matrix": confusion_matrix,
        "classes": classes,
        "model": {
            "id": model.id,
            "name": model.name,
            "algorithm": model.algorithm
        }
    })

@visualization_bp.route('/feature-importance/<int:model_id>', methods=['GET'])
def get_feature_importance(model_id):
    """Get feature importance for a specific model."""
    model = db.session.query(Model).get(model_id)
    
    if not model:
        return jsonify({
            "success": False,
            "message": "Model not found"
        }), 404
        
    # Feature importance is only available for certain algorithms
    if model.algorithm not in ['random_forest', 'logistic_regression', 'svm']:
        return jsonify({
            "success": False,
            "message": f"Feature importance not available for {model.algorithm} models"
        }), 400
        
    if not model.path or not os.path.exists(model.path):
        return jsonify({
            "success": False,
            "message": "Model file not found"
        }), 404
        
    try:
        # Load model
        import pickle
        with open(model.path, 'rb') as f:
            clf = pickle.load(f)
            
        # Get feature names from metadata
        metadata_path = f"{model.path}.meta.json"
        feature_names = None
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    feature_names = metadata.get('feature_names', None)
            except:
                pass
                
        # Get feature importance based on algorithm
        importance = None
        
        if model.algorithm == 'random_forest' and hasattr(clf, 'feature_importances_'):
            importance = clf.feature_importances_
        elif model.algorithm == 'logistic_regression' and hasattr(clf, 'coef_'):
            importance = np.abs(clf.coef_[0]) if clf.coef_.ndim > 1 else np.abs(clf.coef_)
        elif model.algorithm == 'svm' and hasattr(clf, 'coef_'):
            importance = np.abs(clf.coef_[0]) if clf.coef_.ndim > 1 else np.abs(clf.coef_)
            
        if importance is None:
            return jsonify({
                "success": False,
                "message": "Feature importance not available for this model"
            }), 404
            
        # If we have feature names and importance values
        if feature_names and len(feature_names) == len(importance):
            # Sort by importance
            indices = np.argsort(importance)[::-1]
            top_indices = indices[:50]  # Limit to top 50 features
            
            features = []
            for i in top_indices:
                features.append({
                    "feature": feature_names[i],
                    "importance": float(importance[i])
                })
                
            return jsonify({
                "success": True,
                "features": features,
                "model": {
                    "id": model.id,
                    "name": model.name,
                    "algorithm": model.algorithm
                }
            })
        else:
            # Return just the raw values
            top_indices = np.argsort(importance)[::-1][:50]
            
            features = []
            for i, idx in enumerate(top_indices):
                features.append({
                    "feature": f"feature_{idx}",
                    "importance": float(importance[idx])
                })
                
            return jsonify({
                "success": True,
                "features": features,
                "model": {
                    "id": model.id,
                    "name": model.name,
                    "algorithm": model.algorithm
                }
            })
            
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        
        return jsonify({
            "success": False,
            "message": f"Failed to get feature importance: {str(e)}"
        }), 500