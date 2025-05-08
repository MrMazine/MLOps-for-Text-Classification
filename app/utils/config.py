"""
Configuration utilities for fake news classification system.
"""
import os
import yaml
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration class for the fake news classification project."""
    
    DEFAULT_CONFIG = {
        "project": {
            "name": "fake_news_classification",
            "description": "MLOps system for fake news detection"
        },
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "external_dir": "data/external",
            "min_text_length": 50
        },
        "preprocessing": {
            "remove_stopwords": True,
            "lowercase": True,
            "remove_punctuation": True,
            "remove_numbers": False,
            "lemmatize": True,
            "stemming": False,
            "max_features": 5000
        },
        "features": {
            "use_tfidf": True,
            "use_word_embeddings": True,
            "embedding_dim": 100,
            "max_sequence_length": 500
        },
        "model": {
            "type": "transformer",  # Options: svm, naive_bayes, logistic_regression, lstm, transformer
            "batch_size": 64,
            "epochs": 10,
            "validation_split": 0.2,
            "learning_rate": 1e-4
        },
        "dvc": {
            "remote_name": "origin",
            "remote_url": "gdrive://1xR1vVXuT_rJ7XfcxJnUJ8g07Pf5BaX3z"  # Replace with your GDrive folder ID
        }
    }

    def __init__(self, config_path="config.yaml"):
        """
        Initialize configuration with default values or from config file.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load configuration from file if it exists
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        # Update config with values from file
                        self._update_nested_dict(self.config, file_config)
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        else:
            # Save default configuration
            self.save_config()
            
    def _update_nested_dict(self, d, u):
        """
        Update nested dictionary recursively.
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
        """
        import collections.abc
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = self._update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
            
    def save_config(self, config_path=None):
        """
        Save the current configuration to a YAML file.
        
        Args:
            config_path: Path where to save the configuration
        """
        if config_path is None:
            config_path = self.config_path
            
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {str(e)}")
            
    def __getitem__(self, key):
        """Allow dictionary-like access to configuration."""
        return self.config[key]
        
    def get(self, key, default=None):
        """Get configuration value with default."""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Set configuration value."""
        self.config[key] = value