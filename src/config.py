import os
import yaml
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for the DVC text classification project."""
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize configuration with default values or from config file.
        
        Args:
            config_path: Path to the configuration file
        """
        # Default values
        self.raw_data_path = os.path.abspath("data/raw")
        self.processed_data_path = os.path.abspath("data/processed")
        self.dvc_repo_path = os.path.abspath(".")
        self.dvc_remote_url = os.environ.get("DVC_REMOTE_URL", None)
        self.dvc_remote_name = "origin"
        
        # Text preprocessing options
        self.text_preprocessing = {
            "lowercase": True,
            "remove_punctuation": True,
            "remove_numbers": False,
            "remove_stopwords": True,
            "language": "english",
            "stemming": False,
            "lemmatization": True
        }
        
        # Memory optimization settings
        self.batch_size = 1000
        self.use_generators = True
        
        # Load from config file if exists
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Update configuration with values from file
                if config_data:
                    for key, value in config_data.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
                
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
                logger.info("Using default configuration")
        else:
            logger.info(f"Configuration file {config_path} not found. Using default configuration.")
            
            # Create default config file
            self.save_config(config_path)
    
    def save_config(self, config_path="config.yaml"):
        """
        Save the current configuration to a YAML file.
        
        Args:
            config_path: Path where to save the configuration
        """
        try:
            config_dict = {key: value for key, value in self.__dict__.items() 
                          if not key.startswith('_')}
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
            return False
