import os
import logging
from typing import Dict, Any, List, Optional, Union
import tempfile
import shutil

from src.utils.decorators import timing_decorator, retry_decorator
from src.config import Config
from dvc_helpers.dvc_setup import DVCHandler

logger = logging.getLogger(__name__)

class RemoteStorageManager:
    """
    Manager for handling remote storage operations with DVC.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the remote storage manager.
        
        Args:
            config: Configuration object with DVC settings
        """
        self.config = config
        self.dvc_handler = DVCHandler(config.dvc_repo_path)
        self.remote_url = config.dvc_remote_url
        self.remote_name = config.dvc_remote_name
        logger.info(f"RemoteStorageManager initialized with remote {self.remote_name}")
    
    @timing_decorator
    def setup_remote(self) -> bool:
        """
        Set up the remote storage for DVC.
        
        Returns:
            True on success, False on failure
        """
        if not self.remote_url:
            logger.warning("No remote URL configured, skipping remote setup")
            return False
        
        # Check if DVC is initialized
        if not self.dvc_handler.is_dvc_initialized():
            logger.info("DVC not initialized, initializing...")
            if not self.dvc_handler.initialize_dvc():
                logger.error("Failed to initialize DVC")
                return False
        
        # Get DVC info to check if remote already exists
        dvc_info = self.dvc_handler.get_dvc_info()
        remote_exists = False
        
        for remote in dvc_info.get('remotes', []):
            if remote['name'] == self.remote_name:
                remote_exists = True
                if remote['url'] != self.remote_url:
                    logger.warning(f"Remote {self.remote_name} exists with different URL. "
                                  f"Current: {remote['url']}, Config: {self.remote_url}")
                break
        
        # Add remote if it doesn't exist
        if not remote_exists:
            logger.info(f"Adding remote {self.remote_name} with URL {self.remote_url}")
            return self.dvc_handler.add_remote(self.remote_name, self.remote_url, default=True)
        
        logger.info(f"Remote {self.remote_name} already configured")
        return True
    
    @timing_decorator
    @retry_decorator(max_attempts=3, delay=2.0)
    def push_to_remote(self) -> Dict[str, Any]:
        """
        Push local data to remote storage.
        
        Returns:
            Dictionary with result information
        """
        if not self.remote_url:
            return {
                'success': False,
                'message': "No remote URL configured"
            }
        
        # Setup remote if needed
        if not self.setup_remote():
            return {
                'success': False,
                'message': "Failed to setup remote storage"
            }
        
        # Push data to remote
        success, stdout, stderr = self.dvc_handler._run_command(['dvc', 'push'])
        
        if success:
            message = "Successfully pushed data to remote storage"
            logger.info(message)
            return {
                'success': True,
                'message': message
            }
        else:
            message = f"Failed to push data to remote: {stderr}"
            logger.error(message)
            return {
                'success': False,
                'message': message
            }
    
    @timing_decorator
    @retry_decorator(max_attempts=3, delay=2.0)
    def pull_from_remote(self) -> Dict[str, Any]:
        """
        Pull data from remote storage.
        
        Returns:
            Dictionary with result information
        """
        if not self.remote_url:
            return {
                'success': False,
                'message': "No remote URL configured"
            }
        
        # Setup remote if needed
        if not self.setup_remote():
            return {
                'success': False,
                'message': "Failed to setup remote storage"
            }
        
        # Pull data from remote
        success, stdout, stderr = self.dvc_handler._run_command(['dvc', 'pull'])
        
        if success:
            message = "Successfully pulled data from remote storage"
            logger.info(message)
            return {
                'success': True,
                'message': message
            }
        else:
            message = f"Failed to pull data from remote: {stderr}"
            logger.error(message)
            return {
                'success': False,
                'message': message
            }
    
    @timing_decorator
    def get_remote_status(self) -> Dict[str, Any]:
        """
        Get the status of local data compared to remote storage.
        
        Returns:
            Dictionary with status information
        """
        if not self.remote_url:
            return {
                'success': False,
                'message': "No remote URL configured",
                'status': {}
            }
        
        # Setup remote if needed
        if not self.setup_remote():
            return {
                'success': False,
                'message': "Failed to setup remote storage",
                'status': {}
            }
        
        # Get status
        success, stdout, stderr = self.dvc_handler._run_command(['dvc', 'status', '-c'])
        
        if not success:
            logger.error(f"Failed to get remote status: {stderr}")
            return {
                'success': False,
                'message': f"Failed to get remote status: {stderr}",
                'status': {}
            }
        
        # Parse status output
        status = {
            'new': [],
            'modified': [],
            'deleted': [],
            'up_to_date': [],
            'not_in_cache': []
        }
        
        current_section = None
        for line in stdout.strip().split('\n'):
            line = line.strip()
            
            if not line:
                continue
            
            if line.endswith(':'):
                section_name = line[:-1].lower()
                if 'deleted' in section_name:
                    current_section = 'deleted'
                elif 'new' in section_name:
                    current_section = 'new'
                elif 'modified' in section_name:
                    current_section = 'modified'
                elif 'not in cache' in section_name:
                    current_section = 'not_in_cache'
                elif 'unchanged' in section_name or 'up to date' in section_name:
                    current_section = 'up_to_date'
                else:
                    current_section = None
            elif current_section:
                status[current_section].append(line)
        
        return {
            'success': True,
            'message': "Successfully retrieved remote status",
            'status': status
        }
    
    @timing_decorator
    def add_to_remote(self, dataset_path: str, commit_message: str) -> Dict[str, Any]:
        """
        Add a dataset to DVC and push to remote storage.
        
        Args:
            dataset_path: Path to the dataset
            commit_message: Git commit message
            
        Returns:
            Dictionary with result information
        """
        # Add dataset to DVC
        success = self.dvc_handler.add_and_commit_dataset(dataset_path, commit_message)
        
        if not success:
            return {
                'success': False,
                'message': "Failed to add dataset to DVC"
            }
        
        # Push to remote
        return self.push_to_remote()
    
    @timing_decorator
    def get_from_remote(self, dataset_path: str, version: str) -> Dict[str, Any]:
        """
        Get a specific version of a dataset from remote storage.
        
        Args:
            dataset_path: Path to the dataset
            version: Git commit hash or tag
            
        Returns:
            Dictionary with result information
        """
        # Pull from remote first
        pull_result = self.pull_from_remote()
        
        if not pull_result['success']:
            return pull_result
        
        # Checkout the specific version
        success = self.dvc_handler.checkout_version(dataset_path, version)
        
        if success:
            return {
                'success': True,
                'message': f"Successfully retrieved version {version} of dataset {dataset_path}"
            }
        else:
            return {
                'success': False,
                'message': f"Failed to checkout version {version} of dataset {dataset_path}"
            }
