"""
DVC handler for fake news classification system.
"""
import os
import tempfile
import logging
import shutil
import json
from typing import List, Dict, Any, Tuple, Optional
import subprocess
import datetime

logger = logging.getLogger(__name__)

class DVCHandler:
    """Handler for DVC operations to manage news dataset versioning."""

    def __init__(self, repo_path: str):
        """
        Initialize the DVC handler.
        
        Args:
            repo_path: Path to the DVC repository
        """
        self.repo_path = repo_path
        self.temp_dir = os.path.join(tempfile.gettempdir(), "dvc_news_datasets")
        
        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info(f"DVC handler initialized with repo path: {repo_path}")

    def _run_command(self, command: List[str], cwd: Optional[str] = None) -> Tuple[bool, str, str]:
        """
        Run a shell command and return the result.
        
        Args:
            command: Command to run as a list of strings
            cwd: Working directory for the command
            
        Returns:
            Tuple containing (success, stdout, stderr)
        """
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd or self.repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            success = process.returncode == 0
            return success, stdout, stderr
        except Exception as e:
            logger.error(f"Command failed: {' '.join(command)}\nError: {str(e)}")
            return False, "", str(e)

    def is_dvc_initialized(self) -> bool:
        """
        Check if DVC is initialized in the repository.
        
        Returns:
            True if DVC is initialized, False otherwise
        """
        dvc_dir = os.path.join(self.repo_path, ".dvc")
        return os.path.isdir(dvc_dir)

    def initialize_dvc(self) -> bool:
        """
        Initialize DVC in the repository.
        
        Returns:
            True on success, False on failure
        """
        if self.is_dvc_initialized():
            logger.info("DVC already initialized")
            return True

        success, _, stderr = self._run_command(["dvc", "init"])
        if not success:
            logger.error(f"Failed to initialize DVC: {stderr}")
            return False

        # Configure DVC to use the temp directory for cache to avoid permission issues
        temp_cache_dir = os.path.join(self.temp_dir, "cache")
        os.makedirs(temp_cache_dir, exist_ok=True)
        
        success, _, stderr = self._run_command(["dvc", "config", "cache.dir", temp_cache_dir])
        if not success:
            logger.error(f"Failed to configure DVC cache: {stderr}")
            return False

        logger.info("DVC initialized successfully")
        return True

    def add_remote(self, name: str, url: str, default: bool = True) -> bool:
        """
        Add a remote storage to DVC.
        
        Args:
            name: Name of the remote
            url: URL of the remote
            default: Set as default remote
            
        Returns:
            True on success, False on failure
        """
        success, _, stderr = self._run_command(["dvc", "remote", "add", name, url])
        if not success:
            logger.error(f"Failed to add remote: {stderr}")
            return False

        if default:
            success, _, stderr = self._run_command(["dvc", "remote", "default", name])
            if not success:
                logger.error(f"Failed to set default remote: {stderr}")
                return False

        logger.info(f"Remote {name} added successfully")
        return True

    def add_and_commit_dataset(self, dataset_path: str, commit_message: str) -> bool:
        """
        Add a dataset to DVC and commit the changes.
        For Replit environments, we'll create a simulated tracking without using DVC
        because of permission issues with DVC's cache directory.
        
        Args:
            dataset_path: Path to the dataset
            commit_message: Git commit message
            
        Returns:
            True on success, False on failure
        """
        # Create a DVC file manually to simulate DVC tracking
        try:
            relative_path = os.path.relpath(dataset_path, start=self.repo_path)
            dvc_file_path = f"{relative_path}.dvc"
            
            dvc_content = {
                "outs": [
                    {
                        "path": relative_path,
                        "md5": datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
                        "size": os.path.getsize(dataset_path)
                    }
                ],
                "version": "0.0.1"
            }
            
            with open(os.path.join(self.repo_path, dvc_file_path), "w") as f:
                json.dump(dvc_content, f, indent=2)
                
            logger.info(f"Created simulated DVC tracking for {relative_path}")
            
            # Try to add the DVC file to git
            success, _, stderr = self._run_command(["git", "add", dvc_file_path])
            if not success:
                logger.error(f"Failed to add DVC file to git: {stderr}")
                return False
                
            success, _, stderr = self._run_command(["git", "commit", "-m", commit_message])
            if not success:
                logger.error(f"Failed to commit DVC file: {stderr}")
                return False
                
            return True
        except Exception as e:
            logger.warning(f"Could not add dataset to DVC tracking: {str(e)}")
            return False

    def get_dataset_versions(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a specific news dataset.
        
        Args:
            dataset_path: Path to the dataset
            
        Returns:
            List of dictionaries with version information
        """
        relative_path = os.path.relpath(dataset_path, start=self.repo_path)
        dvc_file_path = f"{relative_path}.dvc"
        
        success, stdout, _ = self._run_command(["git", "log", "--pretty=format:%H|%an|%ad|%s", "--date=iso", dvc_file_path])
        if not success or not stdout:
            return []
            
        versions = []
        for line in stdout.splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                commit_hash, author, date, message = parts
                versions.append({
                    "commit_hash": commit_hash,
                    "author": author,
                    "date": date,
                    "message": message
                })
        
        return versions

    def checkout_version(self, dataset_path: str, version: str) -> bool:
        """
        Checkout a specific version of a news dataset.
        
        Args:
            dataset_path: Path to the dataset
            version: Git commit hash or tag
            
        Returns:
            True on success, False on failure
        """
        relative_path = os.path.relpath(dataset_path, start=self.repo_path)
        dvc_file_path = f"{relative_path}.dvc"
        
        # Git checkout the specific version of the DVC file
        success, _, stderr = self._run_command(["git", "checkout", version, "--", dvc_file_path])
        if not success:
            logger.error(f"Failed to checkout dataset version: {stderr}")
            return False
            
        # Simulate DVC checkout by copying from backup if available
        backup_path = os.path.join(self.temp_dir, "backups", relative_path, version)
        if os.path.exists(backup_path):
            try:
                shutil.copy(backup_path, dataset_path)
                logger.info(f"Restored dataset from backup at {backup_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to restore dataset from backup: {str(e)}")
                return False
        else:
            logger.warning(f"No backup found for version {version} of {relative_path}")
            return False

    def checkout_to_temp(self, dataset_path: str, version: str) -> str:
        """
        Checkout a specific version of a news dataset to a temporary directory.
        
        Args:
            dataset_path: Path to the dataset
            version: Git commit hash or tag
            
        Returns:
            Path to the temporary directory containing the dataset
        """
        relative_path = os.path.relpath(dataset_path, start=self.repo_path)
        temp_version_dir = os.path.join(self.temp_dir, "versions", relative_path, version)
        os.makedirs(os.path.dirname(temp_version_dir), exist_ok=True)
        
        # Try to checkout from backup
        backup_path = os.path.join(self.temp_dir, "backups", relative_path, version)
        if os.path.exists(backup_path):
            try:
                shutil.copy(backup_path, temp_version_dir)
                return temp_version_dir
            except Exception as e:
                logger.error(f"Failed to copy from backup: {str(e)}")
        
        # If no backup, try to checkout the current version
        try:
            shutil.copy(dataset_path, temp_version_dir)
            return temp_version_dir
        except Exception as e:
            logger.error(f"Failed to create temporary version: {str(e)}")
            return ""

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all news datasets tracked by DVC.
        
        Returns:
            List of dictionaries with dataset information
        """
        datasets = []
        
        # Look for .dvc files in the repository
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith(".dvc"):
                    dvc_file_path = os.path.join(root, file)
                    dataset_path = dvc_file_path[:-4]  # Remove .dvc extension
                    
                    try:
                        with open(dvc_file_path, "r") as f:
                            dvc_data = json.load(f)
                            
                        relative_path = os.path.relpath(dataset_path, start=self.repo_path)
                        
                        # Get dataset size if available
                        size = 0
                        if os.path.exists(dataset_path):
                            size = os.path.getsize(dataset_path)
                            
                        datasets.append({
                            "path": relative_path,
                            "size": size,
                            "dvc_file": os.path.relpath(dvc_file_path, start=self.repo_path)
                        })
                    except Exception as e:
                        logger.error(f"Failed to parse DVC file {dvc_file_path}: {str(e)}")
        
        return datasets

    def get_dvc_info(self) -> Dict[str, Any]:
        """
        Get information about the DVC repository.
        
        Returns:
            Dictionary with DVC information
        """
        info = {"initialized": self.is_dvc_initialized()}
        
        if info["initialized"]:
            # Get DVC version
            success, stdout, _ = self._run_command(["dvc", "--version"])
            if success:
                info["version"] = stdout.strip()
                
            # Get remotes
            success, stdout, _ = self._run_command(["dvc", "remote", "list"])
            if success:
                remotes = []
                for line in stdout.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        remotes.append({"name": parts[0], "url": parts[1]})
                info["remotes"] = remotes
                
            # Get cache info
            success, stdout, _ = self._run_command(["dvc", "config", "cache.dir"])
            if success:
                info["cache_dir"] = stdout.strip()
        
        return info