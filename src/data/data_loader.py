import os
import logging
import pandas as pd
import csv
import json
from typing import Iterator, Dict, List, Union, Optional, Generator, Any, Tuple
import time
from functools import wraps

from src.utils.decorators import timing_decorator, memory_usage_decorator
from src.config import Config

logger = logging.getLogger(__name__)

class DataLoader:
    """
    A memory-efficient data loader for text classification datasets.
    Uses generators and iterators to load and process data in a memory-friendly way.
    """
    
    def __init__(self, data_path: str, config: Optional[Config] = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the dataset file or directory
            config: Configuration object, if None, a default one will be created
        """
        self.data_path = data_path
        self.config = config or Config()
        self.file_extension = self._get_file_extension(data_path)
        self.is_directory = os.path.isdir(data_path)
        self.batch_size = self.config.batch_size
        
        # Validate the data path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        logger.info(f"DataLoader initialized with path: {data_path}")
    
    def _get_file_extension(self, path: str) -> str:
        """Get the file extension of the dataset."""
        if os.path.isdir(path):
            # Get the extension of the first file in the directory
            for filename in os.listdir(path):
                if os.path.isfile(os.path.join(path, filename)):
                    _, ext = os.path.splitext(filename)
                    return ext.lower()
            return ""
        else:
            _, ext = os.path.splitext(path)
            return ext.lower()
    
    def _file_iterator(self) -> Iterator[str]:
        """Iterator that yields file paths from the data path."""
        if self.is_directory:
            for dirpath, _, filenames in os.walk(self.data_path):
                for filename in filenames:
                    yield os.path.join(dirpath, filename)
        else:
            yield self.data_path
    
    @timing_decorator
    def load_batch_generator(self, batch_size: Optional[int] = None) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Generator that yields batches of data records.
        
        Args:
            batch_size: Size of each batch, defaults to config batch_size
            
        Yields:
            List of data records (dictionaries)
        """
        batch_size = batch_size or self.batch_size
        batch = []
        
        for file_path in self._file_iterator():
            for record in self._parse_file(file_path):
                batch.append(record)
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        
        # Yield any remaining records
        if batch:
            yield batch
    
    def _parse_file(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Parse a file and yield data records.
        
        Args:
            file_path: Path to the file to parse
            
        Yields:
            Dictionary containing the parsed data
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        try:
            if ext == '.csv':
                yield from self._parse_csv(file_path)
            elif ext == '.json':
                yield from self._parse_json(file_path)
            elif ext == '.txt':
                yield from self._parse_txt(file_path)
            else:
                logger.warning(f"Unsupported file extension: {ext} for file {file_path}")
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
    
    def _parse_csv(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        """Parse a CSV file and yield records."""
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield dict(row)
    
    def _parse_json(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        """Parse a JSON file and yield records."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, list):
                for item in data:
                    yield item
            elif isinstance(data, dict):
                # Check if it's a record-per-key format
                for key, value in data.items():
                    if isinstance(value, dict):
                        record = value.copy()
                        record['id'] = key
                        yield record
                    else:
                        # Single record JSON
                        yield data
                        break
    
    def _parse_txt(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        """Parse a text file and yield records (one record per line)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:  # Skip empty lines
                    yield {'id': i, 'text': line}
    
    @memory_usage_decorator
    def load_pandas(self) -> pd.DataFrame:
        """
        Load the entire dataset into a pandas DataFrame.
        Warning: This loads all data into memory.
        
        Returns:
            Pandas DataFrame containing the data
        """
        all_data = []
        for batch in self.load_batch_generator():
            all_data.extend(batch)
        
        return pd.DataFrame(all_data)
    
    def load_from_dvc_version(self, repo_path: str, version: str) -> 'DataLoader':
        """
        Load a specific versioned dataset from DVC.
        
        Args:
            repo_path: Path to the DVC repository
            version: Git commit hash or tag
            
        Returns:
            A new DataLoader instance pointing to the versioned data
        """
        from dvc_helpers.dvc_setup import DVCHandler
        
        dvc_handler = DVCHandler(repo_path)
        versioned_path = dvc_handler.checkout_to_temp(self.data_path, version)
        
        return DataLoader(versioned_path, self.config)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics about the dataset.
        
        Returns:
            Dictionary of statistics
        """
        file_count = 0
        record_count = 0
        fields = set()
        text_lengths = []
        sample_records = []
        
        # Process in batches to calculate statistics
        for batch in self.load_batch_generator(batch_size=1000):
            record_count += len(batch)
            
            # Track field names
            for record in batch:
                fields.update(record.keys())
                
                # Track text length if a 'text' field exists
                if 'text' in record:
                    text_lengths.append(len(str(record['text'])))
                
            # Keep a small sample of records
            sample_records.extend(batch[:5])
            if len(sample_records) > 10:
                sample_records = sample_records[:10]
            
            file_count += 1
        
        # Calculate additional statistics
        avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        max_text_length = max(text_lengths) if text_lengths else 0
        min_text_length = min(text_lengths) if text_lengths else 0
        
        return {
            'file_count': file_count,
            'record_count': record_count,
            'fields': list(fields),
            'avg_text_length': avg_text_length,
            'max_text_length': max_text_length,
            'min_text_length': min_text_length,
            'sample_records': sample_records
        }
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Make the DataLoader iterable so it can be used in loops.
        
        Returns:
            Iterator of data records
        """
        for batch in self.load_batch_generator(batch_size=1):
            yield from batch
