import re
import string
import logging
from typing import List, Dict, Any, Optional, Union, Callable, Generator
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import os
import json
import csv
import time
from functools import wraps

from src.utils.decorators import timing_decorator, memory_usage_decorator
from src.data.data_loader import DataLoader
from src.config import Config

logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {e}")

class TextPreprocessor:
    """
    A text preprocessor for NLP classification tasks.
    Provides memory-efficient preprocessing using generators.
    """
    
    def __init__(self, config: Optional[Union[Config, Dict[str, Any]]] = None):
        """
        Initialize the text preprocessor.
        
        Args:
            config: Configuration object or dictionary with preprocessing options
        """
        if isinstance(config, dict):
            self.config = Config()
            self.text_preprocessing = config
        elif isinstance(config, Config):
            self.config = config
            self.text_preprocessing = config.text_preprocessing
        elif isinstance(config, list):
            # Handle list of preprocessing steps
            self.config = Config()
            self.text_preprocessing = {}
            for step in config:
                self.text_preprocessing[step] = True
        else:
            self.config = Config()
            self.text_preprocessing = self.config.text_preprocessing
        
        # Initialize NLTK components
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Get language for stopwords
        self.language = self.text_preprocessing.get('language', 'english')
        try:
            self.stop_words = set(stopwords.words(self.language))
        except Exception as e:
            logger.warning(f"Failed to load stopwords for language {self.language}: {e}")
            self.stop_words = set()
        
        logger.info(f"TextPreprocessor initialized with options: {self.text_preprocessing}")
    
    @timing_decorator
    def preprocess_text(self, text: str) -> str:
        """
        Apply all enabled preprocessing steps to a text string.
        
        Args:
            text: The input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        if self.text_preprocessing.get('lowercase', True):
            text = text.lower()
        
        # Remove punctuation
        if self.text_preprocessing.get('remove_punctuation', True):
            text = self._remove_punctuation(text)
        
        # Remove numbers
        if self.text_preprocessing.get('remove_numbers', False):
            text = self._remove_numbers(text)
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.text_preprocessing.get('remove_stopwords', True):
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming
        if self.text_preprocessing.get('stemming', False):
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Apply lemmatization
        if self.text_preprocessing.get('lemmatization', True):
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into a string
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text."""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def _remove_numbers(self, text: str) -> str:
        """Remove numbers from text."""
        return re.sub(r'\d+', '', text)
    
    def preprocess_batch(self, batch: List[Dict[str, Any]], text_field: str = 'text') -> List[Dict[str, Any]]:
        """
        Preprocess a batch of records.
        
        Args:
            batch: List of data records
            text_field: The field containing the text to preprocess
            
        Returns:
            List of preprocessed records
        """
        processed_batch = []
        
        for record in batch:
            if text_field in record:
                processed_record = record.copy()
                processed_record[f'processed_{text_field}'] = self.preprocess_text(record[text_field])
                processed_batch.append(processed_record)
            else:
                logger.warning(f"Text field '{text_field}' not found in record: {record}")
                processed_batch.append(record)
        
        return processed_batch
    
    @memory_usage_decorator
    def preprocess_and_save(self, data_loader: DataLoader, output_path: str, text_field: str = 'text') -> None:
        """
        Preprocess an entire dataset and save it to a new location.
        
        Args:
            data_loader: DataLoader instance for the dataset
            output_path: Path where to save the processed dataset
            text_field: The field containing the text to preprocess
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Determine the output format based on the input format
        output_format = data_loader.file_extension
        if not output_format:
            output_format = '.csv'  # Default to CSV if no extension detected
        
        if output_format == '.csv':
            self._save_to_csv(data_loader, output_path, text_field)
        elif output_format == '.json':
            self._save_to_json(data_loader, output_path, text_field)
        else:
            # Default to CSV for unsupported formats
            self._save_to_csv(data_loader, output_path + '.csv', text_field)
            
        logger.info(f"Preprocessed data saved to {output_path}")
    
    def _save_to_csv(self, data_loader: DataLoader, output_path: str, text_field: str) -> None:
        """Save preprocessed data to a CSV file."""
        if not output_path.endswith('.csv'):
            output_path += '.csv'
            
        headers_written = False
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            csv_writer = None
            
            for batch in data_loader.load_batch_generator():
                processed_batch = self.preprocess_batch(batch, text_field)
                
                if not headers_written and processed_batch:
                    # Create CSV writer with the fields from the first record
                    csv_writer = csv.DictWriter(f, fieldnames=processed_batch[0].keys())
                    csv_writer.writeheader()
                    headers_written = True
                
                csv_writer.writerows(processed_batch)
    
    def _save_to_json(self, data_loader: DataLoader, output_path: str, text_field: str) -> None:
        """Save preprocessed data to a JSON file."""
        if not output_path.endswith('.json'):
            output_path += '.json'
            
        all_records = []
        
        for batch in data_loader.load_batch_generator():
            processed_batch = self.preprocess_batch(batch, text_field)
            all_records.extend(processed_batch)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)
