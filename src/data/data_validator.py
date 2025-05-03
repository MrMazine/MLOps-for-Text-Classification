import os
import logging
import pandas as pd
import json
import csv
from typing import Dict, Any, List, Tuple, Union, Optional
import re

from src.data.data_loader import DataLoader
from src.utils.decorators import timing_decorator

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates datasets for text classification tasks.
    Performs checks to ensure data quality and compatibility.
    """
    
    def __init__(self, min_text_length: int = 10, require_fields: Optional[List[str]] = None):
        """
        Initialize the data validator.
        
        Args:
            min_text_length: Minimum required length for text fields
            require_fields: List of field names that are required in the dataset
        """
        self.min_text_length = min_text_length
        self.require_fields = require_fields or ['text']
        logger.info(f"DataValidator initialized with min_text_length={min_text_length}, require_fields={require_fields}")
    
    @timing_decorator
    def validate_dataset(self, data_path: str) -> Tuple[bool, str]:
        """
        Validate a dataset file or directory.
        
        Args:
            data_path: Path to the dataset file or directory
            
        Returns:
            Tuple containing (is_valid, validation_message)
        """
        if not os.path.exists(data_path):
            return False, f"Data path does not exist: {data_path}"
        
        try:
            data_loader = DataLoader(data_path)
            
            # Validate file format
            if not self._validate_file_format(data_path):
                return False, f"Unsupported file format: {data_loader.file_extension}"
            
            # Validate content
            is_valid, msg = self._validate_content(data_loader)
            if not is_valid:
                return False, msg
            
            return True, "Dataset is valid"
            
        except Exception as e:
            logger.error(f"Error validating dataset: {str(e)}")
            return False, f"Error validating dataset: {str(e)}"
    
    def _validate_file_format(self, data_path: str) -> bool:
        """
        Validate that the file format is supported.
        
        Args:
            data_path: Path to the dataset file or directory
            
        Returns:
            True if format is supported, False otherwise
        """
        # For directories, check if there's at least one valid file
        if os.path.isdir(data_path):
            has_valid_file = False
            for root, _, files in os.walk(data_path):
                for file in files:
                    _, ext = os.path.splitext(file)
                    if ext.lower() in ['.csv', '.json', '.txt']:
                        has_valid_file = True
                        break
                if has_valid_file:
                    break
            return has_valid_file
        else:
            # For individual files, check the extension
            _, ext = os.path.splitext(data_path)
            return ext.lower() in ['.csv', '.json', '.txt']
    
    def _validate_content(self, data_loader: DataLoader) -> Tuple[bool, str]:
        """
        Validate the content of the dataset.
        
        Args:
            data_loader: DataLoader instance for the dataset
            
        Returns:
            Tuple containing (is_valid, validation_message)
        """
        # Check first batch for required fields and data quality
        try:
            first_batch = next(data_loader.load_batch_generator())
            
            if not first_batch:
                return False, "Dataset is empty"
            
            # Check for required fields
            sample_record = first_batch[0]
            missing_fields = [field for field in self.require_fields if field not in sample_record]
            
            if missing_fields:
                return False, f"Required fields are missing: {missing_fields}"
            
            # Validate text field
            invalid_records = 0
            empty_text_records = 0
            
            for record in first_batch:
                if 'text' in record:
                    text = record['text']
                    if not text or not isinstance(text, str):
                        empty_text_records += 1
                    elif len(text) < self.min_text_length:
                        invalid_records += 1
            
            # Use percentages to determine validity
            total_records = len(first_batch)
            empty_text_percentage = (empty_text_records / total_records) * 100
            invalid_percentage = (invalid_records / total_records) * 100
            
            if empty_text_percentage > 20:
                return False, f"Too many empty text fields: {empty_text_percentage:.1f}% of records"
            
            if invalid_percentage > 20:
                return False, f"Too many text fields below minimum length: {invalid_percentage:.1f}% of records"
            
            return True, "Content validation passed"
            
        except StopIteration:
            return False, "Dataset is empty or cannot be read"
        except Exception as e:
            logger.error(f"Error validating content: {str(e)}")
            return False, f"Error validating content: {str(e)}"
    
    def validate_text_field(self, text: str) -> Tuple[bool, str]:
        """
        Validate a single text field.
        
        Args:
            text: The text to validate
            
        Returns:
            Tuple containing (is_valid, validation_message)
        """
        if not text or not isinstance(text, str):
            return False, "Text is empty or not a string"
        
        if len(text) < self.min_text_length:
            return False, f"Text is too short (minimum {self.min_text_length} characters)"
        
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text)
        if special_char_ratio > 0.3:
            return False, "Text contains too many special characters"
        
        # Check for excessive repetition (like keyboard mashing)
        repeated_chars = re.findall(r'(.)\1{5,}', text)
        if repeated_chars:
            return False, "Text contains excessive repeated characters"
        
        return True, "Text is valid"
