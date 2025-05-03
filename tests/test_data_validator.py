import os
import unittest
import tempfile
import shutil
import csv
import json

from src.data.data_validator import DataValidator
from src.data.data_loader import DataLoader
from src.config import Config

class TestDataValidator(unittest.TestCase):
    """Test cases for the DataValidator class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create test data files
        self.create_test_data()
        
        # Create config
        self.config = Config()
        
        # Initialize validator
        self.validator = DataValidator(min_text_length=5, require_fields=['text', 'label'])
    
    def tearDown(self):
        """Clean up test environment after each test."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def create_test_data(self):
        """Create test data files for testing."""
        # Create a valid CSV file
        valid_csv_path = os.path.join(self.test_dir, 'valid.csv')
        with open(valid_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'text', 'label'])
            writer.writerow(['1', 'This is a positive review', 'positive'])
            writer.writerow(['2', 'This is a negative review', 'negative'])
            writer.writerow(['3', 'Neutral review with additional information', 'neutral'])
        
        # Create an invalid CSV file (missing required field)
        invalid_csv_path = os.path.join(self.test_dir, 'invalid_missing_field.csv')
        with open(invalid_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'content', 'label'])  # 'text' field is missing
            writer.writerow(['1', 'This is a positive review', 'positive'])
            writer.writerow(['2', 'This is a negative review', 'negative'])
        
        # Create an invalid CSV file (too short text)
        short_text_csv_path = os.path.join(self.test_dir, 'invalid_short_text.csv')
        with open(short_text_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'text', 'label'])
            writer.writerow(['1', 'OK', 'positive'])  # Text is too short
            writer.writerow(['2', 'Bad', 'negative'])  # Text is too short
            writer.writerow(['3', 'This is fine', 'neutral'])  # This one is fine
        
        # Create an empty CSV file
        empty_csv_path = os.path.join(self.test_dir, 'empty.csv')
        with open(empty_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'text', 'label'])
            # No data rows
        
        # Create a valid JSON file
        valid_json_path = os.path.join(self.test_dir, 'valid.json')
        json_data = [
            {'id': '1', 'text': 'This is a positive JSON review', 'label': 'positive'},
            {'id': '2', 'text': 'This is a negative JSON review', 'label': 'negative'}
        ]
        with open(valid_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f)
        
        # Create a text file
        txt_path = os.path.join(self.test_dir, 'sample.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('This is a sample text line 1.\n')
            f.write('This is a sample text line 2.\n')
            f.write('This is a sample text line 3.\n')
        
        # Create an unsupported file format
        unsupported_path = os.path.join(self.test_dir, 'unsupported.xyz')
        with open(unsupported_path, 'w', encoding='utf-8') as f:
            f.write('This is an unsupported file format.\n')
    
    def test_validate_valid_dataset(self):
        """Test validation of a valid dataset."""
        valid_path = os.path.join(self.test_dir, 'valid.csv')
        is_valid, message = self.validator.validate_dataset(valid_path)
        
        self.assertTrue(is_valid)
        self.assertEqual(message, "Dataset is valid")
    
    def test_validate_missing_field(self):
        """Test validation of a dataset with missing required field."""
        invalid_path = os.path.join(self.test_dir, 'invalid_missing_field.csv')
        is_valid, message = self.validator.validate_dataset(invalid_path)
        
        self.assertFalse(is_valid)
        self.assertIn("Required fields are missing", message)
    
    def test_validate_short_text(self):
        """Test validation of a dataset with text that is too short."""
        short_text_path = os.path.join(self.test_dir, 'invalid_short_text.csv')
        is_valid, message = self.validator.validate_dataset(short_text_path)
        
        self.assertFalse(is_valid)
        self.assertIn("Too many text fields below minimum length", message)
    
    def test_validate_empty_dataset(self):
        """Test validation of an empty dataset."""
        empty_path = os.path.join(self.test_dir, 'empty.csv')
        is_valid, message = self.validator.validate_dataset(empty_path)
        
        self.assertFalse(is_valid)
        self.assertEqual(message, "Dataset is empty or cannot be read")
    
    def test_validate_nonexistent_file(self):
        """Test validation of a nonexistent file."""
        nonexistent_path = os.path.join(self.test_dir, 'nonexistent.csv')
        is_valid, message = self.validator.validate_dataset(nonexistent_path)
        
        self.assertFalse(is_valid)
        self.assertIn("Data path does not exist", message)
    
    def test_validate_unsupported_format(self):
        """Test validation of an unsupported file format."""
        unsupported_path = os.path.join(self.test_dir, 'unsupported.xyz')
        is_valid, message = self.validator.validate_dataset(unsupported_path)
        
        self.assertFalse(is_valid)
        self.assertIn("Unsupported file format", message)
    
    def test_validate_json_dataset(self):
        """Test validation of a JSON dataset."""
        json_path = os.path.join(self.test_dir, 'valid.json')
        is_valid, message = self.validator.validate_dataset(json_path)
        
        self.assertTrue(is_valid)
        self.assertEqual(message, "Dataset is valid")
    
    def test_validate_text_file(self):
        """Test validation of a text file."""
        txt_path = os.path.join(self.test_dir, 'sample.txt')
        
        # Create a validator without requiring the 'label' field
        text_validator = DataValidator(min_text_length=5, require_fields=['text'])
        is_valid, message = text_validator.validate_dataset(txt_path)
        
        self.assertTrue(is_valid)
        self.assertEqual(message, "Dataset is valid")
    
    def test_validate_directory(self):
        """Test validation of a directory with multiple files."""
        # Create a validator that only requires 'text' field
        dir_validator = DataValidator(min_text_length=5, require_fields=['text'])
        is_valid, message = dir_validator.validate_dataset(self.test_dir)
        
        # Should be valid as long as at least one file is valid
        self.assertTrue(is_valid)
        self.assertEqual(message, "Dataset is valid")
    
    def test_validate_text_field(self):
        """Test validation of individual text fields."""
        # Valid text
        is_valid, message = self.validator.validate_text_field("This is a good text field")
        self.assertTrue(is_valid)
        self.assertEqual(message, "Text is valid")
        
        # Too short
        is_valid, message = self.validator.validate_text_field("Hi")
        self.assertFalse(is_valid)
        self.assertIn("Text is too short", message)
        
        # Empty
        is_valid, message = self.validator.validate_text_field("")
        self.assertFalse(is_valid)
        self.assertEqual(message, "Text is empty or not a string")
        
        # Not a string
        is_valid, message = self.validator.validate_text_field(123)
        self.assertFalse(is_valid)
        self.assertEqual(message, "Text is empty or not a string")
        
        # Too many special characters
        is_valid, message = self.validator.validate_text_field("@#$%^&*!@#$%^&*!@#$%^&*!")
        self.assertFalse(is_valid)
        self.assertIn("Text contains too many special characters", message)
        
        # Excessive repeated characters
        is_valid, message = self.validator.validate_text_field("aaaaaaaaaaaaaabbbbbbbbbbbbb")
        self.assertFalse(is_valid)
        self.assertIn("Text contains excessive repeated characters", message)

if __name__ == '__main__':
    unittest.main()
