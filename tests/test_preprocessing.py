import os
import unittest
import tempfile
import shutil
import pandas as pd
import csv
import json

from src.data.preprocessing import TextPreprocessor
from src.data.data_loader import DataLoader
from src.config import Config

class TestTextPreprocessor(unittest.TestCase):
    """Test cases for the TextPreprocessor class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create test data files
        self.create_test_data()
        
        # Create default config
        self.config = Config()
        
        # Initialize preprocessor with test settings
        self.preprocessor = TextPreprocessor({
            'lowercase': True,
            'remove_punctuation': True,
            'remove_numbers': False,
            'remove_stopwords': True,
            'language': 'english',
            'stemming': False,
            'lemmatization': True
        })
    
    def tearDown(self):
        """Clean up test environment after each test."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def create_test_data(self):
        """Create test data files for testing."""
        # Create a CSV file
        csv_path = os.path.join(self.test_dir, 'test.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'text', 'label'])
            writer.writerow(['1', 'This is a positive review!', 'positive'])
            writer.writerow(['2', 'This is a negative review...', 'negative'])
            writer.writerow(['3', 'The product costs $19.99 and arrived on 01/02/2023.', 'neutral'])
            writer.writerow(['4', 'I love this product, it\'s amazing!', 'positive'])
            writer.writerow(['5', 'I hate this product, it\'s terrible!', 'negative'])
    
    def test_preprocess_text(self):
        """Test the preprocess_text method."""
        # Test basic lowercase and punctuation removal
        text = "Hello, World! This is a Test."
        processed = self.preprocessor.preprocess_text(text)
        self.assertEqual(processed, "hello world test")
        
        # Test with numbers
        text = "The product costs $19.99."
        processed = self.preprocessor.preprocess_text(text)
        self.assertEqual(processed, "product cost 1999")
        
        # Test with stopwords removal
        text = "This is a great product and I love it."
        processed = self.preprocessor.preprocess_text(text)
        self.assertNotIn("is", processed)
        self.assertNotIn("a", processed)
        self.assertNotIn("and", processed)
        self.assertIn("great", processed)
        self.assertIn("product", processed)
        self.assertIn("love", processed)
        
        # Test lemmatization
        text = "I am running and jumping"
        processed = self.preprocessor.preprocess_text(text)
        self.assertIn("run", processed)
        self.assertIn("jump", processed)
    
    def test_remove_punctuation(self):
        """Test the _remove_punctuation method."""
        text = "Hello, World! This is a Test."
        processed = self.preprocessor._remove_punctuation(text)
        self.assertEqual(processed, "Hello World This is a Test")
    
    def test_remove_numbers(self):
        """Test the _remove_numbers method."""
        text = "The product costs $19.99."
        processed = self.preprocessor._remove_numbers(text)
        self.assertEqual(processed, "The product costs $.")
    
    def test_preprocess_batch(self):
        """Test the preprocess_batch method."""
        batch = [
            {'id': '1', 'text': 'This is a positive review!', 'label': 'positive'},
            {'id': '2', 'text': 'This is a negative review...', 'label': 'negative'}
        ]
        
        processed_batch = self.preprocessor.preprocess_batch(batch)
        
        self.assertEqual(len(processed_batch), 2)
        self.assertIn('processed_text', processed_batch[0])
        self.assertEqual(processed_batch[0]['processed_text'], "positive review")
        self.assertEqual(processed_batch[1]['processed_text'], "negative review")
    
    def test_preprocess_batch_missing_field(self):
        """Test the preprocess_batch method with missing text field."""
        batch = [
            {'id': '1', 'content': 'This is a positive review!', 'label': 'positive'},
            {'id': '2', 'text': 'This is a negative review...', 'label': 'negative'}
        ]
        
        processed_batch = self.preprocessor.preprocess_batch(batch, text_field='text')
        
        self.assertEqual(len(processed_batch), 2)
        self.assertNotIn('processed_text', processed_batch[0])  # First record should be unchanged
        self.assertIn('processed_text', processed_batch[1])
    
    def test_preprocess_and_save_csv(self):
        """Test the preprocess_and_save method with CSV format."""
        # Create a DataLoader for the test CSV
        data_loader = DataLoader(os.path.join(self.test_dir, 'test.csv'), self.config)
        
        # Output path for the processed data
        output_path = os.path.join(self.test_dir, 'processed_test.csv')
        
        # Process and save the data
        self.preprocessor.preprocess_and_save(data_loader, output_path)
        
        # Verify the output file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Load the processed file and verify content
        df = pd.read_csv(output_path)
        self.assertEqual(len(df), 5)
        self.assertIn('processed_text', df.columns)
        
        # Verify a sample of processed text
        self.assertEqual(df.iloc[0]['processed_text'], "positive review")
        self.assertEqual(df.iloc[2]['processed_text'], "product cost 1999 arrived 01022023")
    
    def test_preprocess_and_save_json(self):
        """Test the preprocess_and_save method with JSON format."""
        # Create a JSON file
        json_path = os.path.join(self.test_dir, 'test.json')
        json_data = [
            {'id': '1', 'text': 'This is a positive JSON!', 'label': 'positive'},
            {'id': '2', 'text': 'This is a negative JSON...', 'label': 'negative'}
        ]
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f)
        
        # Create a DataLoader for the test JSON
        data_loader = DataLoader(json_path, self.config)
        
        # Output path for the processed data
        output_path = os.path.join(self.test_dir, 'processed_test.json')
        
        # Process and save the data
        self.preprocessor.preprocess_and_save(data_loader, output_path)
        
        # Verify the output file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Load the processed file and verify content
        with open(output_path, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        
        self.assertEqual(len(processed_data), 2)
        self.assertIn('processed_text', processed_data[0])
        self.assertEqual(processed_data[0]['processed_text'], "positive json")
        self.assertEqual(processed_data[1]['processed_text'], "negative json")
    
    def test_different_preprocessing_configurations(self):
        """Test text preprocessing with different configurations."""
        text = "Running and jumping with numbers 123!"
        
        # Test with stemming enabled instead of lemmatization
        stemming_preprocessor = TextPreprocessor({
            'lowercase': True,
            'remove_punctuation': True,
            'remove_numbers': False,
            'remove_stopwords': True,
            'language': 'english',
            'stemming': True,
            'lemmatization': False
        })
        
        stemmed = stemming_preprocessor.preprocess_text(text)
        self.assertIn("run", stemmed)
        self.assertIn("jump", stemmed)
        self.assertIn("number", stemmed)
        
        # Test with number removal enabled
        number_removal_preprocessor = TextPreprocessor({
            'lowercase': True,
            'remove_punctuation': True,
            'remove_numbers': True,
            'remove_stopwords': True,
            'language': 'english',
            'stemming': False,
            'lemmatization': True
        })
        
        no_numbers = number_removal_preprocessor.preprocess_text(text)
        self.assertNotIn("123", no_numbers)
        
        # Test with stopwords retention
        keep_stopwords_preprocessor = TextPreprocessor({
            'lowercase': True,
            'remove_punctuation': True,
            'remove_numbers': False,
            'remove_stopwords': False,
            'language': 'english',
            'stemming': False,
            'lemmatization': True
        })
        
        with_stopwords = keep_stopwords_preprocessor.preprocess_text(text)
        self.assertIn("and", with_stopwords)
        self.assertIn("with", with_stopwords)

if __name__ == '__main__':
    unittest.main()
