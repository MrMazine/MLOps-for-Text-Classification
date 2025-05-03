import os
import unittest
import tempfile
import shutil
import pandas as pd
import json
import csv

from src.data.data_loader import DataLoader
from src.config import Config

class TestDataLoader(unittest.TestCase):
    """Test cases for the DataLoader class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create test data files
        self.create_test_data()
        
        # Create a config with test settings
        self.config = Config()
        self.config.batch_size = 5
    
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
            writer.writerow(['1', 'This is a positive review.', 'positive'])
            writer.writerow(['2', 'This is a negative review.', 'negative'])
            writer.writerow(['3', 'This is a neutral review.', 'neutral'])
            writer.writerow(['4', 'This is another positive review.', 'positive'])
            writer.writerow(['5', 'This is another negative review.', 'negative'])
            writer.writerow(['6', 'This is another neutral review.', 'neutral'])
            writer.writerow(['7', 'Final positive review.', 'positive'])
            writer.writerow(['8', 'Final negative review.', 'negative'])
        
        # Create a JSON file
        json_path = os.path.join(self.test_dir, 'test.json')
        json_data = [
            {'id': '1', 'text': 'JSON positive text.', 'label': 'positive'},
            {'id': '2', 'text': 'JSON negative text.', 'label': 'negative'},
            {'id': '3', 'text': 'JSON neutral text.', 'label': 'neutral'}
        ]
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f)
        
        # Create a text file
        txt_path = os.path.join(self.test_dir, 'test.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('This is a sample text line 1.\n')
            f.write('This is a sample text line 2.\n')
            f.write('This is a sample text line 3.\n')
        
        # Create a subdirectory with additional data
        subdir = os.path.join(self.test_dir, 'subdir')
        os.makedirs(subdir, exist_ok=True)
        
        # Create a CSV in the subdirectory
        sub_csv_path = os.path.join(subdir, 'sub_test.csv')
        with open(sub_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'text', 'label'])
            writer.writerow(['101', 'Subdir positive text.', 'positive'])
            writer.writerow(['102', 'Subdir negative text.', 'negative'])
    
    def test_initialization(self):
        """Test DataLoader initialization."""
        # Test with file path
        loader = DataLoader(os.path.join(self.test_dir, 'test.csv'), self.config)
        self.assertEqual(loader.file_extension, '.csv')
        self.assertFalse(loader.is_directory)
        
        # Test with directory path
        loader = DataLoader(self.test_dir, self.config)
        self.assertTrue(loader.is_directory)
    
    def test_file_iterator(self):
        """Test the file iterator functionality."""
        # Test with file path
        loader = DataLoader(os.path.join(self.test_dir, 'test.csv'), self.config)
        files = list(loader._file_iterator())
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith('test.csv'))
        
        # Test with directory path
        loader = DataLoader(self.test_dir, self.config)
        files = list(loader._file_iterator())
        self.assertEqual(len(files), 4)  # 3 files in main dir + 1 in subdir
    
    def test_load_batch_generator(self):
        """Test the batch generator functionality."""
        # Test with CSV file
        loader = DataLoader(os.path.join(self.test_dir, 'test.csv'), self.config)
        batches = list(loader.load_batch_generator())
        
        # Should have 2 batches (8 rows / 5 batch size, rounded up)
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 5)  # First batch has 5 records
        self.assertEqual(len(batches[1]), 3)  # Second batch has 3 records
        
        # Verify content
        first_record = batches[0][0]
        self.assertEqual(first_record['id'], '1')
        self.assertEqual(first_record['text'], 'This is a positive review.')
        self.assertEqual(first_record['label'], 'positive')
    
    def test_load_pandas(self):
        """Test loading data into pandas DataFrame."""
        # Test with CSV file
        loader = DataLoader(os.path.join(self.test_dir, 'test.csv'), self.config)
        df = loader.load_pandas()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 8)
        self.assertEqual(list(df.columns), ['id', 'text', 'label'])
        
        # Verify content
        self.assertEqual(df.iloc[0]['text'], 'This is a positive review.')
        self.assertEqual(df.iloc[0]['label'], 'positive')
    
    def test_load_multiple_files(self):
        """Test loading data from multiple files in a directory."""
        loader = DataLoader(self.test_dir, self.config)
        df = loader.load_pandas()
        
        # Should have data from all files
        # 8 from test.csv + 3 from test.json + 3 from test.txt + 2 from subdir/sub_test.csv
        self.assertEqual(len(df), 16)
    
    def test_get_statistics(self):
        """Test the get_statistics functionality."""
        loader = DataLoader(os.path.join(self.test_dir, 'test.csv'), self.config)
        stats = loader.get_statistics()
        
        self.assertEqual(stats['record_count'], 8)
        self.assertIn('avg_text_length', stats)
        self.assertIn('fields', stats)
        self.assertIn('sample_records', stats)
        self.assertLessEqual(len(stats['sample_records']), 10)
    
    def test_iterator_interface(self):
        """Test the iterator interface of DataLoader."""
        loader = DataLoader(os.path.join(self.test_dir, 'test.csv'), self.config)
        records = list(loader)
        
        self.assertEqual(len(records), 8)
        self.assertEqual(records[0]['text'], 'This is a positive review.')
        self.assertEqual(records[0]['label'], 'positive')

if __name__ == '__main__':
    unittest.main()
