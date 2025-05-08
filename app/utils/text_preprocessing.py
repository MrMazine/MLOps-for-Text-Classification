"""
Text preprocessing utilities for fake news classification.
"""
import re
import os
import logging
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextPreprocessor:
    """Class for preprocessing text data specifically for fake news classification."""
    
    def __init__(self, params=None):
        """
        Initialize the text preprocessor with parameters.
        
        Args:
            params: Dictionary of preprocessing parameters
        """
        self.params = params or {}
        self.lemmatizer = WordNetLemmatizer() if self.params.get('lemmatize', True) else None
        self.stemmer = PorterStemmer() if self.params.get('stemming', False) else None
        self.stop_words = set(stopwords.words('english')) if self.params.get('remove_stopwords', True) else set()
        
        logger.info(f"TextPreprocessor initialized with params: {self.params}")
        
    def _remove_punctuation(self, text):
        """Remove punctuation from text."""
        return re.sub(r'[^\w\s]', ' ', text)
        
    def _remove_numbers(self, text):
        """Remove numbers from text."""
        return re.sub(r'\d+', '', text)
        
    def _remove_urls(self, text):
        """Remove URLs from text."""
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
    def _remove_html_tags(self, text):
        """Remove HTML tags from text."""
        return re.sub(r'<.*?>', '', text)
        
    def _remove_extra_whitespace(self, text):
        """Remove extra whitespace from text."""
        return re.sub(r'\s+', ' ', text).strip()
    
    def preprocess_text(self, text):
        """
        Preprocess a single text string.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text string
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Apply preprocessing steps based on parameters
        text = self._remove_html_tags(text)
        text = self._remove_urls(text)
        
        if self.params.get('lowercase', True):
            text = text.lower()
            
        if self.params.get('remove_punctuation', True):
            text = self._remove_punctuation(text)
            
        if self.params.get('remove_numbers', False):
            text = self._remove_numbers(text)
            
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.params.get('remove_stopwords', True):
            tokens = [word for word in tokens if word not in self.stop_words]
            
        # Apply lemmatization
        if self.params.get('lemmatize', True) and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            
        # Apply stemming
        if self.params.get('stemming', False) and self.stemmer:
            tokens = [self.stemmer.stem(word) for word in tokens]
            
        # Rejoin tokens into text
        processed_text = ' '.join(tokens)
        
        # Remove extra whitespace
        processed_text = self._remove_extra_whitespace(processed_text)
        
        return processed_text
        
    def preprocess_batch(self, texts):
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of preprocessed text strings
        """
        return [self.preprocess_text(text) for text in texts]
        
    def preprocess_and_save(self, input_path, output_path, text_column='text', batch_size=1000):
        """
        Preprocess a dataset and save to output file.
        
        Args:
            input_path: Path to input dataset file
            output_path: Path to output processed file
            text_column: Name of the text column to preprocess
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (success, statistics)
        """
        try:
            # Determine file type from extension
            file_type = os.path.splitext(input_path)[1].lower()
            
            if file_type == '.csv':
                # Process CSV file in batches
                stats = self._process_csv(input_path, output_path, text_column, batch_size)
                return True, stats
            elif file_type == '.json':
                # Process JSON file
                df = pd.read_json(input_path)
                if text_column not in df.columns:
                    logger.error(f"Text column '{text_column}' not found in dataset")
                    return False, {}
                    
                df[text_column] = self.preprocess_batch(df[text_column].tolist())
                df.to_json(output_path, orient='records')
                
                return True, {
                    'record_count': len(df),
                    'avg_text_length': df[text_column].str.len().mean()
                }
            else:
                logger.error(f"Unsupported file type: {file_type}")
                return False, {}
                
        except Exception as e:
            logger.error(f"Error preprocessing dataset: {str(e)}")
            return False, {}
            
    def _process_csv(self, input_path, output_path, text_column, batch_size):
        """Process CSV file in batches."""
        # Check if file exists
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            return {}
            
        # Initialize statistics
        total_records = 0
        text_lengths = []
        
        # Process in batches
        chunk_iter = pd.read_csv(input_path, chunksize=batch_size)
        for i, chunk in enumerate(chunk_iter):
            if i == 0:
                # Check if text column exists
                if text_column not in chunk.columns:
                    logger.error(f"Text column '{text_column}' not found in dataset")
                    return {}
                
                # Create output file
                mode = 'w'
                header = True
            else:
                # Append to output file
                mode = 'a'
                header = False
                
            # Preprocess text column
            chunk[text_column] = self.preprocess_batch(chunk[text_column].tolist())
            
            # Update statistics
            total_records += len(chunk)
            text_lengths.extend(chunk[text_column].str.len().tolist())
            
            # Save batch
            chunk.to_csv(output_path, mode=mode, header=header, index=False)
            
            logger.info(f"Processed batch {i+1} ({len(chunk)} records)")
            
        # Calculate statistics
        if text_lengths:
            avg_text_length = np.mean(text_lengths)
            min_text_length = np.min(text_lengths)
            max_text_length = np.max(text_lengths)
        else:
            avg_text_length = 0
            min_text_length = 0
            max_text_length = 0
            
        stats = {
            'record_count': total_records,
            'avg_text_length': avg_text_length,
            'min_text_length': min_text_length,
            'max_text_length': max_text_length
        }
        
        logger.info(f"Preprocessing completed: {stats}")
        return stats