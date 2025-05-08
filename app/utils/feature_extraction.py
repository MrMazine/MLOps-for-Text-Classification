"""
Feature extraction utilities for fake news classification.
"""
import os
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Class for extracting features from text data for fake news classification."""
    
    def __init__(self, params=None):
        """
        Initialize the feature extractor with parameters.
        
        Args:
            params: Dictionary of feature extraction parameters
        """
        self.params = params or {}
        self.tfidf_vectorizer = None
        self.label_encoder = None
        
        logger.info(f"FeatureExtractor initialized with params: {self.params}")
        
    def _create_tfidf_features(self, texts):
        """
        Create TF-IDF features from texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            TF-IDF feature matrix
        """
        max_features = self.params.get('max_features', 5000)
        
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                stop_words='english'
            )
            return self.tfidf_vectorizer.fit_transform(texts)
        else:
            return self.tfidf_vectorizer.transform(texts)
            
    def _create_sequence_features(self, texts):
        """
        Create sequence features for deep learning models.
        This is a placeholder for more complex word embedding-based features.
        
        Args:
            texts: List of text strings
            
        Returns:
            Sequence feature matrix and word index
        """
        # For now, simply create integer indices for words (placeholder)
        # In a real implementation, this would use embeddings like word2vec, GloVe, or BERT
        
        # Tokenize texts and create vocabulary
        tokenized_texts = [text.split() for text in texts]
        all_words = set()
        for tokens in tokenized_texts:
            all_words.update(tokens)
            
        # Create word index
        word_index = {word: i+1 for i, word in enumerate(all_words)}
        
        # Convert texts to sequences of indices
        max_seq_length = self.params.get('max_sequence_length', 500)
        sequences = []
        
        for tokens in tokenized_texts:
            seq = [word_index.get(word, 0) for word in tokens[:max_seq_length]]
            # Pad sequences to max_seq_length
            if len(seq) < max_seq_length:
                seq = seq + [0] * (max_seq_length - len(seq))
            sequences.append(seq)
            
        return np.array(sequences), word_index
        
    def _encode_labels(self, labels):
        """
        Encode categorical labels as integers.
        
        Args:
            labels: List of label strings
            
        Returns:
            Encoded label array
        """
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            return self.label_encoder.fit_transform(labels)
        else:
            return self.label_encoder.transform(labels)
            
    def extract_features(self, texts, labels=None):
        """
        Extract features from texts and encode labels.
        
        Args:
            texts: List of text strings
            labels: List of label strings (optional)
            
        Returns:
            Dictionary with features and metadata
        """
        result = {}
        
        # Choose feature extraction method
        if self.params.get('use_tfidf', True):
            # TF-IDF features for traditional ML models
            X = self._create_tfidf_features(texts)
            result['features'] = X
            result['feature_type'] = 'tfidf'
            
            # Add feature names
            if self.tfidf_vectorizer:
                result['feature_names'] = self.tfidf_vectorizer.get_feature_names_out()
                
        elif self.params.get('use_word_embeddings', False):
            # Sequence features for deep learning models
            X, word_index = self._create_sequence_features(texts)
            result['features'] = X
            result['feature_type'] = 'sequence'
            result['word_index'] = word_index
            result['vocab_size'] = len(word_index) + 1  # +1 for padding
            
        # Encode labels if provided
        if labels is not None:
            y = self._encode_labels(labels)
            result['labels'] = y
            
            # Add label mapping
            if self.label_encoder:
                result['label_mapping'] = dict(zip(
                    self.label_encoder.classes_,
                    self.label_encoder.transform(self.label_encoder.classes_)
                ))
                
        return result
        
    def extract_and_save(self, input_path, output_path, text_column='text', label_column=None):
        """
        Extract features from a dataset and save to output file.
        
        Args:
            input_path: Path to input dataset file
            output_path: Path to output feature file
            text_column: Name of the text column
            label_column: Name of the label column (optional)
            
        Returns:
            Tuple of (success, statistics)
        """
        try:
            # Determine file type from extension
            file_type = os.path.splitext(input_path)[1].lower()
            
            if file_type not in ['.csv', '.json']:
                logger.error(f"Unsupported file type: {file_type}")
                return False, {}
                
            # Load dataset
            if file_type == '.csv':
                df = pd.read_csv(input_path)
            else:
                df = pd.read_json(input_path)
                
            # Check if text column exists
            if text_column not in df.columns:
                logger.error(f"Text column '{text_column}' not found in dataset")
                return False, {}
                
            # Extract texts and labels
            texts = df[text_column].fillna('').tolist()
            labels = df[label_column].tolist() if label_column and label_column in df.columns else None
            
            # Extract features
            result = self.extract_features(texts, labels)
            
            # Save features and metadata to NPZ file
            if 'features' in result:
                # Convert sparse matrix to dense if necessary
                features = result['features']
                if hasattr(features, 'toarray'):
                    features = features.toarray()
                    
                # Save features and metadata
                features_dict = {
                    'features': features,
                    'feature_type': result.get('feature_type', 'unknown')
                }
                
                # Add labels if available
                if 'labels' in result:
                    features_dict['labels'] = result['labels']
                    
                # Add additional metadata
                if 'feature_names' in result:
                    features_dict['feature_names'] = result['feature_names']
                    
                if 'label_mapping' in result:
                    # Convert dict to arrays for saving
                    label_classes = np.array(list(result['label_mapping'].keys()))
                    label_indices = np.array(list(result['label_mapping'].values()))
                    
                    features_dict['label_classes'] = label_classes
                    features_dict['label_indices'] = label_indices
                    
                if 'word_index' in result:
                    # Convert word index to arrays for saving
                    words = np.array(list(result['word_index'].keys()))
                    indices = np.array(list(result['word_index'].values()))
                    
                    features_dict['words'] = words
                    features_dict['indices'] = indices
                    features_dict['vocab_size'] = result.get('vocab_size', 0)
                    
                # Save to NPZ file
                np.savez_compressed(output_path, **features_dict)
                
                # Calculate statistics
                stats = {
                    'record_count': len(texts),
                    'feature_count': features.shape[1] if len(features.shape) > 1 else features.shape[0],
                    'feature_type': result.get('feature_type', 'unknown')
                }
                
                if 'labels' in result:
                    stats['label_count'] = len(np.unique(result['labels']))
                    
                logger.info(f"Feature extraction completed: {stats}")
                return True, stats
            else:
                logger.error("No features extracted")
                return False, {}
                
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return False, {}