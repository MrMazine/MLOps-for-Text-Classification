"""
Database models for fake news classification system.
"""
import datetime
from sqlalchemy import Column, DateTime, Integer, String, Float, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.utils.database import db, Base

class NewsDataset(Base):
    """Model for storing news dataset information."""
    __tablename__ = 'news_datasets'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    path = Column(String(512), nullable=False)
    url = Column(String(1024))  # URL field for dataset links
    file_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    record_count = Column(Integer)
    size_bytes = Column(Integer)
    description = Column(Text)
    category = Column(String(50))  # Type of news: political, entertainment, etc.
    
    # Relationships
    versions = relationship('DatasetVersion', back_populates='dataset', cascade='all, delete-orphan')
    stats = relationship('DatasetStats', back_populates='dataset', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<NewsDataset(name='{self.name}', records={self.record_count})>"

class DatasetVersion(Base):
    """Model for storing dataset version information."""
    __tablename__ = 'dataset_versions'

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('news_datasets.id'), nullable=False)
    commit_hash = Column(String(255), nullable=False)
    message = Column(Text)
    author = Column(String(255))
    date_created = Column(DateTime, default=datetime.datetime.utcnow)
    is_current = Column(Boolean, default=False)

    # Relationships
    dataset = relationship('NewsDataset', back_populates='versions')
    processing_tasks = relationship('ProcessingTask', back_populates='dataset_version', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<DatasetVersion(hash='{self.commit_hash[:7]}', date='{self.date_created}')>"

class DatasetStats(Base):
    """Model for storing dataset statistics specific to news content."""
    __tablename__ = 'dataset_stats'

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('news_datasets.id'), nullable=False)
    record_count = Column(Integer)
    field_count = Column(Integer)
    
    # Text statistics
    avg_title_length = Column(Float)
    avg_article_length = Column(Float)
    min_article_length = Column(Integer)
    max_article_length = Column(Integer)
    
    # News specific statistics
    top_entities = Column(Text)  # JSON string of named entities
    sentiment_distribution = Column(Text)  # JSON string of sentiment scores
    topic_distribution = Column(Text)  # JSON string of topic model
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    dataset = relationship('NewsDataset', back_populates='stats')
    
    def __repr__(self):
        return f"<DatasetStats(dataset_id={self.dataset_id}, records={self.record_count})>"

class ProcessingTask(Base):
    """Model for storing dataset processing task information."""
    __tablename__ = 'processing_tasks'

    id = Column(Integer, primary_key=True)
    dataset_version_id = Column(Integer, ForeignKey('dataset_versions.id'), nullable=False)
    task_type = Column(String(50), nullable=False)  # preprocessing, feature_extraction, model_training, evaluation
    parameters = Column(Text)  # JSON string of parameters
    input_path = Column(String(512))
    output_path = Column(String(512))
    status = Column(String(50), default='pending')  # pending, running, completed, failed
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    execution_time = Column(Float)  # seconds
    error_message = Column(Text)
    
    # For fake news specific tasks
    accuracy = Column(Float)  # Model accuracy if task_type is evaluation
    precision = Column(Float)  # Precision score if task_type is evaluation
    recall = Column(Float)  # Recall score if task_type is evaluation
    f1_score = Column(Float)  # F1 score if task_type is evaluation

    # Relationships
    dataset_version = relationship('DatasetVersion', back_populates='processing_tasks')
    
    def __repr__(self):
        return f"<ProcessingTask(type='{self.task_type}', status='{self.status}')>"

class RemoteStorage(Base):
    """Model for storing remote storage information."""
    __tablename__ = 'remote_storage'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    url = Column(String(512), nullable=False)
    storage_type = Column(String(50))  # s3, gdrive, etc.
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_sync = Column(DateTime)
    
    def __repr__(self):
        return f"<RemoteStorage(name='{self.name}', type='{self.storage_type}')>"

class Model(Base):
    """Model for storing information about trained fake news detection models."""
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    algorithm = Column(String(100))  # LSTM, Transformer, etc.
    path = Column(String(512))  # Path to model file
    training_dataset_id = Column(Integer, ForeignKey('dataset_versions.id'))
    validation_dataset_id = Column(Integer, ForeignKey('dataset_versions.id'))
    parameters = Column(Text)  # JSON string of hyperparameters
    metrics = Column(Text)  # JSON string of evaluation metrics
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<Model(name='{self.name}', version='{self.version}')>"