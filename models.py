from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class Dataset(Base):
    """Model for storing dataset information."""
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    path = Column(String(512), nullable=False)
    url = Column(String(1024))  # Added URL field for dataset links
    file_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    record_count = Column(Integer)
    size_bytes = Column(Integer)
    description = Column(Text)
    
    # Relationships
    versions = relationship('DatasetVersion', back_populates='dataset', cascade='all, delete-orphan')
    stats = relationship('DatasetStats', back_populates='dataset', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Dataset(name='{self.name}', path='{self.path}')>"

class DatasetVersion(Base):
    """Model for storing dataset version information."""
    __tablename__ = 'dataset_versions'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    commit_hash = Column(String(255), nullable=False)
    message = Column(Text)
    author = Column(String(255))
    date_created = Column(DateTime, default=datetime.datetime.utcnow)
    is_current = Column(Boolean, default=False)
    
    # Relationships
    dataset = relationship('Dataset', back_populates='versions')
    processing_tasks = relationship('ProcessingTask', back_populates='dataset_version', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<DatasetVersion(commit_hash='{self.commit_hash}', message='{self.message}')>"

class DatasetStats(Base):
    """Model for storing dataset statistics."""
    __tablename__ = 'dataset_stats'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    record_count = Column(Integer)
    field_count = Column(Integer)
    avg_text_length = Column(Float)
    min_text_length = Column(Integer)
    max_text_length = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    dataset = relationship('Dataset', back_populates='stats')
    
    def __repr__(self):
        return f"<DatasetStats(dataset_id={self.dataset_id}, record_count={self.record_count})>"

class ProcessingTask(Base):
    """Model for storing dataset processing task information."""
    __tablename__ = 'processing_tasks'
    
    id = Column(Integer, primary_key=True)
    dataset_version_id = Column(Integer, ForeignKey('dataset_versions.id'), nullable=False)
    task_type = Column(String(50), nullable=False) # preprocessing, training, evaluation
    parameters = Column(Text) # JSON string of parameters
    input_path = Column(String(512))
    output_path = Column(String(512))
    status = Column(String(50), default='pending') # pending, running, completed, failed
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    execution_time = Column(Float) # seconds
    error_message = Column(Text)
    
    # Relationships
    dataset_version = relationship('DatasetVersion', back_populates='processing_tasks')
    
    def __repr__(self):
        return f"<ProcessingTask(task_type='{self.task_type}', status='{self.status}')>"

class RemoteStorage(Base):
    """Model for storing remote storage information."""
    __tablename__ = 'remote_storage'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    url = Column(String(512), nullable=False)
    storage_type = Column(String(50)) # s3, gdrive, etc.
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_sync = Column(DateTime)
    
    def __repr__(self):
        return f"<RemoteStorage(name='{self.name}', url='{self.url}')>"
