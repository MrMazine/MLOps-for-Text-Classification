# Fake News Classification MLOps Project Guide

## Project Overview

This project is a comprehensive MLOps system for fake news detection, implementing end-to-end data management, preprocessing, feature extraction, model training, and evaluation with robust DevOps practices. The system helps manage and version datasets, preprocess text data, build classification models, and evaluate their performance.

## Folder Structure

```
fake-news-classifier/
│
├── app/                         # Main application code
│   ├── api/                     # API endpoints
│   │   ├── dataset_routes.py    # Dataset management endpoints
│   │   ├── preprocessing_routes.py  # Preprocessing endpoints
│   │   ├── model_routes.py      # Model training/evaluation endpoints
│   │   └── visualization_routes.py  # Data visualization endpoints
│   │
│   ├── models/                  # Database models
│   │   └── database.py          # SQLAlchemy models
│   │
│   ├── templates/               # HTML templates
│   │   ├── index.html           # Home page
│   │   ├── datasets.html        # Dataset management page
│   │   ├── preprocessing.html   # Text preprocessing page
│   │   ├── models.html          # Model training page
│   │   └── visualization.html   # Data visualization page
│   │
│   ├── static/                  # Static assets
│   │   ├── css/                 # CSS stylesheets
│   │   ├── js/                  # JavaScript files
│   │   └── img/                 # Images
│   │
│   └── utils/                   # Utility modules
│       ├── config.py            # Configuration management
│       ├── database.py          # Database connection utils
│       ├── dvc_handler.py       # DVC operations
│       ├── text_preprocessing.py # Text preprocessing utilities
│       └── feature_extraction.py # Feature extraction utilities
│
├── config/                      # Configuration files
│   └── config.yaml              # Main configuration
│
├── data/                        # Data directory (managed by DVC)
│   ├── raw/                     # Raw datasets
│   ├── processed/               # Processed datasets
│   └── external/                # External datasets
│
├── logs/                        # Log files
│
├── models/                      # Trained models
│   ├── svm/                     # SVM models
│   ├── naive_bayes/             # Naive Bayes models
│   └── transformers/            # Transformer models
│
├── notebooks/                   # Jupyter notebooks for exploration
│
├── scripts/                     # Utility scripts
│   ├── setup.py                 # Setup script
│   └── data_import.py           # Data import script
│
├── tests/                       # Test suite
│   ├── test_data_loader.py      # Tests for data loading
│   ├── test_preprocessing.py    # Tests for preprocessing
│   └── test_models.py           # Tests for models
│
├── .dvcignore                   # DVC ignore file
├── .gitignore                   # Git ignore file
├── main.py                      # Application entry point
├── requirements.txt             # Package dependencies
├── README.md                    # Project documentation
└── dvc.yaml                     # DVC pipeline definition
```

## Key Components

### 1. Database Models

Located in `app/models/database.py`, the database models define the schema for:

- **NewsDataset**: Stores dataset information (name, path, URL, type, size)
- **DatasetVersion**: Tracks versions of datasets with commit hashes
- **DatasetStats**: Stores dataset statistics (record count, text lengths)
- **ProcessingTask**: Logs preprocessing and feature extraction tasks
- **RemoteStorage**: Tracks remote storage configurations
- **Model**: Stores information about trained models

### 2. DVC Handler

Located in `app/utils/dvc_handler.py`, the DVC handler manages data versioning:

- Initializes DVC repository
- Tracks datasets with git-like versioning
- Manages dataset versions and history
- Enables checkout of specific dataset versions
- Integrates with remote storage (S3, Google Drive)

### 3. Text Preprocessing

Located in `app/utils/text_preprocessing.py`, the text preprocessing utilities:

- Clean and normalize text data
- Remove punctuation, numbers, and stopwords
- Perform lemmatization and stemming
- Process datasets in memory-efficient batches

### 4. Feature Extraction

Located in `app/utils/feature_extraction.py`, the feature extraction utilities:

- Generate TF-IDF vectors
- Create sequence features for deep learning
- Encode labels for classification
- Save features in compressed format

### 5. API Routes

Located in `app/api/`, the API routes handle:

- Dataset upload, versioning, and management
- Text preprocessing and feature extraction
- Model training and evaluation
- Data visualization

### 6. Web Interface

Located in `app/templates/`, the web interface provides:

- Dataset management (upload, version, download)
- Preprocessing configuration and execution
- Model training and evaluation
- Results visualization

## Workflow

### Data Management Workflow

1. **Upload Dataset**: Upload a news dataset through the web interface or API
2. **Version Control**: The dataset is automatically tracked with DVC
3. **Explore Statistics**: View dataset statistics and information
4. **Version Management**: Manage dataset versions and history

### Preprocessing Workflow

1. **Select Dataset**: Choose a dataset to preprocess
2. **Configure Options**: Set text preprocessing parameters
3. **Run Preprocessing**: Execute the preprocessing task
4. **Track Results**: The processed dataset is versioned with DVC

### Feature Extraction Workflow

1. **Select Dataset**: Choose a preprocessed dataset
2. **Configure Options**: Set feature extraction parameters
3. **Run Extraction**: Execute the feature extraction task
4. **Track Results**: The feature set is saved and versioned

### Model Training Workflow

1. **Select Features**: Choose a feature set for training
2. **Configure Model**: Set model type and hyperparameters
3. **Train Model**: Execute the training task
4. **Evaluate Model**: Evaluate the model's performance

## Configuration

The system is configured through `config/config.yaml`, which includes:

- Data directory paths
- Preprocessing options
- Feature extraction parameters
- Model configuration
- DVC remote storage settings

## Database Setup

The system uses PostgreSQL for persistent storage. Database tables are automatically created when the application starts.

## DVC Integration

Data Version Control (DVC) is used to track datasets and models. The main commands are:

- `dvc init`: Initialize DVC in the repository
- `dvc add <file>`: Track a file with DVC
- `dvc push`: Push datasets to remote storage
- `dvc pull`: Pull datasets from remote storage

## Running the Application

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```
   export DATABASE_URL=postgresql://user:password@localhost/dbname
   export FLASK_SECRET_KEY=your_secret_key
   ```

3. Run the application:
   ```
   gunicorn --bind 0.0.0.0:5000 main:app
   ```

## Testing

Tests are located in the `tests/` directory and can be run with:
```
pytest
```

## Security Considerations

- API authentication is required for sensitive operations
- Dataset permissions are enforced at the database level
- Secrets are stored in environment variables, not in code
- Input validation is performed for all API endpoints

## Best Practices

- Follow the DRY (Don't Repeat Yourself) principle
- Use meaningful variable and function names
- Document code with docstrings and comments
- Write tests for all new functionality
- Use type hints for better code readability
- Follow PEP 8 style guidelines

## Troubleshooting

Common issues and solutions:

1. **Database Connection Errors**: Ensure the DATABASE_URL environment variable is set correctly
2. **DVC Initialization Errors**: Check that Git is initialized in the repository
3. **Permission Issues**: Ensure file permissions are set correctly
4. **Memory Errors**: Use batch processing for large datasets

## Contributing

1. Create a new branch for your feature
2. Write tests for your changes
3. Ensure all tests pass
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.