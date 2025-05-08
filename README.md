# Fake News Classification MLOps System

A comprehensive MLOps platform for classifying fake news articles using data versioning, preprocessing, and machine learning.

## Overview

This project implements a complete MLOps system for fake news detection, with a focus on data management and versioning. It provides tools for dataset management, text preprocessing, feature extraction, model training, and evaluation within a web-based interface.

## Features

- **Data Version Control**: Track and version datasets using DVC (Data Version Control)
- **Dataset Management**: Upload, download, and manage news datasets
- **Text Preprocessing**: Clean and normalize news articles
- **Feature Extraction**: Generate TF-IDF and word embedding features
- **Model Training**: Train various classification models (SVM, Naive Bayes, LSTM, Transformers)
- **Evaluation**: Evaluate model performance with metrics and visualizations
- **Web Interface**: User-friendly UI for all operations

## Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL database
- Git (for DVC)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/fake-news-classification.git
   cd fake-news-classification
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   export DATABASE_URL=postgresql://user:password@localhost/dbname
   export FLASK_SECRET_KEY=your_secret_key
   ```

4. Initialize DVC:
   ```
   dvc init
   ```

5. Start the application:
   ```
   gunicorn --bind 0.0.0.0:5000 main:app
   ```

### Quick Start

1. Access the web interface at http://localhost:5000
2. Upload a dataset or use the "Setup Test Data" button
3. Preprocess the dataset with various text cleaning options
4. Extract features for machine learning
5. Train and evaluate classification models

## Project Structure

For a detailed explanation of the project structure, see [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

## Data Management

### Data Sources

This system works with CSV, JSON, and text files containing news articles. Each dataset should have at least two columns:
- Text column: The news article content
- Label column: The classification (fake/real)

### Data Versioning

Datasets are versioned using DVC, which enables:
- Tracking changes to datasets
- Rolling back to previous versions
- Sharing datasets via remote storage
- Reproducible experiments

## Text Preprocessing

The system offers various text preprocessing options:
- Lowercase conversion
- Punctuation removal
- Number removal
- Stopword removal
- Lemmatization
- Stemming

## Feature Extraction

Features can be extracted using:
- TF-IDF Vectorization
- Word Embeddings

## Model Training

The following models are supported:
- Support Vector Machines (SVM)
- Naive Bayes
- Logistic Regression
- Random Forest
- LSTM Networks
- Transformer Models

## Evaluation

Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Feature Importance

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [DVC](https://dvc.org/)
- [scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)
- [Flask](https://flask.palletsprojects.com/)
- [SQLAlchemy](https://www.sqlalchemy.org/)