# CSC 790 Information Retrieval

Homework 4: To Be Determined

Author: Matthew Branson

## Overview

This is just preparation for the next homework assignment, expecting it will be similar to Homework 2 except using BM25 ranking.

## Installation

### Requirements

#### Core Requirements
- Python 3.9+ (Developed and tested with Python 3.13.1)
- NLTK 3.9.1

### Setup

1. Create and activate a virtual environment:

```bash
# Create a new virtual environment
python -m venv venv

# Activate the virtual environment
# On Unix/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download required NLTK data:

```bash
python -c "import nltk; nltk.download('punkt')"
# This will otherwise be done automatically on first run
```

## Usage

### Basic Usage

Assuming your query file, documents directory, stopwords file, and special characters file are all in the project directory, you can navigate to the project directory and run the system with:

```bash
python main.py
```

### Command Line Options

- `--config PATH`: Path to configuration file (default: "config.json")
- `--queries_file PATH`: File containing queries to search (default: "queries.txt")
- `--documents_dir PATH`: Directory containing documents to index (default: "documents")
- `--stopwords_file PATH`: File containing stopwords to remove (default: "stopwords.txt")
- `--special_chars_file PATH`: File containing special characters to remove (default: "special_chars.txt")
- `--index_file PATH`: Path to save/load the index file (default: "index.pkl")
- `--use_existing`: Use existing index if available, instead of rebuilding
- `--export_json PATH`: Export index to a JSON file for human-readable inspection
- `--index_mode MODE`: Index implementation to use (choices: "auto", "standard", "parallel", default: "auto")
- `--parallel_index_threshold N`: Document threshold for parallel index (default: 5000)
- `--stats`: Display detailed index statistics

### Examples

Provide custom query file, documents directory, stopwords file, and special characters file:
```bash
python main.py --queries_file "path/to/queries.txt" --documents_dir "path/to/documents" --stopwords_file "path/to/stopwords.txt" --special_chars_file "path/to/special_chars.txt"
```

Force a index parallel implementation:
```bash
python main.py --index_mode parallel
```

Export index to JSON for inspection:
```bash
python main.py --use_existing --export_json index_data.json
```

