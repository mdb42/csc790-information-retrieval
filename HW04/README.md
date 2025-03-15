# CSC 790 Information Retrieval

Homework 4: To Be Determined

Author: Matthew Branson

## Overview

This is just preparation for the next homework assignment, assuming backward compatibility with all previous assignments and expecting that the next assignment will be similar to Homework 2 except involving BM25 probabilistic ranking.

## Installation

### Requirements

#### Core Requirements
- Python 3.9+ (Developed and tested with Python 3.13.1)
- NLTK 3.9.1

#### Optional Dependencies (for enhanced performance)
- NumPy (for vector operations)
- SciPy (for sparse matrix representation)
- scikit-learn (for optimized similarity calculations)

Note: Anaconda users should already have NumPy, SciPy, and scikit-learn installed in their environment.

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

Assuming your documents directory, stopwords file, and special characters file are all in the project directory, you can navigate to the project directory and run the system with:

```bash
python main.py
```

### Command Line Options

- `--documents_dir PATH`: Directory containing documents to index (default: "documents")
- `--stopwords_file PATH`: File containing stopwords to remove (default: "stopwords.txt")
- `--special_chars_file PATH`: File containing special characters to remove (default: "special_chars.txt")
- `--index_file PATH`: Path to save/load the index file (default: "index.pkl")
- `--top_k N`: Number of top similar documents to display (If none, user will be prompted)
- `--config PATH`: Path to configuration file (default: "config.json")
- `--use_existing`: Use existing index if available, instead of rebuilding
- `--export_json PATH`: Export index to a JSON file for human-readable inspection
- `--index_mode MODE`: Index implementation to use (choices: "auto", "standard", "parallel", default: "auto")
- `--vsm_mode MODE`: VSM implementation to use (choices: "auto", "standard", "parallel", "sparse", default: "auto")
- `--parallel_index_threshold N`: Document threshold for parallel index (default: 5000)
- `--parallel_vsm_threshold N`: Document threshold for switching to ParallelVSM (default: 2500)
- `--parallelize_weights`: Parallelize weight computation in ParallelVSM (For benchmarking only)
- `--stats`: Display detailed index statistics
- `--check_deps`: Check and display available dependencies

### Examples

Basic usage:
```bash
python main.py --documents_dir "path/to/my/documents" --stopwords_file "path/to/stopwords.txt" --special_chars_file "path/to/special_chars.txt"
```

Force a specific implementation:
```bash
python main.py --vsm_mode parallel --index_mode parallel
```

Check available dependencies before running:
```bash
python main.py --check_deps
```

Export index to JSON for inspection:
```bash
python main.py --use_existing --export_json index_data.json
```

