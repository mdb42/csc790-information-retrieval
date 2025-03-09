# CSC790 Information Retrieval
## Assignment: Homework 03 - Vector Space Model

Author: Matthew Branson
Date: 9 March 2025

### Overview

This project implements a configurable and highly optimized Vector Space Model (VSM) for document similarity analysis.

### Requirements

### Core Requirements
- Python 3.9+ (Developed and tested with Python 3.13.1)
- NLTK 3.9.1
- Basic text files to process

### Optional Dependencies (for enhanced performance)
- NumPy (for vector operations)
- SciPy (for sparse matrix operations)
- scikit-learn (for optimized distance calculations)
- Multiprocessing support (for parallel processing)

## Installation

1. Create and activate a virtual environment to isolate the project dependencies:

```bash
# Create a new virtual environment
python -m venv venv

# Activate the virtual environment
# On Unix/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

2. Install the required dependencies using pip:

```bash
# Install core dependencies
pip install nltk

# Install optional dependencies for optimal performance
pip install numpy scipy scikit-learn
# Anaconda users may expect to already have these dependencies
```

3. Download required NLTK data:

```bash
python -c "import nltk; nltk.download('punkt')"
# This will otherwise be done automatically on first run
```

4. Prepare your text documents in a directory. By default, the program looks for a `documents` folder in the same directory as the script.

5. Ensure you have a stopwords file (`stopwords.txt`) in the same directory. Optionally, provide a special characters file (`special_chars.txt`) to remove unwanted characters from your documents.

## Running the Program

The simplest way to run the program is:

```bash
python main.py
```

After processing the documents and building the index, the program will prompt you to enter the number of top similar document pairs (k) to find.

### Command Line Options

#### Basic Options
```bash
python main.py --documents_dir my_docs --stopwords_file my_stops.txt --use_existing
```

#### Path and File Options
- `--documents_dir PATH`: Directory containing documents to index (default: "documents")
- `--stopwords_file PATH`: File containing stopwords to remove (default: "stopwords.txt")
- `--special_chars_file PATH`: File containing special characters to remove (default: "special_chars.txt")
- `--index_file PATH`: Path to save/load the index file (default: "index.pkl")
- `--use_existing`: Use existing index if available, instead of rebuilding
- `--export_json PATH`: Export index to a JSON file for human-readable inspection

#### Implementation Selection
- `--index_mode MODE`: Index implementation to use (choices: "auto", "standard", "parallel", default: "auto")
- `--vsm_mode MODE`: VSM implementation to use (choices: "auto", "standard", "parallel", "hybrid", "sparse", default: "auto")

#### Advanced Options
- `--parallel_index_threshold N`: Document threshold for parallel index (default: 5000)
- `--hybrid_vsm_threshold N`: Document threshold for hybrid vs parallel VSM (default: 15000)
- `--stats`: Display detailed index statistics
- `--check_deps`: Check and display available dependencies
