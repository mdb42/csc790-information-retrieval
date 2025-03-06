# CSC790 Information Retrieval
## Assignment: Homework 03 - Document Similarity

Author: Matthew Branson
Date: 6 March 2025

Total similarity computation time: 140.5059 seconds

### Requirements

Python 3.13.1
NLTK 3.9.1

### Setup

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
pip install -r requirements.txt
```

3. Prepare your text documents in a directory. By default, the program looks for a 'documents' folder in the same directory as the script.

4. Ensure you have a stopwords file. By default, the program looks for 'stopwords.txt' in the same directory.

5. Create a file with your queries. By default, the program looks for 'queries.txt' with one query per line.

### Running the Program

The simplest way to run the program is:

```bash
python document_similarity.py
```

This will use all default settings.

### Command Line Options

You can customize the behavior using command line arguments:

```bash
python document_similarity.py --documents_dir my_docs --stopwords_file my_stops.txt --special_chars_file my_special_chars.txt
```

Available options:
- `--documents_dir`: Specify where your documents are located
- `--stopwords_file`: Specify your stopwords file location
- `--special_chars_file`: Specify your special characters file location