# CSC790 Information Retrieval
## Assignment: Homework 03 - Document Similarity

Author: Matthew Branson
Date: 6 March 2025


### Requirements

- Python 3.13.1
- NLTK 3.9.1

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

3. Prepare your text documents in a directory. By default, the program looks for a `documents` folder in the same directory as the script.

4. Ensure you have a stopwords file (`stopwords.txt`) in the same directory. Optionally, provide a special characters file (`special_chars.txt`) to remove unwanted characters from your documents.

### Running the Program

The simplest way to run the program is:

```bash
python document_similarity.py
```

This will use all default settings. After starting, the program will prompt you to enter the number of top similar document pairs (k).

### Command Line Options

You can customize the behavior using command line arguments. For example:

```bash
python document_similarity.py --documents_dir my_docs --stopwords_file my_stops.txt --special_chars_file my_special_chars.txt --index_file my_index.pkl --use_existing
```

Available options:
- `--documents_dir`: Specify the directory containing your documents (default: `documents`)
- `--stopwords_file`: Specify the path to your stopwords file (default: `stopwords.txt`)
- `--special_chars_file`: Specify the path to your special characters file (default: `special_chars.txt`)
- `--index_file`: Specify the path to save or load the index (default: `index.pkl`)
- `--use_existing`: Use an existing index if available, instead of rebuilding it