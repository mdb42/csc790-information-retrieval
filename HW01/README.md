# CSC790 Information Retrieval
## Assignment: Homework 01

Author: Matthew Branson
Date: 2025-01-31

### Setup

1. Place your text documents in a directory. By default, the program looks for a 'documents' folder in the same directory as the script.

2. Ensure you have a stopwords file. By default, the program looks for 'stopwords.txt' in the same directory as the script.

### Running the Program

The simplest way to run the program is to open a terminal or command prompt in the project directory and type:

python document_indexer.py

This will use all default settings: looking for documents in a 'documents' folder, using 'stopwords.txt' for stopwords, and saving the index as 'index.pkl'.

### Command Line Options

You can customize the behavior using command line arguments:

python document_indexer.py --documents_dir my_docs --stopwords_file my_stops.txt --index_file my_index.pkl

Available options:
- `--documents_dir`: Specify where your documents are located
- `--stopwords_file`: Specify your stopwords file location
- `--index_file`: Choose where to save the index
- `--use_existing`: Use an existing index file instead of building a new one
- `--no_parallel`: Disable parallel processing
- `--top_n`: Number of most frequent terms to display (default: 10)