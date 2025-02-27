# CSC790 Information Retrieval
## Assignment: Homework 02 - Building a Basic Search System

Author: Matthew Branson
Date: February 26, 2025

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
python query_parser.py
```

This will use all default settings.

### Command Line Options

You can customize the behavior using command line arguments:

```bash
python query_parser.py --documents_dir my_docs --stopwords_file my_stops.txt --index_file my_index.pkl --query_file my_queries.txt --results_file my_results.txt --verbose
```

Available options:
- `--documents_dir`: Specify where your documents are located
- `--stopwords_file`: Specify your stopwords file location
- `--special_chars_file`: Specify your special characters file location
- `--index_file`: Choose where to save/load the index
- `--use_existing`: Use an existing index file instead of building a new one
- `--no_parallel`: Disable parallel processing
- `--query_file`: Specify the file containing queries
- `--results_file`: Specify where to save the search results
- `--verbose`: Include detailed scoring information in the output

### What You Should Expect To See

When you run the program, you should see output similar to the following:
```bash
2025-02-26 22:05:45,657 - INFO - =================== CSC790-IR Homework 02 ===================
2025-02-26 22:05:45,657 - INFO - First Name: Matthew
2025-02-26 22:05:45,657 - INFO - Last Name : Branson
2025-02-26 22:05:45,657 - INFO - =============================================================
2025-02-26 22:05:45,657 - INFO - Loading stopwords from file 'stopwords.txt'...
2025-02-26 22:05:45,657 - INFO - Building a new index...
2025-02-26 22:05:45,657 - INFO - Building index from documents in 'documents' using 16 workers...
2025-02-26 22:05:45,658 - INFO - Processing 17 chunks of approximately 91 files each...
2025-02-26 22:05:47,085 - INFO - All chunks processed, merging results...
2025-02-26 22:05:47,094 - INFO - Saving index to file 'index.pkl'...
2025-02-26 22:05:47,101 - INFO - Exporting index to JSON for inspection...
Press Enter to continue...
```

Since the retrieved results may well overflow your console, the program will pause and wait for you to press Enter before continuing. After pressing Enter to continue, the program will process the queries and present the lists of retrieved results, also saving the output to a results file in the same directory for inspection. You should see output similar to the following:

```bash
================== User Query 1: paper date book ==================

============= Results for: paper AND date AND book ===============
file_770.txt
file_816.txt
==================================================
============= Results for: paper AND date OR book ===============
file_770.txt
file_1055.txt
file_193.txt
...
==================================================
============= Results for: paper OR date AND book ===============
file_770.txt
file_1006.txt
file_115.txt
...
==================================================
============= Results for: paper OR date OR book ===============
file_770.txt
file_1055.txt
file_193.txt
...
==================================================
... And so on for each query in the query file.
```
