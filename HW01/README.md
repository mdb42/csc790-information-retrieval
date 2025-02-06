# CSC790 Information Retrieval
## Assignment: Homework 01

Author: Matthew Branson
Date: February 5, 2025

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

3. Prepare your text documents in a directory. By default, the program looks for a 'documents' folder in the same directory as the script. The documents should be plain text files with .txt extensions.

4. Ensure you have a stopwords file. By default, the program looks for 'stopwords.txt' in the same directory as the script. This file should contain one stopword per line.

### Running the Program

The simplest way to run the program is to open a terminal or command prompt in the project directory and type:

python inverted_index.py

This will use all default settings: looking for documents in a 'documents' folder, using 'stopwords.txt' for stopwords, and saving the index as 'index.pkl'.

### Command Line Options

You can customize the behavior using command line arguments:

For example, to specify a different documents directory, stopwords file, and index file, you could run:

```bash
python inverted_index.py --documents_dir my_docs --stopwords_file my_stops.txt --index_file my_index.pkl
```

Available options:
- `--documents_dir`: Specify where your documents are located
- `--stopwords_file`: Specify your stopwords file location
- `--index_file`: Choose where to save the index
- `--use_existing`: Use an existing index file instead of building a new one
- `--no_parallel`: Disable parallel processing
- `--top_n`: Number of most frequent terms to display (default: 10)

### What You Should Expect To See

When you run the program, you should see output similar to the following:

```
=================== CSC790-IR Homework 01 ===================
First Name: Matthew
Last Name : Branson
=============================================================
[+] Loading stopwords from file 'stopwords.txt'...
[+] Building a new index...
[+] Building index from documents in 'documents' using 16 workers...
[+] Processing 17 chunks of approximately 91 files each...
[+] All chunks processed, merging results...
[+] Saving index to file 'index.pkl'...
[+] Exporting index to JSON for inspection...
=============================================================
[+] MEMORY USAGE BREAKDOWN
-------------------------------------------------------------
Inverted Index                    5944207 bytes  (5.67 MB)
Document ID Map                    163332 bytes  (0.16 MB)
Reverse Document ID Map            107492 bytes  (0.10 MB)
Stopwords Set                       14110 bytes  (0.01 MB)
Term Frequency Counter             208948 bytes  (0.20 MB)
Total Memory Usage                6438089 bytes  (6.14 MB)
Pickled (Compressed) Size          276676 bytes  (0.26 MB)
=============================================================
[+] TOP 10 FREQUENT TERMS
-------------------------------------------------------------
librari              1544
inform               1355
use                  1077
system               1076
index                600
research             541
data                 536
studi                515
book                 508
develop              502

Enter a Boolean query (or type 'exit' to quit):
```

At this prompt, you can enter a simple Boolean query to search the index.
When you are finished, type `exit`.
Additionally, you may view the exported json file to examine the index directly.

### Learn More

You can read about the development process by visiting my [Project Journal](https://github.com/mdb42/csc790-information-retrieval/blob/main/journal.md).

