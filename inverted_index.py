"""
CSC790 Information Retrieval
Homework 01 - Inverted Index
Matthew Branson
5 February 2025
"""

import os
import argparse
from sys import getsizeof
from collections import Counter, defaultdict
from collections.abc import Iterable
import pickle
import json
import nltk
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from query_parser import query_parser_demo
import traceback

###############################################################################
# Utility Functions
###############################################################################

def initialize_nltk():
    """Ensure that the NLTK library is properly initialized.
    """
    try:
        nltk.data.find('tokenizers/punkt/english.pickle')
    except LookupError:
        nltk.download('punkt')

def display_banner():
    """Display the required banner for the program."""
    print("=================== CSC790-IR Homework 01 ===================")
    print("First Name: Matthew")
    print("Last Name : Branson")
    print("=============================================================")

def display_memory_report(inverted_index):
    """Display the component-wise memory usage of the inverted index in bytes
    and megabytes.
    
    Args:
        inverted_index (InvertedIndex): The inverted index object.
    """
    print("=============================================================")
    print("[+] MEMORY USAGE REPORT")
    print("-------------------------------------------------------------")
    for key, value in inverted_index.get_memory_usage().items():
       print(f"{key.ljust(30)} {value:>15,} bytes  ({value / 1024**2:.2f} MB)")

def display_top_n_terms(inverted_index, top_n):
    """Display the top N most frequent terms in the inverted index.

    Args:
        inverted_index (InvertedIndex): The inverted index object.
        top_n (int): The number of most frequent terms to display.
    """
    print("=============================================================")
    print(f"[+] TOP {top_n} FREQUENT TERMS")
    print("-------------------------------------------------------------")
    for term, freq in inverted_index.get_top_n_terms(top_n):
        print(f"{term.ljust(20)} {freq}")

###############################################################################
# The InvertedIndex Class
###############################################################################

class InvertedIndex:
    """An inverted index for a collection of text documents.
    
    Attributes:
        index (dict): The inverted index mapping terms to document frequencies.
        doc_id_map (dict): A mapping of document filenames to unique IDs.
        reverse_doc_id_map (dict): A mapping of document IDs to filenames.
        stopwords (set): A set of stopwords to exclude from the index.
        stemmer (nltk.stem.PorterStemmer): A stemming object from NLTK.
        term_frequency (Counter): A counter of term frequencies in the index.
    """
    def __init__(self, documents_dir=None, index_file=None, use_existing_index=False,
                 use_parallel=True, stopwords=None):
        """Initialize the inverted index.
        
        Args:
            documents_dir (str): The directory containing the text documents.
            index_file (str): The file to save/load the index from.
            use_existing_index (bool): Whether to use an existing index file.
            use_parallel (bool): Whether to use parallel processing for indexing.
            stopwords (set): A set of stopwords to exclude from the index.
        """
        self.index = defaultdict(self.nested_counter)  # {term: {doc_id: count}}
        self.doc_id_map = {}           # {filename: doc_id}
        self.reverse_doc_id_map = {}   # {doc_id: filename}
        self.stopwords = stopwords or set()
        self.stemmer = nltk.stem.PorterStemmer()
        self.term_frequency = Counter()

        if use_existing_index and index_file and os.path.exists(index_file):
            print("[+] Loading existing index from file...")
            self.load(index_file)
        elif documents_dir:
            print("[+] Building a new index...")
            if use_parallel:
                self.build_inverted_index_parallel(documents_dir)
            else:
                self.build_inverted_index(documents_dir)
            if index_file:
                self.save(index_file)
                self.export_to_json(index_file.replace('.pkl', '.json'))

    @staticmethod
    def nested_counter():
        """Optimizes memory by returning a Counter for term-doc frequency mappings.
            Avoids defaultdict nesting overhead and speeds up merge operations."""
        return Counter()

    @staticmethod
    def load_stopwords(stopwords_file):
        """Load stopwords from a file into a set.
        
        Args:
            stopwords_file (str): The path to the stopwords file.

        Returns:
            set: A set of stopwords.
        """
        print(f"[+] Loading stopwords from file '{stopwords_file}'...")
        stopwords = set()
        try:
            with open(stopwords_file, encoding="utf-8") as file:
                stopwords = {line.strip().lower() for line in file if line.strip()}
        except FileNotFoundError:
            print(f"[!] Error: The file {stopwords_file} was not found.")
        except IOError as e:
            print(f"[!] Error: An I/O error occurred while reading {stopwords_file}: {e}")
        except Exception as e:
            print(f"[!] An unexpected error occurred: {e}")
        return stopwords

    def add_document(self, doc_id, text):
        """Tokenize, normalize, and index a document."""
        token_counts = Counter(
            self.stemmer.stem(word.lower()) 
            for word in nltk.word_tokenize(text)
            if word.isalpha() and word.lower() not in self.stopwords
        )
        self.index.update({term: {doc_id: count} for term, count in token_counts.items()})
        self.term_frequency.update(token_counts)

    def get_memory_usage(self):
        """Compute the memory usage of the inverted index components.
        
        Returns:
            dict: A dictionary of memory usage values in bytes.
        """
        seen = set()
        def sizeof(obj):
            if id(obj) in seen:
                return 0
            seen.add(id(obj))
            size = getsizeof(obj)
            if isinstance(obj, dict):
                size += sum(sizeof(k) + sizeof(v) for k, v in obj.items())
            elif isinstance(obj, (set, list, tuple)):
                size += sum(sizeof(x) for x in obj)
            elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)) and not isinstance(obj, dict):
                size += sum(sizeof(x) for x in obj)
            return size

        index_size = sizeof(self.index)
        doc_id_map_size = sizeof(self.doc_id_map)
        reverse_doc_id_map_size = sizeof(self.reverse_doc_id_map)
        stopwords_size = sizeof(self.stopwords)
        term_frequency_size = sizeof(self.term_frequency)
        total_size = index_size + doc_id_map_size + reverse_doc_id_map_size + stopwords_size + term_frequency_size
        pickled_size = len(pickle.dumps(self.index))

        return {
            "Inverted Index": index_size,
            "Document ID Map": doc_id_map_size,
            "Reverse Document ID Map": reverse_doc_id_map_size,
            "Stopwords Set": stopwords_size,
            "Term Frequency Counter": term_frequency_size,
            "Total Memory Usage": total_size,
            "Pickled (Compressed) Size": pickled_size
        }

    def get_top_n_terms(self, n=10):
        """Get the N most frequent terms in the index.

        Args:
            n (int): The number of terms to return.

        Returns:
            list: A list of tuples containing the most frequent terms and their counts.
        """
        return self.term_frequency.most_common(n)

    def save(self, filepath):
        """Save the index to a file using pickle.
        
        Args:
            filepath (str): The path to save the index to.
        """
        print(f"[+] Saving index to file '{filepath}'...")
        with open(filepath, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'doc_id_map': self.doc_id_map,
                'reverse_doc_id_map': self.reverse_doc_id_map,
                'stopwords': self.stopwords,
                'term_frequency': self.term_frequency
            }, f)

    def load(self, filepath):
        """Load the index from a file using pickle.

        Args:
            filepath (str): The path to load the index from.
        """
        print("[+] Loading index from file...")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.index = data['index']
            self.doc_id_map = data['doc_id_map']
            self.reverse_doc_id_map = data['reverse_doc_id_map']
            self.stopwords = data['stopwords']
            self.term_frequency = data['term_frequency']

    def export_to_json(self, filepath):
        """Export the index to a JSON file for inspection.

        Args:
            filepath (str): The path to save the JSON file to.
        """
        print("[+] Exporting index to JSON for inspection...")
        export_data = {
            "metadata": {
                "document_mapping": self.doc_id_map
            },
            "term_frequencies": dict(self.term_frequency),
            "index": {term: dict(doc_freqs)
                      for term, doc_freqs in self.index.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, sort_keys=True)

    def build_inverted_index(self, directory_path):
        """Build an inverted index from a directory of text documents.

        Args:
            directory_path (str): The path to the directory containing the documents.
        """
        print(f"[+] Building index from documents in '{directory_path}'...")
        current_doc_id = 1

        try:
            files = os.listdir(directory_path)
        except FileNotFoundError:
            print(f"[!] Error: Directory {directory_path} not found")
            return
        except PermissionError:
            print(f"[!] Error: Permission denied accessing {directory_path}")
            return

        for filename in files:
            if not filename.endswith('.txt'):
                continue

            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                if not text.strip():
                    print(f"[!] Warning: Empty file {filename}")
                    continue
                self.doc_id_map[filename] = current_doc_id
                self.reverse_doc_id_map[current_doc_id] = filename
                self.add_document(doc_id=current_doc_id, text=text)
                current_doc_id += 1
            except Exception as e:
                print(f"[!] Error processing {filename}: {e}")
                continue

    def build_inverted_index_parallel(self, directory_path, max_workers=None):
        """Build an inverted index from a directory of text documents using parallel processing.

        Args:
            directory_path (str): The path to the directory containing the documents.
            max_workers (int): The number of worker processes to use.
        """
        if max_workers is None:
            max_workers = mp.cpu_count()
        
        print(f"[+] Building index from documents in '{directory_path}' using {max_workers} workers...")

        doc_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        chunk_size = max(1, len(doc_files) // max_workers)
        doc_chunks = [doc_files[i:i + chunk_size]
                      for i in range(0, len(doc_files), chunk_size)]
        chunk_data = [
            (chunk, 1 + i * chunk_size, directory_path, self.stopwords)
            for i, chunk in enumerate(doc_chunks)
        ]

        print(f"[+] Processing {len(doc_chunks)} chunks of approximately {chunk_size} files each...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.__class__._process_document_chunk, chunk_data))
        print("[+] All chunks processed, merging results...")
        processed_files = 0
        for partial_index in results:
            processed_files += len(partial_index['doc_id_map'])
            self.doc_id_map.update(partial_index['doc_id_map'])
            self.reverse_doc_id_map.update(partial_index['reverse_doc_id_map'])
            for term, doc_freqs in partial_index['index'].items():
                self.index[term].update(doc_freqs)
            for term, freq in partial_index['term_frequency'].items():
                self.term_frequency[term] += freq
            if partial_index['errors']:
                for error in partial_index['errors']:
                    print(f"[!] Error in parallel chunk: {error}")

    @staticmethod
    def _process_document_chunk(chunk_data):
        """Process a chunk of documents in parallel.

        Args:
            chunk_data (tuple): A tuple containing the chunk of filenames, the starting doc_id,
                the directory path, and the stopwords set.

        Returns:
            dict: A dictionary containing the partial index, doc_id_map, reverse_doc_id_map,
                term_frequency, and any errors encountered.
        """
        chunk, start_id, directory_path, stopwords = chunk_data
        local_index = InvertedIndex(stopwords=stopwords)
        current_id = start_id
        errors = []
        for filename in chunk:
            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                local_index.doc_id_map[filename] = current_id
                local_index.reverse_doc_id_map[current_id] = filename
                local_index.add_document(current_id, text)
                current_id += 1
            except Exception as e:
                errors.append(f"{filename}: {str(e)}")
                print(f"[!] Error in file {filename}: {traceback.format_exc()}")
                continue
        return {
            'index': local_index.index,
            'doc_id_map': local_index.doc_id_map,
            'reverse_doc_id_map': local_index.reverse_doc_id_map,
            'term_frequency': local_index.term_frequency,
            'errors': errors
        }

###############################################################################
# Main Function - Orchestrates the Indexing and Querying Process
###############################################################################

def main(documents_dir=None, stopwords_file=None,
         index_file=None, use_existing_index=False, use_parallel=True, top_n=10):
    """Main function to build and query an inverted index from text documents.

    Args:
        documents_dir (str): The directory containing the text documents.
        stopwords_file (str): The file containing stopwords.
        index_file (str): The path to save/load the index.
        use_existing_index (bool): Whether to use an existing index file.
        use_parallel (bool): Whether to use parallel processing for indexing.
        top_n (int): The number of most frequent terms to display.
    """
    parser = argparse.ArgumentParser(description='Build and query an inverted index.')
    parser.add_argument('--documents_dir', default='documents', help='Directory containing documents')
    parser.add_argument('--stopwords_file', default='stopwords.txt', help='Stopwords file')
    parser.add_argument('--index_file', default='index.pkl', help='Index file path')
    parser.add_argument('--no_parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--use_existing', action='store_true', help='Use existing index if available')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top frequent terms to display')
    args = parser.parse_args()

    # Override defaults if function arguments are provided
    documents_dir = documents_dir or args.documents_dir
    stopwords_file = stopwords_file or args.stopwords_file
    index_file = index_file or args.index_file
    if use_existing_index is None:
        use_existing_index = args.use_existing
    if use_parallel is None:
        use_parallel = not args.no_parallel
    if top_n is None:
        top_n = args.top_n


    initialize_nltk() # Just in case this is your first time using nltk
    display_banner()
    stopwords = InvertedIndex.load_stopwords(stopwords_file)
    index = InvertedIndex(documents_dir=documents_dir, index_file=index_file,
                          use_existing_index=use_existing_index, use_parallel=use_parallel,
                          stopwords=stopwords)
    display_memory_report(index)
    display_top_n_terms(index, top_n)
    query_parser_demo(index) # Not required, but useful for evaluating the index

if __name__ == "__main__":
    main()
