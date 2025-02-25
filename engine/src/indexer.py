import os
import argparse
from sys import getsizeof
from collections import Counter, defaultdict
from collections.abc import Iterable
import pickle
import copy
import json
import nltk
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
# from query_parser import query_parser_demo

#################################################################
# Utility Functions
#################################################################

def display_banner():
    """
    Displays a banner with the course information and my name.
    """
    print("=================== CSC790-IR Homework 01 ===================")
    print("First Name: Matthew")
    print("Last Name : Branson")
    print("=============================================================")

def initialize_nltk():
    """
    Initializes the NLTK library by ensuring the 'punkt' tokenizer is available.
    
    This function checks if the 'punkt' tokenizer is already downloaded. If it is not found,
    it downloads the 'punkt' tokenizer from the NLTK data repository.

    Raises:
        LookupError: If the 'punkt' tokenizer is not found and cannot be downloaded.
    """
    try:
        nltk.data.find('tokenizers/punkt/english.pickle')
    except LookupError:
        nltk.download('punkt')

def display_info(inverted_index, top_n):
    """
    Displays memory usage breakdown and top N frequent terms of the inverted index.

        Args:
            inverted_index (InvertedIndex): The inverted index object.
            top_n (int): The number of most frequent terms to display.
        
        Raises:
            AttributeError: If the inverted_index object does not have the required methods
                or attributes.
    """
    print("=============================================================")
    print("[+] MEMORY USAGE BREAKDOWN")
    print("-------------------------------------------------------------")
    for key, value in inverted_index.get_memory_usage().items():
        print(f"{key.ljust(30)} {value:>10} bytes  ({value / 1024**2:.2f} MB)")
    print("=============================================================")
    print(f"[+] TOP {top_n} FREQUENT TERMS")
    print("-------------------------------------------------------------")
    for term, freq in inverted_index.get_top_n_terms(top_n):
        print(f"{term.ljust(20)} {freq}")

#################################################################
# The InvertedIndex Class
#################################################################

class InvertedIndex:
    """
    A class to represent an inverted index for a collection of text documents.

    Attributes:
        index (dict): A dictionary of terms mapped to the set of document IDs in which they appear.
        doc_id_map (dict): A dictionary mapping document filenames to unique document IDs.
        reverse_doc_id_map (dict): A dictionary mapping document IDs to their respective filenames.
        stopwords (set): A set of stopwords to exclude from the index.
        stemmer (nltk.stem.PorterStemmer): A stemmer object to reduce terms to their root form.
        term_frequency (collections.Counter): A counter object to store the frequency of terms.
    """
    def __init__(self, documents_dir=None, index_file=None, use_existing_index=False,
                 use_parallel=True, stopwords=None):
        """
        Initializes an instance of the InvertedIndex class.
        
        Args:
            documents_dir (str): The directory containing the text documents to index.
            index_file (str): The file path to save/load the index.
            use_existing_index (bool): Whether to use an existing index file if available.
            use_parallel (bool): Whether to use parallel processing for indexing.
            stopwords (set): A set of stopwords to exclude from the index.

        Raises:
            FileNotFoundError: If the documents directory or index file is not found.
            IOError: If an I/O error occurs while reading the stopwords file.
            Exception: If an unexpected error occurs during initialization
        """
        self.index = defaultdict(set)  # {term: set(doc_ids)}
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
    def load_stopwords(stopwords_file):
        """
        Loads a set of stopwords from a file.

        Args:
            stopwords_file (str): The path to the stopwords file.

        Returns:
            set: A set of stopwords loaded from the file.

        Raises:
            FileNotFoundError: If the stopwords file is not found.
            IOError: If an I/O error occurs while reading the stopwords file.
            Exception: If an unexpected error occurs during loading.
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
        """
        Adds a document to the inverted index.
        
        Args:
            doc_id (int): The unique identifier of the document.
            text (str): The text content of the document.
            
        Raises:
            ValueError: If the document ID is already present in the index.
        """
        tokens = [self.stemmer.stem(word.lower())
                  for word in nltk.word_tokenize(text)
                  if word.isalpha() and word.lower() not in self.stopwords]
        for term in set(tokens):
            self.index[term].add(doc_id)
        self.term_frequency.update(tokens)

    def get_memory_usage(self):
        """
        Calculates the memory usage of the inverted index components.

        Returns:
            dict: A dictionary containing the memory usage of each component.
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
        pickled_size = len(pickle.dumps(copy.deepcopy(self.index)))

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
        """
        Retrieves the top N most frequent terms in the collection.

        Args:
            n (int): The number of most frequent terms to retrieve.

        Returns:
            list: A list of tuples containing the most frequent terms and their frequencies.
        """
        return self.term_frequency.most_common(n)

    def save(self, filepath):
        """
        Saves the inverted index to a file using pickle serialization.
        
        Args:
            filepath (str): The path to save the index file.
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
        """
        Loads the inverted index from a file using pickle deserialization.

        Args:
            filepath (str): The path to load the index file from.
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
        """
        Exports the inverted index to a JSON file for inspection.

        Args:
            filepath (str): The path to save the JSON file.
        """
        print("[+] Exporting index to JSON for inspection...")
        export_data = {
            "metadata": {
                "document_mapping": self.doc_id_map
            },
            "term_frequencies": dict(self.term_frequency),
            "index": {term: sorted(list(doc_ids))
                      for term, doc_ids in self.index.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, sort_keys=True)

    def build_inverted_index(self, directory_path):
        """
        Builds an inverted index from a collection of text documents in a directory.

        Args:
            directory_path (str): The path to the directory containing text documents.

        Raises:
            FileNotFoundError: If the directory is not found.
            PermissionError: If permission is denied while accessing the directory.
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
        """
        Builds an inverted index from a collection of text documents in a directory using parallel processing.

        Args:
            directory_path (str): The path to the directory containing text documents.
            max_workers (int): The maximum number of worker processes to use for parallel processing.
                If None, the number of workers is set to the number of available CPU cores.

        Raises:
            FileNotFoundError: If the directory is not found.
            PermissionError: If permission is denied while accessing the directory.
        """
        if max_workers is None:
            max_workers = mp.cpu_count()
        
        print(f"[+] Building index from documents in '{directory_path}' using {max_workers} workers...")

        doc_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]

        # Determine chunk size based on the number of files and available workers.
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
            for term, doc_ids in partial_index['index'].items():
                self.index[term].update(doc_ids)
            for term, freq in partial_index['term_frequency'].items():
                self.term_frequency[term] += freq
            if partial_index['errors']:
                for error in partial_index['errors']:
                    print(f"[!] Error in parallel chunk: {error}")

    @staticmethod
    def _process_document_chunk(chunk_data):
        """
        Processes a chunk of documents in parallel to build an inverted index.

        Args:
            chunk_data (tuple): A tuple containing the chunk of filenames, starting document ID,
                directory path, and stopwords set.

        Returns:
            dict: A dictionary containing the partial inverted index, document ID mapping,
                reverse document ID mapping, term frequency, and errors encountered.
        """
        chunk, start_id, directory_path, stopwords = chunk_data
        # Create a new (temporary) instance to accumulate results.
        local_index = InvertedIndex(stopwords=stopwords)
        current_id = start_id
        errors = []  # Aggregate errors in this chunk
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
                continue
        return {
            'index': local_index.index,
            'doc_id_map': local_index.doc_id_map,
            'reverse_doc_id_map': local_index.reverse_doc_id_map,
            'term_frequency': local_index.term_frequency,
            'errors': errors
        }

#################################################################
# Main Function - Orchestrating the Indexing + Querying
#################################################################

def main(documents_dir=None, stopwords_file=None,
         index_file=None, use_existing_index=False, use_parallel=True, top_n=10):
    """
    Main function to build and query an inverted index from text documents.

    Args:
        documents_dir (str): The directory containing documents to index.
        stopwords_file (str): The file containing stopwords.
        index_file (str): The path to save/load the index.
        use_existing_index (bool): Whether to use an existing index if available.
        use_parallel (bool): Whether to use parallel processing for indexing.
        top_n (int): Number of most frequent terms to display.

    Raises:
        AttributeError: If the arguments are not provided and the script is run interactively.
    """
    if documents_dir is None or stopwords_file is None or index_file is None: 
        parser = argparse.ArgumentParser(
            description='Build and query an inverted index from text documents.')
        parser.add_argument('--documents_dir', default='documents',
                          help='Directory containing documents to index')
        parser.add_argument('--stopwords_file', default='stopwords.txt',
                          help='File containing stopwords')
        parser.add_argument('--index_file', default='index.pkl',
                          help='Path to save/load the index')
        parser.add_argument('--no_parallel', action='store_true',
                          help='Disable parallel processing')
        parser.add_argument('--use_existing', action='store_true',
                          help='Use existing index if available')
        parser.add_argument('--top_n', type=int, default=10,
                          help='Number of most frequent terms to display')
        
        args = parser.parse_args()
        documents_dir = args.documents_dir
        stopwords_file = args.stopwords_file
        index_file = args.index_file
        use_existing_index = args.use_existing
        use_parallel = not args.no_parallel
        top_n = args.top_n

    initialize_nltk()
    display_banner()
    stopwords = InvertedIndex.load_stopwords(stopwords_file)
    index = InvertedIndex(documents_dir=documents_dir, index_file=index_file,
                          use_existing_index=use_existing_index, use_parallel=use_parallel,
                          stopwords=stopwords)
    display_info(index, top_n)
    # query_parser_demo(index)

if __name__ == "__main__":
    main()
