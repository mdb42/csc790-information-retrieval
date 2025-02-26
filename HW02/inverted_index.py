"""
CSC 790 Information Retrieval
Homework 02: Building a Basic Search System - Support File
Author: Matthew Branson
Date: 02/26/2025
"""

import os
import logging
from collections import Counter, defaultdict
import pickle
import json
import nltk
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        self.index = defaultdict(dict)  # {term: {doc_id: tf}}
        self.doc_id_map = {}           # {filename: doc_id}
        self.reverse_doc_id_map = {}   # {doc_id: filename}
        self.stopwords = stopwords or set()
        self.stemmer = nltk.stem.PorterStemmer()
        self.collection_term_frequency = Counter()

        if use_existing_index and index_file and os.path.exists(index_file):
            logging.info("Loading existing index from file...")
            self.load(index_file)
        elif documents_dir:
            logging.info("Building a new index...")
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
        logging.info(f"Loading stopwords from file '{stopwords_file}'...")
        stopwords = set()
        try:
            with open(stopwords_file, encoding="utf-8") as file:
                stopwords = {line.strip().lower() for line in file if line.strip()}
        except FileNotFoundError:
            logging.error(f"The file {stopwords_file} was not found.")
        except IOError as e:
            logging.error(f"An I/O error occurred while reading {stopwords_file}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
        return stopwords

    def add_document(self, doc_id, text):
        # Tokenize
        tokens = nltk.word_tokenize(text)
        processed_tokens = []
        for word in tokens:
            # Lowercase
            word_lower = word.lower()
            # Remove punctuation (keep only alphabetic)
            if not word.isalpha():
                continue
            # Remove stopwords
            if word_lower in self.stopwords:
                continue
            # Stemming
            stemmed_word = self.stemmer.stem(word_lower)
            processed_tokens.append(stemmed_word)

        # Count term frequencies in the document
        term_counts = Counter(processed_tokens)
        
        # Update the index with term frequencies
        for term, count in term_counts.items():
            # Frequencies per document
            self.index[term][doc_id] = count
            # Frequencies across the collection
            self.collection_term_frequency[term] += count

    def save(self, filepath):
        """
        Saves the inverted index to a file using pickle serialization.
        
        Args:
            filepath (str): The path to save the index file.
        """
        logging.info(f"Saving index to file '{filepath}'...")
        with open(filepath, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'doc_id_map': self.doc_id_map,
                'reverse_doc_id_map': self.reverse_doc_id_map,
                'stopwords': self.stopwords,
                'collection_term_frequency': self.collection_term_frequency
            }, f)

    def load(self, filepath):
        """
        Loads the inverted index from a file using pickle deserialization.

        Args:
            filepath (str): The path to load the index file from.
        """
        logging.info("Loading index from file...")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.index = data['index']
            self.doc_id_map = data['doc_id_map']
            self.reverse_doc_id_map = data['reverse_doc_id_map']
            self.stopwords = data['stopwords']
            self.collection_term_frequency = data.get('collection_term_frequency', Counter())

    def export_to_json(self, filepath):
        logging.info("Exporting index to JSON for inspection...")
        export_data = {
            "metadata": {
                "document_mapping": self.doc_id_map,
                "collection_term_frequency": dict(self.collection_term_frequency)
            },
            "index": {term: dict(doc_counts) for term, doc_counts in self.index.items()}
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
        logging.info(f"Building index from documents in '{directory_path}'...")
        current_doc_id = 1

        try:
            files = os.listdir(directory_path)
        except FileNotFoundError:
            logging.error(f"Directory {directory_path} not found")
            return
        except PermissionError:
            logging.error(f"Permission denied accessing {directory_path}")
            return

        for filename in files:
            if not filename.endswith('.txt'):
                continue

            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                if not text.strip():
                    logging.warning(f"Empty file {filename}")
                    continue
                self.doc_id_map[filename] = current_doc_id
                self.reverse_doc_id_map[current_doc_id] = filename
                self.add_document(doc_id=current_doc_id, text=text)
                current_doc_id += 1
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
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
        
        logging.info(f"Building index from documents in '{directory_path}' using {max_workers} workers...")

        doc_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]

        # Determine chunk size based on the number of files and available workers.
        chunk_size = max(1, len(doc_files) // max_workers)
        doc_chunks = [doc_files[i:i + chunk_size]
                      for i in range(0, len(doc_files), chunk_size)]
        chunk_data = [
            (chunk, 1 + i * chunk_size, directory_path, self.stopwords)
            for i, chunk in enumerate(doc_chunks)
        ]

        logging.info(f"Processing {len(doc_chunks)} chunks of approximately {chunk_size} files each...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.__class__._process_document_chunk, chunk_data))
        logging.info("All chunks processed, merging results...")
        for partial_index in results:
            self.doc_id_map.update(partial_index['doc_id_map'])
            self.reverse_doc_id_map.update(partial_index['reverse_doc_id_map'])
            for term, doc_counts in partial_index['index'].items():
                self.index[term].update(doc_counts)
            for term, count in partial_index['collection_term_frequency'].items():
                self.collection_term_frequency[term] += count
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
            'collection_term_frequency': local_index.collection_term_frequency,
            'errors': errors
        }
