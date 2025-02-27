"""
CSC 790 Information Retrieval
Homework 02: Building a Basic Search System - Support File
Author: Matthew Branson
Date: 02/26/2025

Description:
Improving upon the InvertedIndex class implementation from Homework 1, the new index structure
now contains term frequencies per document to facilitate ranked retrieval. As well, the 
add_document method is now organized into more explicitly defined steps for text processing,
and other methods not relevant to the current assignment have been removed.
"""

import os
import logging
from collections import Counter, defaultdict
import pickle
import json
import nltk
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InvertedIndex:
    """
    A class to build, manage, and query an inverted index with term frequencies.

    Attributes:
        index (defaultdict): Nested dictionary storing term frequencies per document.
        doc_id_map (dict): Mapping from filenames to unique document IDs.
        reverse_doc_id_map (dict): Inverse mapping from document IDs to filenames.
        stopwords (set): Set of stopwords excluded from indexing.
        stemmer (PorterStemmer): Stemmer instance for term normalization.
        special_chars (set): Set of special characters to remove
    """

    def __init__(self, documents_dir=None, index_file=None, use_existing_index=False,
                 use_parallel=True, stopwords=None, special_chars=None
                 ):
        """
        Initialize the InvertedIndex. Loads existing index or builds new from documents.

        Args:
            documents_dir (str, optional): Directory containing .txt files to index.
            index_file (str, optional): Path to save/load index (pickle format).
            use_existing_index (bool): Load existing index if available.
            use_parallel (bool): Use multiprocessing for faster index construction.
            stopwords (set, optional): Custom stopwords. Defaults to empty set.
        """
        self.index = defaultdict(dict)  # {term: {doc_id: tf}}
        self.doc_id_map = {}           # {filename: doc_id}
        self.reverse_doc_id_map = {}   # {doc_id: filename}
        self.stopwords = stopwords or set()
        self.special_chars = special_chars or set()
        self.stemmer = nltk.stem.PorterStemmer()

        if use_existing_index and index_file and os.path.exists(index_file):
            self.load(index_file)
            logging.info("Existing index loaded successfully.")
        elif documents_dir:
            logging.info("Building new index...")
            if use_parallel:
                self.build_inverted_index_parallel(documents_dir)
            else:
                self.build_inverted_index(documents_dir)
            if index_file:
                self.save(index_file)
                self.export_to_json(index_file.replace('.pkl', '.json'))
                logging.info(f"Index saved to {index_file}.")

    @staticmethod
    def load_stopwords(stopwords_file):
        """
        Load stopwords from a text file (one stopword per line).

        Args:
            stopwords_file (str): Path to stopwords file.

        Returns:
            set: Lowercase stopwords. Returns empty set on error.
        """
        logging.info(f"Loading stopwords from '{stopwords_file}'...")
        stopwords = set()
        try:
            with open(stopwords_file, encoding="utf-8") as file:
                stopwords = {line.strip().lower() for line in file if line.strip()}
            logging.info(f"Loaded {len(stopwords)} stopwords.")
        except FileNotFoundError:
            logging.error(f"Stopwords file not found: {stopwords_file}")
        except Exception as e:
            logging.error(f"Error loading stopwords: {str(e)}")
        return stopwords

    def add_document(self, doc_id, text):
        """
        Process and add a document to the index following the exact workflow:
        1. Read documents
        2. Tokenize
        3. Lower case
        4. Remove punctuation
        5. Remove stopwords
        6. Stemming
        7. Create index
        
        Args:
            doc_id (int): Unique identifier for the document.
            text (str): Raw text content of the document.
        """
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        processed_tokens = []
        
        for word in tokens:
            # Lower case
            word_lower = word.lower()
            
            # Remove punctuation (apply special characters removal)
            word_cleaned = word_lower
            if self.special_chars:
                for char in self.special_chars:
                    word_cleaned = word_cleaned.replace(char, '')
            
            # Skip empty tokens after punctuation removal
            if not word_cleaned:
                continue
            
            # Remove stopwords
            if word_cleaned in self.stopwords:
                continue
            
            # Stemming
            stemmed_word = self.stemmer.stem(word_cleaned)
            processed_tokens.append(stemmed_word)
        
        term_counts = Counter(processed_tokens)
        for term, count in term_counts.items():
            self.index[term][doc_id] = count

    def save(self, filepath):
        """
        Serialize the index and metadata to a pickle file.

        Args:
            filepath (str): Destination path for the pickle file.
        """
        logging.info(f"Saving index to '{filepath}'...")
        with open(filepath, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'doc_id_map': self.doc_id_map,
                'reverse_doc_id_map': self.reverse_doc_id_map,
                'stopwords': self.stopwords
            }, f)

    def load(self, filepath):
        """
        Deserialize the index and metadata from a pickle file.

        Args:
            filepath (str): Path to the pickle file containing the index.
        """
        logging.info(f"Loading index from '{filepath}'...")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.index = data['index']
            self.doc_id_map = data['doc_id_map']
            self.reverse_doc_id_map = data['reverse_doc_id_map']
            self.stopwords = data['stopwords']

    def export_to_json(self, filepath):
        """
        Export the index to a human-readable JSON file.

        Args:
            filepath (str): Destination path for the JSON file.
        """
        logging.info(f"Exporting index to JSON at '{filepath}'...")
        export_data = {
            "metadata": {
                "document_mapping": self.doc_id_map,
                "document_count": len(self.doc_id_map)
            },
            "index": {term: dict(doc_counts) for term, doc_counts in self.index.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, sort_keys=True)

    def build_inverted_index(self, directory_path):
        """
        Build the index by sequentially processing all .txt files in a directory.

        Args:
            directory_path (str): Path to directory containing documents to index.
        """
        logging.info(f"Building index from '{directory_path}' sequentially...")
        current_doc_id = 1  # Document IDs start at 1 and increment sequentially

        try:
            files = os.listdir(directory_path)
        except FileNotFoundError:
            logging.error(f"Directory not found: {directory_path}")
            return
        except PermissionError:
            logging.error(f"Permission denied for directory: {directory_path}")
            return

        for filename in files:
            if not filename.endswith('.txt'):
                continue  # Skip non-text files

            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
                if not text:
                    logging.warning(f"Skipped empty file: {filename}")
                    continue

                # Register document and add to index
                self.doc_id_map[filename] = current_doc_id
                self.reverse_doc_id_map[current_doc_id] = filename
                self.add_document(current_doc_id, text)
                current_doc_id += 1
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")

    def build_inverted_index_parallel(self, directory_path, max_workers=None):
        """
        Build the index in parallel using multiple worker processes.

        Args:
            directory_path (str): Path to directory containing documents.
            max_workers (int, optional): Number of worker processes. Defaults to CPU count.
        """
        max_workers = max_workers or mp.cpu_count()
        logging.info(f"Parallel indexing with {max_workers} workers...")

        # Identify all .txt files and split into chunks for workers
        doc_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        chunk_size = max(1, len(doc_files) // max_workers)
        doc_chunks = [doc_files[i:i + chunk_size] 
                      for i in range(0, len(doc_files), chunk_size)]
        
        # Assign each chunk a starting doc_id to avoid overlaps
        chunk_data = [
            (chunk, 1 + i * chunk_size, directory_path, self.stopwords)
            for i, chunk in enumerate(doc_chunks)
        ]

        # Process chunks in parallel and merge results
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.__class__._process_document_chunk, chunk_data))
        
        logging.info("Merging partial indexes...")
        for partial_index in results:
            self.doc_id_map.update(partial_index['doc_id_map'])
            self.reverse_doc_id_map.update(partial_index['reverse_doc_id_map'])
            for term, doc_counts in partial_index['index'].items():
                self.index[term].update(doc_counts)  # Merge term counts

    @staticmethod
    def _process_document_chunk(chunk_data):
        """
        Static method to process a document chunk in a worker process.

        Args:
            chunk_data (tuple): Contains (filenames, start_id, directory_path, stopwords).

        Returns:
            dict: Partial index, document mappings, and errors encountered.
        """
        chunk, start_id, directory_path, stopwords = chunk_data
        local_index = InvertedIndex(stopwords=stopwords)
        current_id = start_id
        errors = []

        for filename in chunk:
            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
                if not text:
                    errors.append(f"{filename}: Empty file")
                    continue

                # Add document to the local index
                local_index.doc_id_map[filename] = current_id
                local_index.reverse_doc_id_map[current_id] = filename
                local_index.add_document(current_id, text)
                current_id += 1
            except Exception as e:
                errors.append(f"{filename}: {str(e)}")

        return {
            'index': local_index.index,
            'doc_id_map': local_index.doc_id_map,
            'reverse_doc_id_map': local_index.reverse_doc_id_map,
            'errors': errors
        }