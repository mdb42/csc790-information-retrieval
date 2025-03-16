# src/index/standard_index.py
import os
import re
import json
import pickle
import copy
from sys import getsizeof
from collections import defaultdict, Counter
from threading import Lock
from typing import List, Tuple, Dict, Any, Optional

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from src.index.base import BaseIndex


class StandardIndex(BaseIndex):
    def __init__(self, documents_dir=None, stopwords_file=None, special_chars_file=None, profiler=None):
        super().__init__(documents_dir, stopwords_file, special_chars_file, profiler)
        
        # Load stopwords and special characters
        self.stopwords = self._load_stopwords(stopwords_file)
        self.special_chars = self._load_special_chars(special_chars_file)
        
        # Compile regex pattern once for efficiency
        self.special_chars_pattern = None
        if self.special_chars:
            self.special_chars_pattern = re.compile(f'[{re.escape("".join(self.special_chars))}]')
        
        # Ensure NLTK data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        self.stemmer = PorterStemmer()
        
        # Initialize index data structures
        self.term_doc_freqs = defaultdict(dict)  # term -> {doc_id -> freq}
        self.doc_term_freqs = defaultdict(dict)  # doc_id -> {term -> freq}
        self.filenames = []                      # doc_id -> filename
        self._doc_count = 0
        self._lock = Lock()  # Thread safety

    def _load_stopwords(self, filepath):
        if not filepath:
            return set()
        try:
            with open(filepath, encoding="utf-8") as file:
                return {line.strip().lower() for line in file if line.strip()}
        except Exception as e:
            print(f"Warning: Failed to load stopwords. {e}")
            return set()

    def _load_special_chars(self, filepath):
        if not filepath:
            return set()
        try:
            with open(filepath, encoding="utf-8") as file:
                return {line.strip() for line in file if line.strip()}
        except Exception as e:
            print(f"Warning: Failed to load special characters. {e}")
            return set()
    
    def _preprocess_text(self, text: str) -> List[str]:
        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())

        # Remove special characters using pre-compiled regex
        if self.special_chars_pattern:
            tokens = [self.special_chars_pattern.sub('', t) for t in tokens]

        # Filter alphabetic tokens and remove stopwords
        tokens = [t for t in tokens if t.isalpha() and t not in self.stopwords]
        
        # Apply stemming
        tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens

    def _process_single_file(self, filepath: str) -> Optional[Tuple[str, Dict[str, int]]]:
        try:
            # Read file content with error handling for encoding issues
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            if not text:
                return None

            # Preprocess and tokenize
            tokens = self._preprocess_text(text)
            if not tokens:
                return None

            # Extract filename and count term frequencies
            filename = os.path.basename(filepath)
            term_counts = Counter(tokens)
            return filename, term_counts
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None
    
    def build_index(self):
        if not self.documents_dir:
            raise ValueError("Cannot build index: documents_dir not specified")
        
        timer_label = "Sequential Index Building"
        
        # Get list of text files using os.scandir() for better memory efficiency
        filepaths = [entry.path for entry in os.scandir(self.documents_dir)
                     if entry.is_file() and entry.name.endswith('.txt')]
        
        # Process files sequentially with timing if profiler is available
        if self.profiler:
            with self.profiler.timer(timer_label):
                results = [self._process_single_file(fp) for fp in filepaths]
        else:
            results = [self._process_single_file(fp) for fp in filepaths]
        
        # Filter out None results (failed processing)
        results = [r for r in results if r]
        
        # Add each document to the index
        for filename, term_freqs in results:
            self.add_document(term_freqs, filename)
        
        return self
    
    @property
    def doc_count(self) -> int:
        return self._doc_count
    
    @property
    def vocab_size(self) -> int:
        return len(self.term_doc_freqs)
    
    def add_document(self, term_freqs: dict, filename: str = None) -> int:
        with self._lock:
            # Assign document ID and store term frequencies
            doc_id = self._doc_count
            self.doc_term_freqs[doc_id] = term_freqs
            
            # Store filename if provided
            if filename:
                self.filenames.append(filename)
            
            # Store reference to avoid repeated lookups
            term_doc_freqs = self.term_doc_freqs
            
            # Update inverted index (term -> doc)
            for term, freq in term_freqs.items():
                term_doc_freqs[term][doc_id] = freq
            
            # Increment document counter
            self._doc_count += 1
            return doc_id
    
    def get_document_lengths(self):
        """
        Get the length of each document and the average document length.
        
        Returns:
            Tuple[Dict[int, int], float]: Document lengths dictionary and average length
        """
        doc_lengths = {
            doc_id: sum(term_freqs.values())
            for doc_id, term_freqs in self.doc_term_freqs.items()
        }
        
        avg_length = sum(doc_lengths.values()) / max(len(doc_lengths), 1)
        
        return doc_lengths, avg_length

    def get_term_freq(self, term: str, doc_id: int) -> int:
        return self.doc_term_freqs.get(doc_id, {}).get(term, 0)

    def get_doc_freq(self, term: str) -> int:
        return len(self.term_doc_freqs.get(term, {}))
    
    def get_most_frequent_terms(self, n: int = 10) -> List[Tuple[str, int]]:
        import heapq
        
        # Calculate total frequency of each term across all documents
        term_totals = {}
        for term, doc_freqs in self.term_doc_freqs.items():
            term_totals[term] = sum(doc_freqs.values())

        # Use heapq.nlargest for efficient top-n selection (O(n log k) complexity)
        return heapq.nlargest(n, term_totals.items(), key=lambda x: x[1])
    
    def save(self, filepath: str) -> None:
        """
        Save the index to a file using pickle serialization.
        
        Args:
            filepath (str): Path where the index should be saved
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'term_doc_freqs': dict(self.term_doc_freqs),
                'doc_term_freqs': dict(self.doc_term_freqs),
                'filenames': self.filenames
            }, f)
    
    @classmethod
    def load(cls, filepath: str):
        index = cls()
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            # Convert back to defaultdict for convenient access
            index.term_doc_freqs = defaultdict(dict, data['term_doc_freqs'])
            index.doc_term_freqs = defaultdict(dict, data['doc_term_freqs'])
            index.filenames = data['filenames']
            index._doc_count = len(index.doc_term_freqs)
        return index
    
    def get_memory_usage(self) -> Dict[str, int]:
        def sizeof_iterative(obj):
            """Calculate size of complex objects iteratively to avoid recursion limits."""
            seen = set()
            to_process = [obj]
            total_size = 0
            
            while to_process:
                current = to_process.pop()
                if id(current) in seen:
                    continue
                    
                seen.add(id(current))
                total_size += getsizeof(current)
                
                # Add contained objects to processing queue
                if isinstance(current, dict):
                    to_process.extend(current.keys())
                    to_process.extend(current.values())
                elif isinstance(current, (list, tuple, set)):
                    to_process.extend(current)
                    
            return total_size
        
        # Calculate sizes of main components
        term_doc_size = sizeof_iterative(self.term_doc_freqs)
        doc_term_size = sizeof_iterative(self.doc_term_freqs)
        filenames_size = sizeof_iterative(self.filenames)
        total_size = term_doc_size + doc_term_size + filenames_size
        
        # Calculate serialized size for comparison
        sample_data = {
            'term_doc_freqs': dict(self.term_doc_freqs),
            'doc_term_freqs': dict(self.doc_term_freqs),
            'filenames': self.filenames
        }
        pickled_size = len(pickle.dumps(sample_data))
        
        return {
            "Term-Doc Index": term_doc_size,
            "Doc-Term Index": doc_term_size,
            "Filenames": filenames_size,
            "Total Memory Usage": total_size,
            "Pickled Size": pickled_size
        }
    
    def export_json(self, filepath: str = None) -> Optional[str]:
        # Prepare export data - convert defaultdicts to regular dicts for serialization
        export_data = {
            "term_doc_freqs": {term: dict(docs) for term, docs in self.term_doc_freqs.items()},
            "document_count": self.doc_count,
            "vocabulary_size": self.vocab_size,
            "top_terms": self.get_most_frequent_terms(20),
            "filenames": self.filenames
        }
        
        # Convert to JSON string with pretty formatting
        json_str = json.dumps(export_data, indent=2)
        
        # Save to file if filepath is provided
        if filepath:
            if not filepath.endswith(".json"):
                filepath += ".json"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
            return None
        else:
            return json_str
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the index.
        
        Returns:
            Dict[str, Any]: Dictionary containing various statistics about the index
        """
        # Calculate document lengths (number of terms in each document)
        doc_lengths = [sum(terms.values()) for terms in self.doc_term_freqs.values()]
        
        # Calculate average document length
        avg_doc_length = sum(doc_lengths) / max(1, len(doc_lengths))
        
        # Calculate term frequency statistics
        term_counts = []
        for term, docs in self.term_doc_freqs.items():
            term_counts.append(sum(docs.values()))
        
        if term_counts:
            avg_term_freq = sum(term_counts) / len(term_counts)
            max_term_freq = max(term_counts)
            min_term_freq = min(term_counts)
        else:
            avg_term_freq = max_term_freq = min_term_freq = 0
        
        # Calculate document frequency statistics
        doc_freqs = [len(docs) for docs in self.term_doc_freqs.values()]
        
        if doc_freqs:
            avg_doc_freq = sum(doc_freqs) / len(doc_freqs)
            max_doc_freq = max(doc_freqs)
            min_doc_freq = min(doc_freqs)
        else:
            avg_doc_freq = max_doc_freq = min_doc_freq = 0
        
        # Return comprehensive statistics
        return {
            "document_count": self.doc_count,
            "vocabulary_size": self.vocab_size,
            "avg_doc_length": avg_doc_length,
            "max_doc_length": max(doc_lengths) if doc_lengths else 0,
            "min_doc_length": min(doc_lengths) if doc_lengths else 0,
            "avg_term_freq": avg_term_freq,
            "max_term_freq": max_term_freq,
            "min_term_freq": min_term_freq,
            "avg_doc_freq": avg_doc_freq,
            "max_doc_freq": max_doc_freq,
            "min_doc_freq": min_doc_freq,
            "memory_usage": self.get_memory_usage()
        }