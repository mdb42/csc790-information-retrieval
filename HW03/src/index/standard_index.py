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
        
        self.stopwords = self._load_stopwords(stopwords_file)
        self.special_chars = self._load_special_chars(special_chars_file)
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        self.stemmer = PorterStemmer()
        
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
        tokens = word_tokenize(text.lower())

        if self.special_chars:
            pattern = re.compile(f'[{re.escape("".join(self.special_chars))}]')
            tokens = [pattern.sub('', t) for t in tokens]

        tokens = [t for t in tokens if t.isalpha() and t not in self.stopwords]
        tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens

    def _process_single_file(self, filepath: str) -> Optional[Tuple[str, Dict[str, int]]]:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            if not text:
                return None

            tokens = self._preprocess_text(text)
            if not tokens:
                return None

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
        
        filepaths = [os.path.join(self.documents_dir, f) 
                    for f in os.listdir(self.documents_dir) 
                    if f.endswith('.txt')]
        
        if self.profiler:
            with self.profiler.timer(timer_label):
                results = [self._process_single_file(fp) for fp in filepaths]
        else:
            results = [self._process_single_file(fp) for fp in filepaths]
        
        results = [r for r in results if r]
        
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
            doc_id = self._doc_count
            self.doc_term_freqs[doc_id] = term_freqs
            if filename:
                self.filenames.append(filename)
            for term, freq in term_freqs.items():
                self.term_doc_freqs[term][doc_id] = freq
            self._doc_count += 1
            return doc_id
    
    def get_term_freq(self, term: str, doc_id: int) -> int:
        return self.doc_term_freqs.get(doc_id, {}).get(term, 0)

    def get_doc_freq(self, term: str) -> int:
        return len(self.term_doc_freqs.get(term, {}))
    
    def get_most_frequent_terms(self, n: int = 10) -> List[Tuple[str, int]]:
        term_totals = {}
        for term, doc_freqs in self.term_doc_freqs.items():
            term_totals[term] = sum(doc_freqs.values())
        return sorted(term_totals.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def save(self, filepath: str) -> None:
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
            index.term_doc_freqs = defaultdict(dict, data['term_doc_freqs'])
            index.doc_term_freqs = defaultdict(dict, data['doc_term_freqs'])
            index.filenames = data['filenames']
            index._doc_count = len(index.doc_term_freqs)
        return index
    
    def get_memory_usage(self) -> Dict[str, int]:
        seen = set()
        
        def sizeof(obj):
            if id(obj) in seen:
                return 0
            seen.add(id(obj))
            size = getsizeof(obj)
            if isinstance(obj, dict):
                size += sum(sizeof(k) + sizeof(v) for k, v in obj.items())
            elif isinstance(obj, (list, tuple, set)):
                size += sum(sizeof(x) for x in obj)
            return size
        
        term_doc_size = sizeof(self.term_doc_freqs)
        doc_term_size = sizeof(self.doc_term_freqs)
        filenames_size = sizeof(self.filenames)
        total_size = term_doc_size + doc_term_size + filenames_size
        
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
        export_data = {
            "term_doc_freqs": {term: dict(docs) for term, docs in self.term_doc_freqs.items()},
            "document_count": self.doc_count,
            "vocabulary_size": self.vocab_size,
            "top_terms": self.get_most_frequent_terms(20),
            "filenames": self.filenames
        }
        
        json_str = json.dumps(export_data, indent=2)
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
            return None
        else:
            return json_str
    
    def get_statistics(self) -> Dict[str, Any]:
        doc_lengths = [sum(terms.values()) for terms in self.doc_term_freqs.values()]
        
        avg_doc_length = sum(doc_lengths) / max(1, len(doc_lengths))
        
        term_counts = []
        for term, docs in self.term_doc_freqs.items():
            term_counts.append(sum(docs.values()))
        
        if term_counts:
            avg_term_freq = sum(term_counts) / len(term_counts)
            max_term_freq = max(term_counts)
            min_term_freq = min(term_counts)
        else:
            avg_term_freq = max_term_freq = min_term_freq = 0
        
        doc_freqs = [len(docs) for docs in self.term_doc_freqs.values()]
        
        if doc_freqs:
            avg_doc_freq = sum(doc_freqs) / len(doc_freqs)
            max_doc_freq = max(doc_freqs)
            min_doc_freq = min(doc_freqs)
        else:
            avg_doc_freq = max_doc_freq = min_doc_freq = 0
        
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