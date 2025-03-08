# src/index/memory_index.py
import pickle
from collections import defaultdict
from threading import Lock
from src.index.base_index import BaseIndex

class MemoryIndex(BaseIndex):
    def __init__(self):
        self.term_doc_freqs = defaultdict(dict)
        self.doc_term_freqs = defaultdict(dict)
        self.filenames = []
        self._doc_count = 0
        self._lock = Lock()  # Thread safety

    @property
    def doc_count(self) -> int:
        return self._doc_count

    def add_document(self, term_freqs: dict, filename: str = None) -> int:
        with self._lock:
            doc_id = self._doc_count  # Auto-incrementing ID
            self.doc_term_freqs[doc_id] = term_freqs
            if filename:
                self.filenames.append(filename)
            for term, freq in term_freqs.items():
                self.term_doc_freqs[term][doc_id] = freq
            self._doc_count += 1
            return doc_id
    
    @property
    def vocab_size(self) -> int:
        """Number of unique terms in the index."""
        return len(self.term_doc_freqs)

    def get_most_frequent_terms(self, n: int = 10) -> list[tuple[str, int]]:
        """Returns top-n terms by total frequency across all documents."""
        term_totals = {}
        for term, doc_freqs in self.term_doc_freqs.items():
            term_totals[term] = sum(doc_freqs.values())
        return sorted(term_totals.items(), key=lambda x: x[1], reverse=True)[:n]

    def get_term_freq(self, term: str, doc_id: int) -> int:
        return self.doc_term_freqs.get(doc_id, {}).get(term, 0)

    def get_doc_freq(self, term: str) -> int:
        return len(self.term_doc_freqs.get(term, {}))

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'term_doc_freqs': dict(self.term_doc_freqs),
                'doc_term_freqs': dict(self.doc_term_freqs),
                'filenames': self.filenames
            }, f)

    @classmethod
    def load(cls, filepath: str):
        index = MemoryIndex()
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            index.term_doc_freqs = defaultdict(dict, data['term_doc_freqs'])
            index.doc_term_freqs = defaultdict(dict, data['doc_term_freqs'])
            index.filenames = data['filenames']
            index._doc_count = len(index.doc_term_freqs)  # Fix: Assign to _doc_count
        return index