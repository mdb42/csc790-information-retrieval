# src/index/base.py
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional

class BaseIndex(ABC):
    def __init__(self, documents_dir=None, stopwords_file=None, special_chars_file=None, profiler=None):
        self.documents_dir = documents_dir
        self.stopwords_file = stopwords_file
        self.special_chars_file = special_chars_file
        self.profiler = profiler
    
    @property
    @abstractmethod
    def doc_count(self) -> int: ...
    
    @property
    @abstractmethod
    def vocab_size(self) -> int: ...
    
    @abstractmethod
    def build_index(self): ...
    
    @abstractmethod
    def add_document(self, term_freqs: dict, filename: str = None) -> int: ...
    
    @abstractmethod
    def _preprocess_text(self, text: str) -> List[str]: ...
    
    @abstractmethod
    def _process_single_file(self, filepath: str) -> Optional[Tuple[str, Dict[str, int]]]: ...
    
    @abstractmethod
    def get_term_freq(self, term: str, doc_id: int) -> int: ...
    
    @abstractmethod
    def get_doc_freq(self, term: str) -> int: ...
    
    @abstractmethod
    def get_most_frequent_terms(self, n: int = 10) -> List[Tuple[str, int]]: ...
    
    @abstractmethod
    def save(self, filepath: str) -> None: ...
    
    @classmethod
    @abstractmethod
    def load(cls, filepath: str): ...
    
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, int]: ...
    
    @abstractmethod
    def export_json(self, filepath: str = None) -> Optional[str]: ...
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]: ...