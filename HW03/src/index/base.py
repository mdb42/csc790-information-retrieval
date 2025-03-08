# src/index/base_index.py
from abc import ABC, abstractmethod

class BaseIndex(ABC):
    @abstractmethod
    def add_document(self, doc_id: int, term_freqs: dict): ...
    
    @abstractmethod
    def get_term_freq(self, term: str, doc_id: int) -> int: ...
    
    @abstractmethod
    def get_doc_freq(self, term: str) -> int: ...
    
    @abstractmethod
    def save(self, filepath: str): ...
    
    @classmethod
    @abstractmethod
    def load(cls, filepath: str): ...
