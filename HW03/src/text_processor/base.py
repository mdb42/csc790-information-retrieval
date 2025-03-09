# src/text_processor/base.py
from abc import ABC, abstractmethod

class BaseTextProcessor(ABC):
    def __init__(self, documents_dir, stopwords_file=None, special_chars_file=None, profiler=None):
        self.documents_dir = documents_dir
        self.profiler = profiler
    
    @abstractmethod
    def process_documents(self):
        pass
    
    @abstractmethod
    def _preprocess_text(self, text):
        pass
    
    @abstractmethod
    def _process_single_file(self, filepath):
        pass