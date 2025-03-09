# src/index/factory.py
import os
from typing import Optional
from src.performance_monitoring import Profiler

class IndexFactory:

    DEFAULT_PARALLEL_DOC_THRESHOLD = 5000
    
    @staticmethod
    def check_multiprocessing():
        try:
            import multiprocessing
            try:
                with multiprocessing.Pool(1) as p:
                    pass
                return True
            except:
                return False
        except ImportError:
            return False
    
    @staticmethod
    def create_index(documents_dir=None, stopwords_file=None, special_chars_file=None,
                    profiler=None, mode='auto', doc_count=None, parallel_threshold=None):
        if profiler is None:
            from src.performance_monitoring import Profiler
            profiler = Profiler()
        
        if parallel_threshold is None:
            parallel_threshold = IndexFactory.DEFAULT_PARALLEL_DOC_THRESHOLD
        
        from src.index.standard_index import StandardIndex
        from src.index.parallel_index import ParallelIndex
        
        has_multiprocessing = IndexFactory.check_multiprocessing()
        
        if doc_count is None and mode == 'auto' and documents_dir:
            try:
                doc_count = len([f for f in os.listdir(documents_dir) if f.endswith('.txt')])
            except Exception as e:
                print(f"Warning: Could not count files in {documents_dir}: {e}")
                return StandardIndex(documents_dir, stopwords_file, special_chars_file, profiler)
        
        if mode == 'auto':
            if has_multiprocessing and doc_count and doc_count >= parallel_threshold:
                return ParallelIndex(documents_dir, stopwords_file, special_chars_file, profiler)
            else:
                return StandardIndex(documents_dir, stopwords_file, special_chars_file, profiler)
        elif mode == 'parallel':
            if has_multiprocessing:
                return ParallelIndex(documents_dir, stopwords_file, special_chars_file, profiler)
            else:
                print("Warning: Parallel index requested but multiprocessing not available. Using standard index.")
                return StandardIndex(documents_dir, stopwords_file, special_chars_file, profiler)
        else:
            return StandardIndex(documents_dir, stopwords_file, special_chars_file, profiler)