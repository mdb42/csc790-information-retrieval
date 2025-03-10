# src/index/factory.py
"""
Index Comparisons

| Document Count | Standard Index | Parallel Index |
| -------------- | -------------- | -------------- |
| 1000 Docs      | 0.8525         | 3.4042         |
| 2000 Docs      | 1.7143         | 3.5417         |
| 3000 Docs      | 2.5156         | 3.7468         |
| 4000 Docs      | 3.4333         | 3.8841         |
| 5000 Docs      | 4.2605         | 4.1409         |
| 15000 Docs     | 12.4163        | 5.7538         |
| 30000 Docs     | 24.9075        | 8.3567         |

"""
import os
from typing import Optional
from src.performance_monitoring import Profiler

class IndexFactory:

    DEFAULT_PARALLEL_DOC_THRESHOLD = 5000

    INDEX_CLASSES = {
        "parallel": "parallel_index.ParallelIndex",  # For large document collections with multiprocessing
        "standard": "standard_index.StandardIndex"   # Fallback option, single-threaded
    }

    @staticmethod
    def check_multiprocessing():
        try:
            import multiprocessing
            try:
                with multiprocessing.Pool(1) as p:
                    pass
                return True
            except (ImportError, OSError, ValueError):
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
        
        if documents_dir is None:
            if profiler:
                profiler.log_message("Warning: No document directory provided. Defaulting to StandardIndex.")
            from src.index.standard_index import StandardIndex
            return StandardIndex(None, stopwords_file, special_chars_file, profiler)
        
        if not os.path.exists(documents_dir):
            if profiler:
                profiler.log_message(f"Warning: Document directory '{documents_dir}' not found. Defaulting to StandardIndex.")
            from src.index.standard_index import StandardIndex
            return StandardIndex(None, stopwords_file, special_chars_file, profiler)
        
        from src.index.standard_index import StandardIndex
        
        has_multiprocessing = IndexFactory.check_multiprocessing()
        
        if doc_count is None and mode == 'auto' and documents_dir and os.path.exists(documents_dir):
            try:
                doc_count = sum(1 for f in os.scandir(documents_dir) 
                            if f.is_file() and f.name.endswith('.txt'))
                if profiler:
                    profiler.log_message(f"Counted {doc_count} documents in {documents_dir}")
            except Exception as e:
                if profiler:
                    profiler.log_message(f"Warning: Could not count files in {documents_dir}: {e}")
                return StandardIndex(documents_dir, stopwords_file, special_chars_file, profiler)
        
        original_mode = mode
        if mode == 'auto':
            if has_multiprocessing and doc_count and doc_count >= parallel_threshold:
                mode = 'parallel'
            else:
                mode = 'standard'
            
            if profiler:
                profiler.log_message(f"Auto-selected index mode: {mode} " + 
                                   f"(doc_count={doc_count}, threshold={parallel_threshold}, " +
                                   f"multiprocessing={'available' if has_multiprocessing else 'unavailable'})")
        
        if mode == 'parallel' and not has_multiprocessing:
            if profiler:
                profiler.log_message(f"Warning: Parallel index requested but multiprocessing not available. Using standard index.")
            mode = 'standard'
        
        # Log explicitly requested modes
        if profiler and original_mode != 'auto':
            profiler.log_message(f"Using explicitly requested index mode: {mode}")
        
        # Dynamic import of the selected implementation
        try:
            selected_class = IndexFactory.INDEX_CLASSES.get(mode, IndexFactory.INDEX_CLASSES['standard'])
            module_name, class_name = selected_class.rsplit(".", 1)
            module = __import__(f"src.index.{module_name}", fromlist=[class_name])
            
            # Log the final selection
            if profiler:
                profiler.log_message(f"Created index implementation: {module_name}.{class_name}")
            
            return getattr(module, class_name)(documents_dir, stopwords_file, special_chars_file, profiler)
        except ImportError as e:
            if profiler:
                profiler.log_message(f"Warning: Failed to import {mode} index implementation: {e}. Using standard index.")
            print(f"Warning: Failed to import {mode} index implementation: {e}. Using standard index.")
            from src.index.standard_index import StandardIndex
            return StandardIndex(documents_dir, stopwords_file, special_chars_file, profiler)