# src/index/factory.py
"""
Index Factory
Author: Matthew Branson
Date: March 14, 2025

This module implements a factory pattern for creating appropriate document index
implementations based on document count, available system resources, and user preferences.
"""
import os
from typing import Optional
from src.profiler import Profiler
from src.utils import check_multiprocessing


class IndexFactory:
    """
    Factory class for creating document index implementations.
    
    This factory dynamically selects and instantiates the most appropriate 
    document index implementation based on available system resources,
    document count, and user preferences.
    
    Class Attributes:
        DEFAULT_PARALLEL_DOC_THRESHOLD (int): Document count threshold for switching 
                                             from StandardIndex to ParallelIndex
        INDEX_CLASSES (dict): Mapping of implementation names to class paths
    """

    DEFAULT_PARALLEL_DOC_THRESHOLD = 5000  # Switch from Standard to Parallel at 5K docs

    INDEX_CLASSES = {
        "parallel": "parallel_index.ParallelIndex",  # For large document collections with multiprocessing
        "standard": "standard_index.StandardIndex"   # Fallback option, single-threaded
    }
    
    @staticmethod
    def create_index(documents_dir=None, stopwords_file=None, special_chars_file=None,
                    profiler=None, mode='auto', doc_count=None, parallel_threshold=None):
        """
        Create and return the appropriate document index implementation.
                
        Args:
            documents_dir (str, optional): Directory containing documents to index
            stopwords_file (str, optional): File containing stopwords to remove
            special_chars_file (str, optional): File containing special characters to remove
            profiler (Profiler, optional): Performance profiler for timing operations
            mode (str): Index implementation to use ('auto', 'standard', or 'parallel')
            doc_count (int, optional): Known document count (if available)
            parallel_threshold (int, optional): Document count threshold for using ParallelIndex
                                              over StandardIndex
            
        Returns:
            BaseIndex: An instance of the selected index implementation
            
        Note:
            If the requested implementation is unavailable, this method falls back to
            the best available alternative and logs a warning.
        """
        # Create a profiler if none was provided
        if profiler is None:
            from HW03.src.profiler import Profiler
            profiler = Profiler()
        
        # Use default threshold if none specified
        if parallel_threshold is None:
            parallel_threshold = IndexFactory.DEFAULT_PARALLEL_DOC_THRESHOLD
        
        # Handle missing document directory (none provided)
        if documents_dir is None:
            if profiler:
                profiler.log_message("Warning: No document directory provided. Defaulting to StandardIndex.")
            from src.index.standard_index import StandardIndex
            return StandardIndex(None, stopwords_file, special_chars_file, profiler)
        
        # Handle non-existent document directory (invalid path)
        if not os.path.exists(documents_dir):
            if profiler:
                profiler.log_message(f"Warning: Document directory '{documents_dir}' not found. Defaulting to StandardIndex.")
            from src.index.standard_index import StandardIndex
            return StandardIndex(None, stopwords_file, special_chars_file, profiler)
        
        # Import StandardIndex for potential use
        from src.index.standard_index import StandardIndex
        
        # Check if multiprocessing is available
        has_multiprocessing = check_multiprocessing()
        
        # Count documents if needed for auto mode and not provided
        if doc_count is None and mode == 'auto' and documents_dir and os.path.exists(documents_dir):
            try:
                # Count only .txt files in the directory
                doc_count = sum(1 for f in os.scandir(documents_dir) 
                            if f.is_file() and f.name.endswith('.txt'))
                if profiler:
                    profiler.log_message(f"Counted {doc_count} documents in {documents_dir}")
            except Exception as e:
                if profiler:
                    profiler.log_message(f"Warning: Could not count files in {documents_dir}: {e}")
                # Fall back to StandardIndex if counting fails
                return StandardIndex(documents_dir, stopwords_file, special_chars_file, profiler)
        
        # Store original mode for logging purposes
        original_mode = mode
        
        # Auto-select mode based on document count and multiprocessing availability
        if mode == 'auto':
            if has_multiprocessing and doc_count and doc_count >= parallel_threshold:
                mode = 'parallel'
            else:
                mode = 'standard'
            
            if profiler:
                profiler.log_message(f"Auto-selected index mode: {mode} " + 
                                   f"(doc_count={doc_count}, threshold={parallel_threshold}, " +
                                   f"multiprocessing={'available' if has_multiprocessing else 'unavailable'})")
        
        # Handle case where parallel was requested but multiprocessing isn't available
        if mode == 'parallel' and not has_multiprocessing:
            if profiler:
                profiler.log_message(f"Warning: Parallel index requested but multiprocessing not available. Using standard index.")
            mode = 'standard'
        
        # Log if an explicit mode was requested (not auto)
        if profiler and original_mode != 'auto':
            profiler.log_message(f"Using explicitly requested index mode: {mode}")
        
        # Dynamic import of the selected implementation
        try:
            # Get the class path string for the selected mode
            selected_class = IndexFactory.INDEX_CLASSES.get(mode, IndexFactory.INDEX_CLASSES['standard'])
            
            # Split into module name and class name
            module_name, class_name = selected_class.rsplit(".", 1)
            
            # Import the module dynamically
            module = __import__(f"src.index.{module_name}", fromlist=[class_name])
            
            # Log the final selection
            if profiler:
                profiler.log_message(f"Created index implementation: {module_name}.{class_name}")
            
            # Instantiate and return the index class
            return getattr(module, class_name)(documents_dir, stopwords_file, special_chars_file, profiler)
        except ImportError as e:
            # Handle import errors by falling back to StandardIndex
            if profiler:
                profiler.log_message(f"Warning: Failed to import {mode} index implementation: {e}. Using standard index.")
            print(f"Warning: Failed to import {mode} index implementation: {e}. Using standard index.")
            from src.index.standard_index import StandardIndex
            return StandardIndex(documents_dir, stopwords_file, special_chars_file, profiler)