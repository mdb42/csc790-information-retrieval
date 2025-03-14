# src/index/parallel_index.py
"""
Parallel Document Index
Author: Matthew Branson
Date: March 14, 2025

This module implements a parallelized version of the document index that leverages
multiprocessing to efficiently process and index large document collections.
"""
import os
import multiprocessing
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict, Any, Optional

from src.index.standard_index import StandardIndex


class ParallelIndex(StandardIndex):
    """
    Parallel implementation of a document index.
    
    This class extends StandardIndex to provide parallel document processing
    capabilities. It distributes document processing across multiple processes
    to improve indexing speed for larger document collections.
    
    Attributes:
        num_workers (int): Number of worker processes to use
        chunk_size (int): Number of documents to process per worker chunk
    """
    def __init__(self, documents_dir=None, stopwords_file=None, special_chars_file=None, 
                 profiler=None, num_workers=None, chunk_size=None):
        """
        Initialize the ParallelIndex.
        
        Args:
            documents_dir (str, optional): Directory containing documents to index
            stopwords_file (str, optional): File containing stopwords to remove
            special_chars_file (str, optional): File containing special characters to remove
            profiler (Profiler, optional): Performance profiler for timing operations
            num_workers (int, optional): Number of worker processes to use
                                        If None, will be determined automatically
            chunk_size (int, optional): Number of documents to process per worker chunk
                                       If None, will be calculated dynamically
        """
        super().__init__(documents_dir, stopwords_file, special_chars_file, profiler)
        self.num_workers = num_workers or self.get_optimal_num_workers()
        self.chunk_size = chunk_size

    def build_index(self):
        """
        Build the index by processing documents in parallel.
        
        Returns:
            ParallelIndex: The index instance (self)
            
        Raises:
            ValueError: If documents_dir is not specified
        """
        if not self.documents_dir:
            raise ValueError("Cannot build index: documents_dir not specified")
        
        timer_label = f"Parallel Index Building ({self.num_workers} workers)"
        
        # Get list of text files using os.scandir() for better memory efficiency
        filepaths = [entry.path for entry in os.scandir(self.documents_dir)
                     if entry.is_file() and entry.name.endswith('.txt')]
        
        # Calculate chunk size dynamically if not specified
        if self.chunk_size is None:
            # Balance between too many small chunks (overhead) and too few large chunks (poor load balancing)
            self.chunk_size = max(10, min(100, len(filepaths) // (self.num_workers * 2)))
        
        # Process files in parallel with timing if profiler is available
        if self.profiler:
            with self.profiler.timer(timer_label):
                results = self._parallel_process_files(filepaths)
        else:
            results = self._parallel_process_files(filepaths)
        
        # Filter out None results (failed processing)
        results = [r for r in results if r]
        
        # Add each document to the index
        for filename, term_freqs in results:
            self.add_document(term_freqs, filename)
        
        return self

    def _parallel_process_files(self, filepaths: List[str]) -> List[Tuple[str, Dict[str, int]]]:
        """
        Process multiple files in parallel using a multiprocessing Pool.
        
        Args:
            filepaths (List[str]): List of file paths to process
            
        Returns:
            List[Tuple[str, Dict[str, int]]]: List of (filename, term_frequencies) tuples
        """
        # Only use parallel processing if we have enough files to justify the overhead
        if len(filepaths) > self.num_workers:
            try:
                # Sort files by size (largest first) for better load balancing
                # This helps distribute work more evenly across workers
                file_sizes = [(fp, os.path.getsize(fp)) for fp in filepaths]
                sorted_files = sorted(file_sizes, key=lambda x: x[1], reverse=True)
                
                # Prepare arguments for worker processes
                args_list = []
                for i, (fp, _) in enumerate(sorted_files):
                    args_list.append((fp, self.stopwords, self.special_chars))
                
                # Process files in parallel using a process pool
                with Pool(processes=self.num_workers) as pool:
                    results = pool.map(
                        ParallelIndex._process_file_static, 
                        args_list,
                        chunksize=self.chunk_size
                    )
            except Exception as e:
                # Fall back to sequential processing if parallel processing fails
                if self.profiler:
                    self.profiler.log_message(f"Error in parallel processing: {e}. Falling back to sequential.")
                results = [self._process_single_file(fp) for fp in filepaths]
        else:
            # Use sequential processing for small file sets
            results = [self._process_single_file(fp) for fp in filepaths]
        
        return results

    @staticmethod
    def _process_file_static(args: Tuple) -> Optional[Tuple[str, Dict[str, int]]]:
        """
        Static method for processing a file in a worker process.
        
        This method must be static to be picklable for multiprocessing.
        It implements a similar pipeline to _process_single_file but is
        self-contained to work in separate processes.
        
        Args:
            args (Tuple): Tuple containing (filepath, stopwords, special_chars)
            
        Returns:
            Optional[Tuple[str, Dict[str, int]]]: Tuple containing the filename
                                                and term frequency dictionary,
                                                or None if processing failed
        """
        filepath, stopwords, special_chars = args
        try:
            # Import necessary modules within the method to ensure they're available
            # in the worker process
            import os
            import re
            from collections import Counter
            import nltk
            from nltk.stem import PorterStemmer
            from nltk.tokenize import word_tokenize
            
            stemmer = PorterStemmer()
            
            # Read file content with error handling for encoding issues
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            
            if not text:
                return None
            
            # Tokenize and convert to lowercase
            tokens = word_tokenize(text.lower())
            
            # Remove special characters
            if special_chars:
                pattern = re.compile(f'[{re.escape("".join(special_chars))}]')
                tokens = [pattern.sub('', t) for t in tokens]
            
            # Filter alphabetic tokens and remove stopwords
            tokens = [t for t in tokens if t.isalpha() and t not in stopwords]
            
            # Apply stemming
            tokens = [stemmer.stem(t) for t in tokens]
            
            if not tokens:
                return None
            
            # Extract filename and count term frequencies
            filename = os.path.basename(filepath)
            term_counts = Counter(tokens)
            
            return filename, term_counts
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None
    
    @staticmethod
    def get_optimal_num_workers() -> int:
        """
        Determine the optimal number of worker processes based on system resources.
        
        Returns:
            int: Optimal number of worker processes
        """
        num_cores = cpu_count()
        return min(max(1, num_cores - 1), 16) # Reserve one core for system tasks