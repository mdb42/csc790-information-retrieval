# src/index/parallel_index.py
"""
- better default chunking
- file size-based load balancing
- limited maximum workers to prevent diminishing returns from over-parallelization
- added fallback to sequential processing if parallel fails
"""
import os
import multiprocessing
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict, Any, Optional

from src.index.standard_index import StandardIndex

class ParallelIndex(StandardIndex):    
    def __init__(self, documents_dir=None, stopwords_file=None, special_chars_file=None, 
                 profiler=None, num_workers=None, chunk_size=None):
        super().__init__(documents_dir, stopwords_file, special_chars_file, profiler)
        self.num_workers = num_workers or self.get_optimal_num_workers()
        self.chunk_size = chunk_size

    def build_index(self):
        if not self.documents_dir:
            raise ValueError("Cannot build index: documents_dir not specified")
        
        timer_label = f"Parallel Index Building ({self.num_workers} workers)"
        
        filepaths = [entry.path for entry in os.scandir(self.documents_dir)
                     if entry.is_file() and entry.name.endswith('.txt')]
        
        if self.chunk_size is None:
            self.chunk_size = max(10, min(100, len(filepaths) // (self.num_workers * 2)))
        
        if self.profiler:
            with self.profiler.timer(timer_label):
                results = self._parallel_process_files(filepaths)
        else:
            results = self._parallel_process_files(filepaths)
        
        results = [r for r in results if r]
        
        for filename, term_freqs in results:
            self.add_document(term_freqs, filename)
        
        return self

    def _parallel_process_files(self, filepaths: List[str]) -> List[Tuple[str, Dict[str, int]]]:
        if len(filepaths) > self.num_workers:
            try:
                file_sizes = [(fp, os.path.getsize(fp)) for fp in filepaths]
                sorted_files = sorted(file_sizes, key=lambda x: x[1], reverse=True)
                
                args_list = []
                for i, (fp, _) in enumerate(sorted_files):
                    args_list.append((fp, self.stopwords, self.special_chars))
                
                with Pool(processes=self.num_workers) as pool:
                    results = pool.map(
                        ParallelIndex._process_file_static, 
                        args_list,
                        chunksize=self.chunk_size
                    )
            except Exception as e:
                if self.profiler:
                    self.profiler.log_message(f"Error in parallel processing: {e}. Falling back to sequential.")
                results = [self._process_single_file(fp) for fp in filepaths]
        else:
            results = [self._process_single_file(fp) for fp in filepaths]
        
        return results

    @staticmethod
    def _process_file_static(args: Tuple) -> Optional[Tuple[str, Dict[str, int]]]:
        filepath, stopwords, special_chars = args
        try:
            import os
            import re
            from collections import Counter
            import nltk
            from nltk.stem import PorterStemmer
            from nltk.tokenize import word_tokenize
            
            stemmer = PorterStemmer()
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            
            if not text:
                return None
            
            tokens = word_tokenize(text.lower())
            
            if special_chars:
                pattern = re.compile(f'[{re.escape("".join(special_chars))}]')
                tokens = [pattern.sub('', t) for t in tokens]
            
            tokens = [t for t in tokens if t.isalpha() and t not in stopwords]
            tokens = [stemmer.stem(t) for t in tokens]
            
            if not tokens:
                return None
            
            filename = os.path.basename(filepath)
            term_counts = Counter(tokens)
            
            return filename, term_counts
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None
    
    @staticmethod
    def get_optimal_num_workers() -> int:
        num_cores = cpu_count()
        return min(max(1, num_cores - 1), 16)