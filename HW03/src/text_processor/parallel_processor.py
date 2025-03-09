# src/text_processor/parallel_processor.py
import os
from multiprocessing import Pool, cpu_count
from src.text_processor import StandardTextProcessor

class ParallelTextProcessor(StandardTextProcessor):    
    def __init__(self, documents_dir, stopwords_file=None, special_chars_file=None, 
                 profiler=None, num_workers=None, chunk_size=None):
        super().__init__(documents_dir, stopwords_file, special_chars_file, profiler)
        self.num_workers = num_workers or self.get_optimal_num_workers()
        self.chunk_size = chunk_size
    
    def process_documents(self):
        timer_label = f"Parallel Document Processing ({self.num_workers} workers)"
        
        filepaths = [os.path.join(self.documents_dir, f) 
                    for f in os.listdir(self.documents_dir) 
                    if f.endswith('.txt')]
        
        if self.chunk_size is None:
            self.chunk_size = max(5, len(filepaths) // (self.num_workers * 2))
        
        
        if self.profiler:
            with self.profiler.timer(timer_label):
                results = self._parallel_process_files(filepaths)
        else:
            results = self._parallel_process_files(filepaths)
            
        results = [r for r in results if r]
        
        if results:
            self.filenames, term_freqs = zip(*results)
        else:
            self.filenames, term_freqs = [], []
        
        return self.filenames, term_freqs
    
    def _parallel_process_files(self, filepaths):
        if len(filepaths) > self.num_workers:
            with Pool(processes=self.num_workers) as pool:
                results = pool.map(
                    ParallelTextProcessor._process_file_static, 
                    [(self, fp) for fp in filepaths],
                    chunksize=self.chunk_size
                )
        else:
            results = [self._process_single_file(fp) for fp in filepaths]
        
        return results
    
    @staticmethod
    def _process_file_static(args):
        processor, filepath = args
        return processor._process_single_file(filepath)
    
    @staticmethod
    def get_optimal_num_workers():
        return max(1, cpu_count() - 1)