# src/text_processor/factory.py
import os

class TextProcessorFactory:

    PARALLEL_DOC_THRESHOLD = 5000
    
    @staticmethod
    def create_processor(documents_dir, stopwords_file=None, special_chars_file=None, 
                         profiler=None, mode='auto', doc_count=None):
        from src.text_processor import StandardTextProcessor
        from src.text_processor import ParallelTextProcessor
        
        if doc_count is None and mode == 'auto':
            try:
                doc_count = len([f for f in os.listdir(documents_dir) if f.endswith('.txt')])
            except Exception as e:
                print(f"Warning: Could not count files in {documents_dir}: {e}")
                return StandardTextProcessor(documents_dir, stopwords_file, special_chars_file, profiler)
        
        if mode == 'auto':
            # Based on benchmark analysis, parallel becomes beneficial around 5000 docs
            if doc_count and doc_count >= TextProcessorFactory.PARALLEL_DOC_THRESHOLD:
                return ParallelTextProcessor(documents_dir, stopwords_file, special_chars_file, profiler)
            else:
                return StandardTextProcessor(documents_dir, stopwords_file, special_chars_file, profiler)
        elif mode == 'parallel':
            return ParallelTextProcessor(documents_dir, stopwords_file, special_chars_file, profiler)
        else:  # Default to standard for any other mode
            return StandardTextProcessor(documents_dir, stopwords_file, special_chars_file, profiler)