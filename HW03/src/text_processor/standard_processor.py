# src/text_processor/standard_processor.py
import os
import re
from collections import Counter
from src.text_processor import BaseTextProcessor

class StandardTextProcessor(BaseTextProcessor):
    def __init__(self, documents_dir, stopwords_file=None, special_chars_file=None, profiler=None):
        super().__init__(documents_dir, stopwords_file, special_chars_file, profiler)
        self.stopwords = self._load_stopwords(stopwords_file)
        self.special_chars = self._load_special_chars(special_chars_file)
        self.filenames = []
        
        import nltk
        from nltk.stem import PorterStemmer
        from nltk.tokenize import word_tokenize
        
        self.word_tokenize = word_tokenize
        self.stemmer = PorterStemmer()
                    
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

    def _load_stopwords(self, filepath):
        if not filepath:
            return set()
        try:
            with open(filepath, encoding="utf-8") as file:
                return {line.strip().lower() for line in file if line.strip()}
        except Exception as e:
            print(f"Warning: Failed to load stopwords. {e}")
            return set()

    def _load_special_chars(self, filepath):
        if not filepath:
            return set()
        try:
            with open(filepath, encoding="utf-8") as file:
                return {line.strip() for line in file if line.strip()}
        except Exception as e:
            print(f"Warning: Failed to load special characters. {e}")
            return set()

    def _preprocess_text(self, text):
        tokens = self.word_tokenize(text.lower())

        if self.special_chars:
            pattern = re.compile(f'[{re.escape("".join(self.special_chars))}]')
            tokens = [pattern.sub('', t) for t in tokens]

        tokens = [t for t in tokens if t.isalpha() and t not in self.stopwords]
        tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens

    def _process_single_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            if not text:
                return None

            tokens = self._preprocess_text(text)
            if not tokens:
                return None

            filename = os.path.basename(filepath)
            term_counts = Counter(tokens)
            return filename, term_counts
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None

    def process_documents(self):
        timer_label = "Sequential Document Processing"
        
        filepaths = [os.path.join(self.documents_dir, f) 
                    for f in os.listdir(self.documents_dir) 
                    if f.endswith('.txt')]

        if self.profiler:
            with self.profiler.timer(timer_label):
                results = [self._process_single_file(fp) for fp in filepaths]
        else:
            results = [self._process_single_file(fp) for fp in filepaths]
            
        results = [r for r in results if r]
        
        if results:
            self.filenames, term_freqs = zip(*results)
        else:
            self.filenames, term_freqs = [], []
            
        return self.filenames, term_freqs