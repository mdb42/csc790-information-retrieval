# src/text_processor.py
import os
import re
import nltk
from collections import Counter
from multiprocessing import Pool
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class TextProcessor:
    def __init__(self, documents_dir, stopwords_file=None, special_chars_file=None, parallel=False):
        self.documents_dir = documents_dir
        self.parallel = parallel
        self.stopwords = self._load_stopwords(stopwords_file)
        self.special_chars = self._load_special_chars(special_chars_file)
        self.stemmer = PorterStemmer()
        self.documents = []
        self.filenames = []

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

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
        tokens = word_tokenize(text.lower())

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
        filepaths = [os.path.join(self.documents_dir, f) for f in os.listdir(self.documents_dir) if f.endswith('.txt')]

        if self.parallel:
            with Pool() as pool:
                results = pool.map(self._process_single_file, filepaths)
        else:
            results = [self._process_single_file(fp) for fp in filepaths]

        results = [r for r in results if r]  # Remove None values
        self.filenames, term_freqs = zip(*results) if results else ([], [])

        return self.filenames, term_freqs
