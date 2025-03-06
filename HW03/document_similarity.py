"""
Well, that worked out nicely! 14.6539 seconds!
I'm going to try numpy arrays for the vectors and see if that speeds things up.
It will probably slow things down at first until I work out parallelization for it.
I already know scipy.sparse and scikit-learn can do this faster, but I'll maybe make
it adaptively use whatever depending on hardware and environment.
"""
import os
import time
import math
import argparse
import re
from collections import Counter, defaultdict
import heapq
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

####################################################
# Utility functions and classes
####################################################
class Logger:
    def __init__(self, level="INFO"):
        self.level = level
        
    def info(self, message):
        if self.level in ["INFO", "DEBUG"]:
            print(f"[INFO] {time.strftime('%Y-%m-%d %H:%M:%S')} - {message}")
            
    def error(self, message):
        print(f"[ERROR] {time.strftime('%Y-%m-%d %H:%M:%S')} - {message}")

# Initialize logger
logging = Logger()

class Timer:
    """Simple context manager for timing code blocks"""
    def __init__(self, name=None):
        self.name = name
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.name:
            logging.info(f"{self.name} took {self.interval:.4f} seconds")

####################################################
# Vector Space Model implementation
####################################################
class VectorSpaceModel:
    def __init__(self, documents_dir=None, stopwords_file=None, special_chars_file=None):
        self.documents_dir = documents_dir
        self.stopwords_file = stopwords_file
        self.special_chars_file = special_chars_file

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Load pre-processing resources
        self.stopwords = self._load_stopwords()
        self.special_chars = self._load_special_chars()
        self.stemmer = PorterStemmer()
        
        # Data structures
        self.documents = []
        self.filenames = []
        self.term_doc_freqs = defaultdict(Counter)  # {term: {doc_id: freq}}
        self.doc_term_freqs = defaultdict(Counter)  # {doc_id: {term: freq}}
        self.doc_lengths = {}  # {doc_id: total_terms}
        
        # Statistics
        self.vocab_size = 0
        self.doc_count = 0
        self.most_frequent = []
        self.idf_values = {}  # {term: idf_value}
        
        # Precomputed weights and magnitudes (for various weighting schemes)
        self.weights = {}
        self.magnitudes = {}
        
        # Load documents
        if documents_dir:
            self._load_documents()
    
    def _load_stopwords(self):
        stopwords = set()
        if not self.stopwords_file:
            return stopwords
            
        try:
            with open(self.stopwords_file, encoding="utf-8") as file:
                stopwords = {line.strip().lower() for line in file if line.strip()}
            logging.info(f"Loaded {len(stopwords)} stopwords")
        except Exception as e:
            logging.error(f"Error loading stopwords: {str(e)}")
        return stopwords
    
    def _load_special_chars(self):
        special_chars = set()
        if not self.special_chars_file:
            return special_chars
            
        try:
            with open(self.special_chars_file, encoding="utf-8") as file:
                special_chars = {line.strip() for line in file if line.strip()}
            logging.info(f"Loaded {len(special_chars)} special characters")
        except Exception as e:
            logging.error(f"Error loading special characters: {str(e)}")
        return special_chars
    
    def _preprocess_text(self, text):
        """
        Processing pipeline:
          1. Tokenize & lowercase
          2. Remove special characters using regex
          3. Filter tokens to only alphabetic characters
          4. Remove stopwords
          5. Apply stemming
        """
        # Tokenize and lowercase
        tokens = word_tokenize(text.lower())
        
        # Remove special characters in one pass using regex (if applicable)
        if self.special_chars:
            pattern = re.compile(f'[{re.escape("".join(self.special_chars))}]')
            tokens = [pattern.sub('', t) for t in tokens]
            tokens = [t for t in tokens if t.isalpha()]
        else:
            tokens = [t for t in tokens if t.isalpha()]
        
        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stopwords]
        
        # Apply stemming
        tokens = [self.stemmer.stem(t) for t in tokens]
        
        return tokens
    
    def _load_documents(self):
        with Timer("Document loading"):
            # Get all text files in the directory
            all_files = [f for f in os.listdir(self.documents_dir) if f.endswith('.txt')]
            
            # Manually track document IDs to avoid gaps for invalid documents
            doc_id = 0
            for filename in all_files:
                filepath = os.path.join(self.documents_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    
                    if not text.strip():
                        continue
                    
                    # Process document
                    tokens = self._preprocess_text(text)
                    if not tokens:
                        continue
                    
                    # Store document information
                    self.filenames.append(filename)
                    
                    # Count term frequencies
                    term_counts = Counter(tokens)
                    
                    # Update term-document and document-term indexes
                    for term, freq in term_counts.items():
                        self.term_doc_freqs[term][doc_id] = freq
                        self.doc_term_freqs[doc_id][term] = freq
                    
                    # Store document length (total terms)
                    self.doc_lengths[doc_id] = sum(term_counts.values())
                    
                    doc_id += 1  # Increment only for valid documents
                    
                except Exception as e:
                    logging.error(f"Error reading {filename}: {str(e)}")
            
            # Update statistics
            self.doc_count = len(self.doc_term_freqs)
            self.vocab_size = len(self.term_doc_freqs)
            
            # Compute IDF values
            self._compute_idf_values()
            
            # Compute most frequent terms
            self._compute_most_frequent_terms()
            
            # Precompute document vectors and their magnitudes
            self._precompute_weights()
            
            logging.info(f"Loaded {self.doc_count} documents with {self.vocab_size} unique terms")
    
    def _compute_idf_values(self):
        for term, doc_freqs in self.term_doc_freqs.items():
            # Number of documents containing the term
            df = len(doc_freqs)
            # IDF = log(N/df)
            self.idf_values[term] = math.log10(self.doc_count / max(df, 1))
    
    def _compute_most_frequent_terms(self):
        term_totals = Counter()
        for term, doc_freqs in self.term_doc_freqs.items():
            term_totals[term] = sum(doc_freqs.values())
        self.most_frequent = term_totals.most_common(10)
    
    def _precompute_weights(self):
        self.weights = {'tf': {}, 'tfidf': {}, 'sublinear': {}}
        self.magnitudes = {'tf': {}, 'tfidf': {}, 'sublinear': {}}
        
        for doc_id, terms in self.doc_term_freqs.items():
            tf_weights = {}
            tfidf_weights = {}
            sublinear_weights = {}
            
            for term, freq in terms.items():
                tf_weights[term] = freq
                idf = self.idf_values.get(term, 0)
                tfidf_weights[term] = freq * idf
                sublinear_weights[term] = (1 + math.log10(freq)) * idf if freq > 0 else 0
            
            self.weights['tf'][doc_id] = tf_weights
            self.weights['tfidf'][doc_id] = tfidf_weights
            self.weights['sublinear'][doc_id] = sublinear_weights
            
            self.magnitudes['tf'][doc_id] = math.sqrt(sum(w**2 for w in tf_weights.values()))
            self.magnitudes['tfidf'][doc_id] = math.sqrt(sum(w**2 for w in tfidf_weights.values()))
            self.magnitudes['sublinear'][doc_id] = math.sqrt(sum(w**2 for w in sublinear_weights.values()))
    
    def _compute_vector_similarity(self, doc1_id, doc2_id, weighting='tf'):
        weights1 = self.weights[weighting][doc1_id]
        weights2 = self.weights[weighting][doc2_id]
        common_terms = set(weights1.keys()) & set(weights2.keys())
        if not common_terms:
            return 0.0
        dot_product = sum(weights1[term] * weights2[term] for term in common_terms)
        magnitude1 = self.magnitudes[weighting][doc1_id]
        magnitude2 = self.magnitudes[weighting][doc2_id]
        return dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 != 0 else 0.0
    
    def find_similar_documents(self, k=10, weighting='tf'):
        with Timer(f"Similarity calculation ({weighting})"):

            # Compute all pairwise similarities
            similarities = []

            # Only compute upper triangle (avoid duplicates)
            for i in range(self.doc_count):
                for j in range(i + 1, self.doc_count):
                    sim = self._compute_vector_similarity(i, j, weighting)
                    if sim > 0:  # Only store non-zero similarities
                        similarities.append((i, j, sim))
            
            # Get top-k most similar pairs
            top_pairs = heapq.nlargest(k, similarities, key=lambda x: x[2])
            
            # Format results
            formatted_pairs = []
            for doc1_id, doc2_id, similarity in top_pairs:
                formatted_pairs.append((
                    self.filenames[doc1_id],
                    self.filenames[doc2_id],
                    similarity
                ))
            return formatted_pairs
    
    def display_info(self):
        print("=================== CSC790-IR Homework 03 (Minimal Dependencies) ===================")
        print("First Name: Matthew")
        print("Last Name : Branson")
        print("=============================================================")
        print(f"The number of unique words is: {self.vocab_size:,}")
        print("The top 10 most frequent words are:")
        for i, (term, freq) in enumerate(self.most_frequent, 1):
            print(f"    {i}. {term} ({freq:,})")
        print("=============================================================")

####################################################
# Main function
####################################################
def main():
    parser = argparse.ArgumentParser(
        description='Vector Space Model for document similarity (Minimal Dependencies implementation)'
    )
    parser.add_argument('--documents_dir', default='documents',
                      help='Directory containing documents')
    parser.add_argument('--stopwords_file', default='stopwords.txt',
                      help='File containing stopwords')
    parser.add_argument('--special_chars_file', default='special_chars.txt',
                      help='File containing special characters to filter')
    
    args = parser.parse_args()
    
    # Initialize the model
    vsm = VectorSpaceModel(
        documents_dir=args.documents_dir,
        stopwords_file=args.stopwords_file,
        special_chars_file=args.special_chars_file
    )
    
    # Display information
    vsm.display_info()
    
    # Get k from user
    while True:
        try:
            k = int(input("\nEnter the number of top similar document pairs (k): "))
            if k <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Run similarity calculations with different weighting schemes
    print("\nThe top k closest documents are:")
    
    # Term Frequency
    tf_start = time.time()
    tf_similar = vsm.find_similar_documents(k=k, weighting='tf')
    tf_time = time.time() - tf_start
    print(f"\n1. Using tf (computed in {tf_time:.4f} seconds):")
    for i, (doc1, doc2, sim) in enumerate(tf_similar, 1):
        print(f"    {i}. {doc1}, {doc2} with similarity of {sim:.4f}")
    
    # TF-IDF
    tfidf_start = time.time()
    tfidf_similar = vsm.find_similar_documents(k=k, weighting='tfidf')
    tfidf_time = time.time() - tfidf_start
    print(f"\n2. Using tfidf (computed in {tfidf_time:.4f} seconds):")
    for i, (doc1, doc2, sim) in enumerate(tfidf_similar, 1):
        print(f"    {i}. {doc1}, {doc2} with similarity of {sim:.4f}")
    
    # Sublinear TF-IDF
    sublinear_start = time.time()
    sublinear_similar = vsm.find_similar_documents(k=k, weighting='sublinear')
    sublinear_time = time.time() - sublinear_start
    print(f"\n3. Using wfidf (computed in {sublinear_time:.4f} seconds):")
    for i, (doc1, doc2, sim) in enumerate(sublinear_similar, 1):
        print(f"    {i}. {doc1}, {doc2} with similarity of {sim:.4f}")
    
    # Overall timing
    print("\n=============================================================")
    print(f"Total similarity computation time: {tf_time + tfidf_time + sublinear_time:.4f} seconds")
    print("=============================================================")


if __name__ == "__main__":
    main()