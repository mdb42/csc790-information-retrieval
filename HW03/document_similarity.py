"""
I'm taking care of the nmost frequent terms now, because the refactor to use
an abstract VSM class is going to take forever and I will otherwise forget.
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
import pickle
from utils import Logger, Timer

logging = Logger()

####################################################
# Vector Space Model implementation
####################################################
class VectorSpaceModel:
    def __init__(self, documents_dir=None, stopwords_file=None, special_chars_file=None,       
                 index_file=None, use_existing_index=False):
        self.documents_dir = documents_dir
        self.stopwords_file = stopwords_file
        self.special_chars_file = special_chars_file
        self.index_file = index_file
        self.use_existing_index = use_existing_index

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        self.stopwords = self._load_stopwords()
        self.special_chars = self._load_special_chars()
        self.stemmer = PorterStemmer()
        
        self.documents = []
        self.filenames = []
        self.term_doc_freqs = defaultdict(Counter)
        self.doc_term_freqs = defaultdict(Counter)
        self.doc_lengths = {}
        
        self.vocab_size = 0
        self.doc_count = 0
        self.idf_values = {}
        
        self.weights = {}
        self.magnitudes = {}
        
        if use_existing_index and index_file and os.path.exists(index_file):
            if self.load_index(index_file):
                logging.info("Using existing index")
            else:
                logging.info("Failed to load existing index. Building new index...")
                self._load_documents()
                if index_file:
                    self.save_index(index_file)
        elif documents_dir:
            self._load_documents()
            if index_file:
                self.save_index(index_file)
        else:
            logging.error("Either documents_dir or a valid index_file with use_existing_index=True must be provided")
    
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
        tokens = word_tokenize(text.lower())
        if self.special_chars:
            pattern = re.compile(f'[{re.escape("".join(self.special_chars))}]')
            tokens = [pattern.sub('', t) for t in tokens]
            tokens = [t for t in tokens if t.isalpha()]
        else:
            tokens = [t for t in tokens if t.isalpha()]
        tokens = [t for t in tokens if t not in self.stopwords]
        tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens
    
    def _load_documents(self):
        with Timer("Document loading"):
            all_files = [f for f in os.listdir(self.documents_dir) if f.endswith('.txt')]
            doc_id = 0
            for filename in all_files:
                filepath = os.path.join(self.documents_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    if not text.strip():
                        continue
                    tokens = self._preprocess_text(text)
                    if not tokens:
                        continue
                    self.filenames.append(filename)
                    term_counts = Counter(tokens)
                    for term, freq in term_counts.items():
                        self.term_doc_freqs[term][doc_id] = freq
                        self.doc_term_freqs[doc_id][term] = freq
                    self.doc_lengths[doc_id] = sum(term_counts.values())
                    doc_id += 1
                except Exception as e:
                    logging.error(f"Error reading {filename}: {str(e)}")
            
            self.doc_count = len(self.doc_term_freqs)
            self.vocab_size = len(self.term_doc_freqs)
            self._compute_idf_values()
            self._precompute_weights()
            logging.info(f"Loaded {self.doc_count} documents with {self.vocab_size} unique terms")
    
    def _compute_idf_values(self):
        for term, doc_freqs in self.term_doc_freqs.items():
            df = len(doc_freqs)
            self.idf_values[term] = math.log10(self.doc_count / max(df, 1))
    
    def get_most_frequent_terms(self, n=10):
        term_totals = Counter()
        for term, doc_freqs in self.term_doc_freqs.items():
            term_totals[term] = sum(doc_freqs.values())
        return term_totals.most_common(n)
    
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
            similarities = []
            for i in range(self.doc_count):
                for j in range(i + 1, self.doc_count):
                    sim = self._compute_vector_similarity(i, j, weighting)
                    if sim > 0:
                        similarities.append((i, j, sim))
            top_pairs = heapq.nlargest(k, similarities, key=lambda x: x[2])
            formatted_pairs = []
            for doc1_id, doc2_id, similarity in top_pairs:
                formatted_pairs.append((
                    self.filenames[doc1_id],
                    self.filenames[doc2_id],
                    similarity
                ))
            return formatted_pairs
    
    def save_index(self, filepath):
        try:
            with Timer("Saving index"):
                term_doc_dict = {term: dict(docs) for term, docs in self.term_doc_freqs.items()}
                doc_term_dict = {doc_id: dict(terms) for doc_id, terms in self.doc_term_freqs.items()}
                weights_dict = {}
                for weight_type, doc_weights in self.weights.items():
                    weights_dict[weight_type] = {doc_id: dict(weights) for doc_id, weights in doc_weights.items()}
                index_data = {
                    'filenames': self.filenames,
                    'term_doc_freqs': term_doc_dict,
                    'doc_term_freqs': doc_term_dict,
                    'doc_lengths': self.doc_lengths,
                    'vocab_size': self.vocab_size,
                    'doc_count': self.doc_count,
                    'idf_values': self.idf_values,
                    'most_frequent': self.most_frequent,
                    'weights': weights_dict,
                    'magnitudes': self.magnitudes
                }
                logging.info(f"Attempting to save index to: {os.path.abspath(filepath)}")
                directory = os.path.dirname(filepath)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory)
                with open(filepath, 'wb') as f:
                    pickle.dump(index_data, f)
                if os.path.exists(filepath):
                    file_size = os.path.getsize(filepath) / (1024 * 1024)
                    logging.info(f"Index saved to {filepath} ({file_size:.2f} MB)")
                else:
                    logging.error(f"Failed to create index file at {filepath}")
        except Exception as e:
            logging.error(f"Error saving index: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            
    def load_index(self, filepath):
        try:
            with Timer("Loading index"):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                self.filenames = data['filenames']
                self.doc_count = data['doc_count']
                self.vocab_size = data['vocab_size']
                self.most_frequent = data['most_frequent']
                self.idf_values = data['idf_values']
                self.doc_lengths = data['doc_lengths']
                self.term_doc_freqs = defaultdict(Counter)
                for term, docs in data['term_doc_freqs'].items():
                    self.term_doc_freqs[term] = Counter(docs)
                self.doc_term_freqs = defaultdict(Counter)
                for doc_id, terms in data['doc_term_freqs'].items():
                    doc_id = int(doc_id) if isinstance(doc_id, str) else doc_id
                    self.doc_term_freqs[doc_id] = Counter(terms)
                self.weights = {}
                for weight_type, doc_weights in data['weights'].items():
                    self.weights[weight_type] = {}
                    for doc_id, weights in doc_weights.items():
                        doc_id = int(doc_id) if isinstance(doc_id, str) else doc_id
                        self.weights[weight_type][doc_id] = weights
                self.magnitudes = data['magnitudes']
                logging.info(f"Successfully loaded index from {filepath}")
                return True
        except Exception as e:
            logging.error(f"Error loading index: {str(e)}")
            return False

    def display_info(self):
        print("=================== CSC790-IR Homework 03 ===================")
        print("First Name: Matthew")
        print("Last Name : Branson")
        print("=============================================================")

    def display_most_frequent_terms(self, n=10):
        print(f"The number of unique words is: {self.vocab_size:,}")
        print(f"The top {n} most frequent words are:")
        for i, (term, freq) in enumerate(self.get_most_frequent_terms(n), 1):
            print(f"    {i}. {term} ({freq:,})")
        print("=============================================================")
    
####################################################
# Main function
####################################################
def main():
    logging.info("========== Starting new run ==========")
    initialization_start = time.time()

    parser = argparse.ArgumentParser(
        description='Vector Space Model for document similarity.')
    parser.add_argument('--documents_dir', default='documents',
                    help='Directory containing documents to index')
    parser.add_argument('--stopwords_file', default='stopwords.txt',
                    help='File containing stopwords')
    parser.add_argument('--special_chars_file', default='special_chars.txt',
                    help='File containing special characters to remove')
    parser.add_argument('--index_file', default='index.pkl',
                    help='Path to save/load the index')
    parser.add_argument('--use_existing', action='store_true',
                    help='Use existing index if available')
    
    args = parser.parse_args()
    
    vsm = None
    if args.use_existing and os.path.exists(args.index_file):
        logging.info(f"Loading existing index from {args.index_file}")
        vsm = VectorSpaceModel(
            stopwords_file=args.stopwords_file,
            special_chars_file=args.special_chars_file
        )
        if not vsm.load_index(args.index_file):
            logging.error("Failed to load index, building new one")
            vsm = VectorSpaceModel(
                documents_dir=args.documents_dir,
                stopwords_file=args.stopwords_file,
                special_chars_file=args.special_chars_file
            )
            vsm.save_index(args.index_file)
    else:
        vsm = VectorSpaceModel(
            documents_dir=args.documents_dir,
            stopwords_file=args.stopwords_file,
            special_chars_file=args.special_chars_file
        )
        logging.info(f"Saving index to {args.index_file}")
        vsm.save_index(args.index_file)
    
    initialization_time = time.time() - initialization_start

    vsm.display_info()
    vsm.display_most_frequent_terms()
    
    while True:
        try:
            k = int(input("\nEnter the number of top similar document pairs (k): "))
            if k <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    print("\nThe top k closest documents are:")
    
    similarity_start = time.time()

    tf_similar = vsm.find_similar_documents(k=k, weighting='tf')
    print(f"\n1. Using tf:")
    for i, (doc1, doc2, sim) in enumerate(tf_similar, 1):
        print(f"    {doc1}, {doc2} with similarity of {sim:.2f}")
    
    tfidf_similar = vsm.find_similar_documents(k=k, weighting='tfidf')
    print(f"\n2. Using tfidf:")
    for i, (doc1, doc2, sim) in enumerate(tfidf_similar, 1):
        print(f"    {doc1}, {doc2} with similarity of {sim:.2f}")
    
    sublinear_similar = vsm.find_similar_documents(k=k, weighting='sublinear')
    print(f"\n3. Using wfidf:")
    for i, (doc1, doc2, sim) in enumerate(sublinear_similar, 1):
        print(f"    {doc1}, {doc2} with similarity of {sim:.2f}")

    similarity_time = time.time() - similarity_start

    total_time = initialization_time + similarity_time
    logging.info(f"Total processing time: {total_time:.4f} seconds (Initialization: {initialization_time:.4f}s, Similarity calculations: {similarity_time:.4f}s)")
    print("\n=============================================================")
    print("Total processing time: {:.4f} seconds".format(total_time))
    print("=============================================================")

if __name__ == "__main__":
    main()