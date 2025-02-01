import os
import argparse
from sys import getsizeof
from collections import Counter, defaultdict
from collections.abc import Iterable
import pickle
import json
import nltk
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from query_parser import QueryParser, query_parser_demo

###############################################################################
# Utility Functions
###############################################################################

def print_banner():
    print("""
=================== CSC790-IR Homework 01 ===================
First Name: Matthew
Last Name : Branson
=============================================================
""")

def load_stopwords(stop_words_file):
    print(f"[+] Loading stopwords from file '{stop_words_file}'...")
    stopwords = set()
    try:
        with open(stop_words_file, encoding="utf-8") as file:
            stopwords = {line.strip().lower() for line in file if line.strip()}
    except FileNotFoundError:
        print(f"[!] Error: The file {stop_words_file} was not found.")
    except IOError as e:
        print(f"[!] Error: An I/O error occurred while reading {stop_words_file}: {e}")
    except Exception as e:
        print(f"[!] An unexpected error occurred: {e}")
    return stopwords

def display_top_n_terms(inv_index, n=10):
    print(f"\nTop {n} most frequent terms in the entire corpus:")
    for term, freq in inv_index.term_frequency.most_common(n):
        print(f"  {term}: {freq}")

###############################################################################
# The InvertedIndex Class
###############################################################################

class InvertedIndex:
    def __init__(self, stop_words=None):
        self.index = defaultdict(set) # {term: set(doc_ids)}
        self.doc_id_map = {}          # {filename: doc_id}
        self.reverse_doc_id_map = {}  # {doc_id: filename}
        self.stop_words = stop_words or set()
        self.stemmer = nltk.stem.PorterStemmer()
        self.term_frequency = Counter()

    def add_document(self, doc_id, text):
        tokens = [self.stemmer.stem(word.lower()) 
                for word in nltk.word_tokenize(text) 
                if word.isalpha() and word.lower() not in self.stop_words]
        for term in set(tokens):
            self.index[term].add(doc_id)
        self.term_frequency.update(tokens)

    def get_document_count(self):
        return len(self.doc_id_map)
    
    def get_vocabulary_size(self):
        return len(self.index)
    
    def get_index_size_in_bytes(self, obj=None): 
        seen = set()
        
        def sizeof(obj):
            if id(obj) in seen:
                return 0
            seen.add(id(obj))
            size = getsizeof(obj)
            
            if isinstance(obj, dict):
                size += sum(sizeof(k) + sizeof(v) for k, v in obj.items())
            elif isinstance(obj, (set, list, tuple)):
                size += sum(sizeof(x) for x in obj)
            elif isinstance(obj, str):
                # String size is already included in getsizeof
                pass
            elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
                size += sum(sizeof(x) for x in obj)
                
            return size
        
        if obj is None:
            total = sizeof(self.index)
            total += sizeof(self.doc_id_map)
            total += sizeof(self.reverse_doc_id_map)
            total += sizeof(self.stop_words)
            total += sizeof(self.term_frequency)
            return total
        
        return sizeof(obj)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'doc_id_map': self.doc_id_map,
                'reverse_doc_id_map': self.reverse_doc_id_map,
                'stop_words': self.stop_words,
                'term_frequency': self.term_frequency
            }, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.index = data['index']
            self.doc_id_map = data['doc_id_map']
            self.reverse_doc_id_map = data['reverse_doc_id_map']
            self.stop_words = data['stop_words']
            self.term_frequency = data['term_frequency']

    def boolean_retrieve(self, query_str):
        all_docs = set(self.doc_id_map.values())
        parser = QueryParser(self.index, self.stop_words, self.stemmer, all_docs)
        try:
            return parser.parse(query_str)
        except Exception as e:
            print(f"[!] Could not parse query expression '{query_str}': {e}")
            return set()

    def export_to_json(self, filepath):
        export_data = {
            "metadata": {
                "document_count": self.get_document_count(),
                "vocabulary_size": self.get_vocabulary_size(),
                "index_size_bytes": self.get_index_size_in_bytes(),
                "document_mapping": self.doc_id_map
            },
            "term_frequencies": dict(self.term_frequency),
            "index": {term: sorted(list(doc_ids)) 
                    for term, doc_ids in self.index.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, sort_keys=True)

###############################################################################
# Building the Inverted Index
###############################################################################

def build_inverted_index(directory_path, stop_words):
    print(f"[+] Building index from documents in '{directory_path}'...")
    inv_index = InvertedIndex(stop_words=stop_words)
    current_doc_id = 1

    try:
        files = os.listdir(directory_path)
    except FileNotFoundError:
        print(f"[!] Error: Directory {directory_path} not found")
        return inv_index
    except PermissionError:
        print(f"[!] Error: Permission denied accessing {directory_path}")
        return inv_index

    for filename in files:
        if not filename.endswith('.txt'):
            continue
        
        filepath = os.path.join(directory_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            if not text.strip():
                print(f"[!] Warning: Empty file {filename}")
                continue         
            inv_index.doc_id_map[filename] = current_doc_id
            inv_index.reverse_doc_id_map[current_doc_id] = filename
            inv_index.add_document(doc_id=current_doc_id, text=text)
            current_doc_id += 1
        except Exception as e:
            print(f"[!] Error processing {filename}: {e}")
            continue

    return inv_index

def process_document_chunk(chunk_data):
    chunk, start_id, directory_path, stop_words = chunk_data
    local_idx = InvertedIndex(stop_words=stop_words)
    current_id = start_id
    errors = [] # Aggregate errors now
    for filename in chunk:
        filepath = os.path.join(directory_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            local_idx.doc_id_map[filename] = current_id
            local_idx.reverse_doc_id_map[current_id] = filename
            local_idx.add_document(current_id, text)
            current_id += 1
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
            continue
    return {
        'index': local_idx.index,
        'doc_id_map': local_idx.doc_id_map,
        'reverse_doc_id_map': local_idx.reverse_doc_id_map,
        'term_frequency': local_idx.term_frequency,
        'errors': errors
    }

def build_inverted_index_parallel(directory_path, stop_words, max_workers=None):
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    inv_index = InvertedIndex(stop_words=stop_words)
    doc_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    
    chunk_size = max(1, len(doc_files) // max_workers)
    doc_chunks = [doc_files[i:i + chunk_size] 
                 for i in range(0, len(doc_files), chunk_size)]
    chunk_data = [
        (chunk, 1 + i * chunk_size, directory_path, stop_words)
        for i, chunk in enumerate(doc_chunks)
    ]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_document_chunk, chunk_data))
    
    for partial_index in results:
        inv_index.doc_id_map.update(partial_index['doc_id_map'])
        inv_index.reverse_doc_id_map.update(partial_index['reverse_doc_id_map'])
        
        for term, doc_ids in partial_index['index'].items():
            inv_index.index[term].update(doc_ids)
        
        for term, freq in partial_index['term_frequency'].items():
            inv_index.term_frequency[term] += freq
    return inv_index

###############################################################################
# Run Function - Orchestrating the Indexing + Querying
###############################################################################

def run(documents_dir, stopwords_file, 
        index_file, use_existing_index, use_parallel, top_n):
    try:
        nltk.data.find('tokenizers/punkt_tab/english.pickle')
    except LookupError:
        nltk.download('punkt_tab')

    print_banner()
    stop_words = load_stopwords(stopwords_file)
    if use_existing_index and os.path.exists(index_file):
        print("[+] Loading existing index from file...")
        inv_index = InvertedIndex()
        inv_index.load(index_file)
    else:
        print("[+] Building a new index from the documents directory...")
        if use_parallel:
            print(f"[+] Building index using {mp.cpu_count()} CPU cores...")
            inv_index = build_inverted_index_parallel(documents_dir, stop_words=stop_words)
        else:
            inv_index = build_inverted_index(documents_dir, stop_words=stop_words)

        print("[+] Saving the new index to disk...")
        inv_index.save(index_file)

    size_in_bytes = inv_index.get_index_size_in_bytes()
    size_in_mb = size_in_bytes / (1024 * 1024)
    print(f"[+] Inverted Index built/loaded. Size in memory: "
          f"{size_in_bytes} bytes ({size_in_mb:.2f} MB)")

    display_top_n_terms(inv_index, n=top_n)

    print("[+] Exporting index to JSON for inspection...")
    json_path = index_file.replace('.pkl', '.json')
    inv_index.export_to_json(json_path)

    query_parser_demo(inv_index, inv_index.stemmer)

def main(documents_dir=None, stopwords_file=None, 
         index_file=None, use_existing_index=False, use_parallel=True, top_n=10):
    if documents_dir is None or stopwords_file is None or index_file is None:
        parser = argparse.ArgumentParser(
            description='Build and query an inverted index from text documents.')
        parser.add_argument('--documents_dir', default='documents',
                          help='Directory containing documents to index')
        parser.add_argument('--stopwords_file', default='stopwords.txt',
                          help='File containing stopwords')
        parser.add_argument('--index_file', default='index.pkl',
                          help='Path to save/load the index')
        parser.add_argument('--no_parallel', action='store_true',
                          help='Disable parallel processing')
        parser.add_argument('--use_existing', action='store_true',
                          help='Use existing index if available')
        parser.add_argument('--top_n', type=int, default=10,
                          help='Number of most frequent terms to display')
        
        args = parser.parse_args()
        documents_dir = args.documents_dir
        stopwords_file = args.stopwords_file
        index_file = args.index_file
        use_existing_index = args.use_existing
        use_parallel = not args.no_parallel
        top_n = args.top_n

    run(documents_dir, stopwords_file, index_file, 
        use_existing_index=use_existing_index, use_parallel=use_parallel, 
        top_n=top_n)

if __name__ == "__main__":
    main()
