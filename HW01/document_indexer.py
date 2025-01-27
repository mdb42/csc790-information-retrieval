import os
import sys
import pickle
import nltk
from collections import Counter

def print_banner():
    print("================================================")
    print("=============== CSC790-IR Homework 01 ===============")
    print("First Name: Matthew")
    print("Last Name : Branson")
    print("================================================")

def load_stopwords(stop_words_file):
    stopwords = set()
    with open(stop_words_file, 'r', encoding='utf-8') as f:
        for line in f:
            w = line.strip()
            if w:
                stopwords.add(w.lower())
    return stopwords

class InvertedIndex:
    def __init__(self, stop_words=None):
        try:
            nltk.data.find('tokenizers/punkt_tab/english.pickle')
        except LookupError:
            nltk.download('punkt_tab')

        if not os.path.exists('documents'):
            os.makedirs('documents')
        
        self.index = {}               # {term: set_of_docIDs}
        self.doc_id_map = {}          # {filename: doc_id}
        self.reverse_doc_id_map = {}  # {doc_id: filename}
        self.stop_words = stop_words if stop_words else set()
        self.stemmer = nltk.stem.PorterStemmer()
        self.term_frequency = Counter()

    def add_document(self, doc_id, text):
        tokens = nltk.word_tokenize(text)

        normalized_tokens = []
        for token in tokens:
            lower = token.lower()
            if lower not in self.stop_words and lower.isalpha():
                stemmed = self.stemmer.stem(lower)
                normalized_tokens.append(stemmed)

        for term in normalized_tokens:
            if term not in self.index:
                self.index[term] = set()
            self.index[term].add(doc_id)
            self.term_frequency[term] += 1

    def get_document_count(self):
        return len(self.doc_id_map)
    
    def get_vocabulary_size(self):
        return len(self.index)
    
    def get_index_size_in_bytes(self):
        return sys.getsizeof(self.index)

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
        tokens = nltk.word_tokenize(query_str)
        parsed_tokens = []
        for token in tokens:
            lower = token.lower()
            if lower in ('and', 'or', '(', ')'):
                if lower == 'and':
                    parsed_tokens.append('&')
                elif lower == 'or':
                    parsed_tokens.append('|')
                else:
                    parsed_tokens.append(token)
            else:
                # This token is probably a term...
                if lower.isalpha():
                    stemmed = self.stemmer.stem(lower)
                    if stemmed in self.index:
                        parsed_tokens.append(f"set({list(self.index[stemmed])})")
                    else:
                        parsed_tokens.append("set()")
                else:
                    parsed_tokens.append("set()")

        expression = " ".join(parsed_tokens)
        try:
            results = eval(expression)
        except:
            print(f"[!] Could not parse query expression: {query_str}")
            results = set()
        return results

def build_inverted_index(directory_path, stop_words):
    print(f"[+] Building index from documents in '{directory_path}'...")
    inv_index = InvertedIndex(stop_words=stop_words)
    current_doc_id = 1

    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if filename.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            inv_index.doc_id_map[filename] = current_doc_id
            inv_index.reverse_doc_id_map[current_doc_id] = filename
            inv_index.add_document(doc_id=current_doc_id, text=text)
            current_doc_id += 1

    return inv_index

def display_top_n_terms(inv_index, n=10):
    print(f"\nTop {n} most frequent terms in the entire corpus:")
    for term, freq in inv_index.term_frequency.most_common(n):
        print(f"  {term}: {freq}")

def main():
    print_banner()
    stop_words_file = "stopwords.txt"
    stop_words = load_stopwords(stop_words_file)
    documents_dir = "documents"
    index_file_path = "saved_index.pkl"
    use_existing_index = False

    if use_existing_index and os.path.exists(index_file_path):
        print("[+] Loading existing index from file...")
        inv_index = InvertedIndex()
        inv_index.load(index_file_path)
    else:
        print("[+] Building a new index from the documents directory...")
        inv_index = build_inverted_index(documents_dir, stop_words=stop_words)
        print("[+] Saving the new index to disk...")
        inv_index.save(index_file_path)

    size_in_bytes = inv_index.get_index_size_in_bytes()
    size_in_mb = size_in_bytes / (1024 * 1024)
    print(f"[+] Inverted Index built/loaded. Size in memory: "
          f"{size_in_bytes} bytes ({size_in_mb:.2f} MB)")

    display_top_n_terms(inv_index, n=10)

    # Query and Retrieval
    while True:
        query_str = input("\nEnter a Boolean query (or type 'exit' to quit): ")
        if query_str.lower() == 'exit':
            break

        matching_doc_ids = inv_index.boolean_retrieve(query_str)
        if not matching_doc_ids:
            print("No documents matched your query.")
        else:
            matching_filenames = [inv_index.reverse_doc_id_map[doc_id] 
                                  for doc_id in sorted(matching_doc_ids)]
            print(f"Documents matching '{query_str}':")
            for fname in matching_filenames:
                print(f"  - {fname}")

if __name__ == "__main__":
    main()
