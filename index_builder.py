import os
import sys
import pickle
import nltk

class InvertedIndex:
    def __init__(self, stop_words=None):

        try:
            # Test if 'punkt_tab' is already available
            nltk.data.find('tokenizers/punkt_tab/english.pickle')
        except LookupError:
            nltk.download('punkt_tab')

        if not os.path.exists('documents'):
            os.makedirs('documents')
        
        self.index = {}             # {term: set_of_docIDs}
        self.doc_id_map = {}        # {filename: doc_id}
        self.reverse_doc_id_map = {}# {doc_id: filename} for retrieval
        self.stop_words = stop_words if stop_words else set()
        self.stemmer = nltk.stem.PorterStemmer()

    def add_document(self, doc_id, text):

        # 1) Tokenize
        tokens = nltk.word_tokenize(text)

        # 2) Normalize & filter:
        #    a) Case-fold
        #    b) Remove stop words
        #    c) Stem
        normalized_tokens = []
        for token in tokens:
            lower = token.lower()
            if lower not in self.stop_words and lower.isalpha():
                stemmed = self.stemmer.stem(lower)
                normalized_tokens.append(stemmed)

        # 3) Insert into index
        #    For each term, add doc_id to the postings set
        for term in normalized_tokens:
            if term not in self.index:
                self.index[term] = set()
            self.index[term].add(doc_id)
    
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
                'stop_words': self.stop_words
            }, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.index = data['index']
            self.doc_id_map = data['doc_id_map']
            self.reverse_doc_id_map = data['reverse_doc_id_map']
            self.stop_words = data['stop_words']

    def boolean_retrieve(self, query_str):
        # Tokenize the query
        tokens = nltk.word_tokenize(query_str)

        parsed_tokens = []
        for token in tokens:
            lower = token.lower()
            if lower in ('and', 'or', '(', ')'):
                # Boolean operators or parentheses
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
                    # Replace the term with a Python set expression
                    if stemmed in self.index:
                        parsed_tokens.append(f"set({list(self.index[stemmed])})")
                    else:
                        # Term not in index -> empty set
                        parsed_tokens.append("set()")
                else:
                    # Non-alphabetic tokens (numbers, punctuation, etc.) => empty
                    parsed_tokens.append("set()")

        # Join everything back into a single expression
        expression = " ".join(parsed_tokens)
        # Safely evaluate the expression
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
            print(f"  - Processing '{filename}'...")
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            inv_index.doc_id_map[filename] = current_doc_id
            inv_index.reverse_doc_id_map[current_doc_id] = filename
            inv_index.add_document(doc_id=current_doc_id, text=text)
            current_doc_id += 1

    return inv_index

def main():
    """
      1) Build or load the index
      2) Show index size
      3) Prompt user for queries
      4) Return matching documents
    """
    # Directory containing your documents
    documents_dir = "documents"

    # Just a few common stop words. Will find a better option later.
    stop_words = {"the", "a", "for", "to", "is", "are", "in", "of", "and"}

    # Decide whether to build or load from file
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

    # Display the size of the index
    size_in_bytes = inv_index.get_index_size_in_bytes()
    print(f"[+] Inverted Index built/loaded. Size in memory: {size_in_bytes} bytes")

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