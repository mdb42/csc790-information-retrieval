"""
CSC 790 Information Retrieval
Homework 01: Building an Inverted Index - Support File
Author: Matthew Branson
Date: 02/05/2025
"""
import nltk

class QueryParser:
    """A simple boolean query parser for a document index."""
    def __init__(self, index, stopwords, stemmer, all_docs):
        self.index = index
        self.stopwords = stopwords
        self.stemmer = stemmer
        self.all_docs = all_docs
        self.tokens = []
        self.current = 0

    def parse(self, query):
        """Parse a boolean query and return the set of matching doc IDs.
        
        Args:
            query (str): A boolean query string.
        """
        self.tokens = []
        for word in nltk.word_tokenize(query):
            lower = word.lower()
            if lower in {'and', 'or', 'not', '(', ')'}:
                self.tokens.append(lower)
            elif word.isalpha() and lower not in self.stopwords:
                self.tokens.append(self.stemmer.stem(lower))
        self.current = 0
        return self.parse_expression()

    def parse_expression(self):
        """Parse an expression, which is a series of terms combined by AND and OR operators."""
        left = self.parse_term()
        
        while self.current < len(self.tokens) and self.tokens[self.current].lower() == 'or':
            self.current += 1
            right = self.parse_term()
            left = left.union(right)
            
        return left

    def parse_term(self):
        """Parse a term, which may be a single word or a NOT operator followed by a word."""
        left = self.parse_factor()
        
        while (self.current < len(self.tokens) and 
               (self.tokens[self.current].lower() == 'and' or 
                (self.peek().isalpha() and not self.is_operator(self.peek())))):
            if self.tokens[self.current].lower() == 'and':
                self.current += 1
            right = self.parse_factor()
            left = left.intersection(right)
            
        return left

    def parse_factor(self):
        """Parse a factor, which may be a word, a NOT operator followed by a word, or a parenthesized expression."""
        if self.current >= len(self.tokens):
            return set()

        token = self.tokens[self.current].lower()
        
        if token == 'not':
            self.current += 1
            result = self.parse_factor()
            return self.all_docs - result
            
        if token == '(':
            self.current += 1
            result = self.parse_expression()
            if self.current < len(self.tokens) and self.tokens[self.current] == ')':
                self.current += 1
            return result
            
        if self.is_operator(token):
            return set()
            
        self.current += 1
        term = self.stemmer.stem(token)
        return set(self.index.get(term, set()))

    def peek(self):
        """Look at next token without consuming it."""
        return self.tokens[self.current] if self.current < len(self.tokens) else ''

    def is_operator(self, token):
        """Check if a token is a boolean operator."""
        return token.lower() in {'and', 'or', 'not'}

def boolean_retrieve(index, query_str):
    """Retrieve documents matching a boolean query.

    Args:
        index (InvertedIndex): An inverted index.
        query_str (str): A boolean query string.

    Returns:
        set: The set of doc IDs matching the query.
    """
    stopwords = index.stopwords
    stemmer = nltk.stem.PorterStemmer()
    
    all_docs = set(index.doc_id_map.values())
    parser = QueryParser(index.index, stopwords, stemmer, all_docs)
    try:
        return parser.parse(query_str)
    except Exception as e:
        print(f"[!] Could not parse query expression '{query_str}': {e}")
        return set()

def query_parser_demo(index):
    """A simple command-line interface for the boolean query parser.

    Args:
        index (InvertedIndex): An inverted index.
    """
    while True:
        query_str = input("\nEnter a Boolean query (or type 'exit' to quit): ")
        if query_str.lower() == 'exit':
            break 
        matching_doc_ids = boolean_retrieve(index, query_str)
        if not matching_doc_ids:
            print("No documents matched your query.")
        else:
            matching_filenames = [
                index.reverse_doc_id_map[doc_id] 
                for doc_id in sorted(matching_doc_ids)
            ]
            plural = "s" if len(matching_filenames) != 1 else ""
            print(f"\nFound {len(matching_filenames)} document{plural} matching '{query_str}':")
            
            results_text = ", ".join(matching_filenames)
            
            if len(matching_filenames) > 4:
                chunks = [matching_filenames[i:i+3] for i in range(0, len(matching_filenames), 3)]
                for chunk in chunks:
                    print("  " + ", ".join(chunk))
            else:
                print(f"  {results_text}")