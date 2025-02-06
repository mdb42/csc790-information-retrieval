"""
CSC790-IR Homework 01
Query Parser
Matthew Branson
5 February 2025

This module isn't part of the graded assignment, but it provides interaction 
with the index for simple inspection. Read my journal linked in the readme if
you'd like to learn about the catastrophe that was my first version of this.
This one isn't as versatile, but it has a nonzero probability of bricking your
computer, which is an improvement over past iterations.

"""
import nltk

class QueryParser:
    """Parses a Boolean query expression and retrieves documents matching it."""
    def __init__(self, index, stopwords, stemmer, all_docs):
        self.index = index
        self.stopwords = stopwords
        self.stemmer = stemmer
        self.all_docs = all_docs
        self.tokens = []
        self.current = 0

    def parse(self, query):
        """Parse a Boolean query expression and return the set of matching document IDs.
        
        Args:
            query (str): A Boolean query expression.
            
        Returns:
            set: A set of document IDs that match the query expression.
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
        """Parse a Boolean expression."""
        
        left = self.parse_term()
        
        while self.current < len(self.tokens) and self.tokens[self.current].lower() == 'or':
            self.current += 1
            right = self.parse_term()
            left = left.union(right)
            
        return left

    def parse_term(self):
        """Parse a Boolean term."""
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
        """Parse a Boolean factor."""
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
        """Check if a token is a boolean operator.
        
        Args:
            token (str): A token from the query expression.

        Returns:
            bool: True if the token is a boolean operator, False otherwise.
        """
        return token.lower() in {'and', 'or', 'not'}

def boolean_retrieve(index, query_str):
    """Retrieve documents matching a Boolean query expression.

    Args:
        index (dict): An inverted index of the documents.
        query_str (str): A Boolean query expression.
    
    Returns:
        list: A list of document IDs that match the query expression.
    """
    stopwords = index.stopwords
    stemmer = nltk.stem.PorterStemmer()
    
    all_docs = set(index.doc_id_map.values())
    parser = QueryParser(index.index, stopwords, stemmer, all_docs)
    
    try:
        matching_doc_ids = parser.parse(query_str)
    except Exception as e:
        print(f"[!] Could not parse query expression '{query_str}': {e}")
        return []

    if not matching_doc_ids:
        return []
    
    # Compute total term frequency in each matching document
    doc_scores = {}
    
    for term in parser.tokens:  # Only include terms from the query
        stemmed_term = stemmer.stem(term)
        if stemmed_term in index.index:
            for doc_id, freq in index.index[stemmed_term].items():
                if doc_id in matching_doc_ids:  # Only consider matched docs
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + freq
    
    # Sort documents by highest total term frequency (descending)
    ranked_docs = sorted(doc_scores.keys(), key=lambda doc: doc_scores[doc], reverse=True)
    
    return ranked_docs  # Returns a sorted list instead of a set


def query_parser_demo(index):
    """A simple demonstration of the query parser.
    
    Args:
        index (InvertedIndex): An inverted index of the documents.
    """
    while True:
        query_str = input("\nEnter a Boolean query (or type 'exit' to quit): ")
        if query_str.lower() == 'exit':
            break 
        ranked_doc_ids = boolean_retrieve(index, query_str)
        
        if not ranked_doc_ids:
            print("No documents matched your query.")
        else:
            matching_filenames = [
                index.reverse_doc_id_map[doc_id] 
                for doc_id in ranked_doc_ids  # Now sorted by relevance!
            ]
            
            plural = "s" if len(matching_filenames) != 1 else ""
            print(f"\nFound {len(matching_filenames)} document{plural} matching '{query_str}':")
            
            # Then display the results as a comma-separated list
            results_text = ", ".join(matching_filenames)
            
            # If there are more than 4 results, break them into chunks
            if len(matching_filenames) > 4:
                # Break into chunks of 3 for display
                chunks = [matching_filenames[i:i+3] for i in range(0, len(matching_filenames), 3)]
                for chunk in chunks:
                    print("  " + ", ".join(chunk))
            else:
                # For fewer results, just show them all on one line
                print(f"  {results_text}")
