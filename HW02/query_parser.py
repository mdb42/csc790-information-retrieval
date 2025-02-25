"""
CSC 790 Information Retrieval
Homework 02: Building a Basic Search System
Author: Matthew Branson
Date: 02/05/2025
"""

import io
import nltk
import itertools
import argparse
import logging
from inverted_index import InvertedIndex

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#################################################################
# Utility Functions
#################################################################

def display_banner():
    """
    Displays a banner with the course information and my name.
    """
    logging.info("=================== CSC790-IR Homework 02 ===================")
    logging.info("First Name: Matthew")
    logging.info("Last Name : Branson")
    logging.info("=============================================================")

def initialize_nltk():
    """
    Initializes the NLTK library by ensuring the 'punkt' tokenizer is available.
    
    This function checks if the 'punkt' tokenizer is already downloaded. If it is not found,
    it downloads the 'punkt' tokenizer from the NLTK data repository.

    Raises:
        LookupError: If the 'punkt' tokenizer is not found and cannot be downloaded.
    """
    try:
        nltk.data.find('tokenizers/punkt/english.pickle')
    except LookupError:
        nltk.download('punkt')

#################################################################
# Query Parser
#################################################################

class QueryParser:
    """A boolean query parser that evaluates operators left-to-right with equal precedence.
    
    This class parses a boolean query expression and evaluates it using an inverted index.
    The query expression can contain the following operators: 'AND', 'OR', 'NOT', '(', ')'.
    The parser evaluates the operators left-to-right with equal precedence.

    Args:
        index (dict): An inverted index where keys are terms and values are sets of doc IDs.
        stopwords (set): A set of stopwords to ignore in the query.
        stemmer (nltk.stem.PorterStemmer): A stemmer to normalize terms in the query.
        all_docs (set): A set of all doc IDs in the index.
    
    Attributes:
        index (dict): An inverted index where keys are terms and values are sets of doc IDs.
        stopwords (set): A set of stopwords to ignore in the query.
        stemmer (nltk.stem.PorterStemmer): A stemmer to normalize terms in the query.
        all_docs (set): A set of all doc IDs in the index.
        tokens (list): A list of tokens extracted from the query.
        current (int): The current index in the tokens list being processed.
    """
    def __init__(self, index, stopwords, stemmer, all_docs):
        """Initializes the QueryParser with an inverted index, stopwords, stemmer, and all doc IDs.

        Args:
            index (dict): An inverted index where keys are terms and values are sets of doc IDs.
            stopwords (set): A set of stopwords to ignore in the query.
            stemmer (nltk.stem.PorterStemmer): A stemmer to normalize terms in the query.
            all_docs (set): A set of all doc IDs in the index.
        """
        self.index = index
        self.stopwords = stopwords
        self.stemmer = stemmer
        self.all_docs = all_docs
        self.tokens = []
        self.current = 0

    def parse(self, query):
        """Parses and evaluates a boolean query expression.

        Args:
            query (str): A boolean query expression.

        Returns:
            set: The set of doc IDs matching the query.
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
        """Parses an expression containing terms and operators.
        
        Returns:
            set: The set of doc IDs matching the expression.
        """
        left = self.parse_factor()
        while self.current < len(self.tokens):
            op = self.tokens[self.current].lower()
            if op not in {'and', 'or'}:
                break
            self.current += 1
            right = self.parse_factor()
            if op == 'and':
                left = left.intersection(right)
            else:
                left = left.union(right)
        return left

    def parse_factor(self):
        """Parses a factor containing a term, NOT operator, or sub-expression.

        Returns:
            set: The set of doc IDs matching the factor.
        """
        if self.current >= len(self.tokens):
            return set()

        token = self.tokens[self.current].lower()
        if token == 'not':
            self.current += 1
            result = self.parse_factor()
            return self.all_docs - result
        elif token == '(':
            self.current += 1
            result = self.parse_expression()
            if self.current < len(self.tokens) and self.tokens[self.current] == ')':
                self.current += 1
            return result
        elif self.is_operator(token):
            return set()
        else:
            self.current += 1
            term = self.stemmer.stem(token)
            return set(self.index.get(term, set()))

    def is_operator(self, token):
        """Checks if a token is a boolean operator.

        Args:
            token (str): A token to check.

        Returns:
            bool: True if the token is a boolean operator, False otherwise.
        """
        return token.lower() in {'and', 'or', 'not'}

#################################################################
# Query Processing
#################################################################

def generate_operator_combinations(terms):
    """Generates all possible operator combinations for a list of terms.

    Args:
        terms (list): A list of terms to combine.

    Returns:
        list: A list of all possible operator combinations for the terms.
    """
    n = len(terms)
    if n < 2:
        return [terms[0]] if n == 1 else []
    num_ops = n - 1
    op_combinations = itertools.product(['AND', 'OR'], repeat=num_ops)
    combinations = []
    for ops in op_combinations:
        query_parts = []
        for i in range(num_ops):
            query_parts.append(terms[i])
            query_parts.append(ops[i])
        query_parts.append(terms[-1])
        combinations.append(' '.join(query_parts))
    return combinations

def process_queries(index, queries_file, results_file):
    """Process boolean queries from a file and write the results to another file.

    Args:
        index (InvertedIndex): An inverted index.
        queries_file (str): The file containing boolean queries.
        results_file (str): The file to save query results.
    """
    try:
        with open(queries_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"Queries file '{queries_file}' not found.")
        return

    with open(results_file, 'w') as results:
        for query_num, query in enumerate(queries, 1):
            terms = query.split()
            if len(terms) < 1 or len(terms) > 4:
                logging.warning(f"Query {query_num} has invalid number of terms. Skipping.")
                continue

            combinations = generate_operator_combinations(terms)
            output = io.StringIO()
            output.write(f"\n================== User Query {query_num}: {query} ==================\n")
            
            for combo in combinations:
                matching_doc_ids = boolean_retrieve(index, combo)
                matching_filenames = [index.reverse_doc_id_map[doc_id] for doc_id in sorted(matching_doc_ids)]
                
                output.write(f"\n============= Results for: {combo} ===============\n")
                if matching_filenames:
                    for filename in matching_filenames:
                        output.write(f"{filename}\n")
                else:
                    output.write("No results found.\n")
                output.write("=" * 50)
            
            # Print to console
            print(output.getvalue())
            # Write to file
            results.write(output.getvalue())
            output.close()

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
        logging.error(f"Could not parse query expression '{query_str}': {e}")
        return set()

#################################################################
# Main Function - Build/Load and Query an Inverted Index
#################################################################

def main(documents_dir=None, stopwords_file=None,
         index_file=None, use_existing_index=False, use_parallel=True, 
         query_file=None, results_file=None):
    """
    Main function to build/load and query an inverted index from text documents.

    Args:
        documents_dir (str): The directory containing documents to index.
        stopwords_file (str): The file containing stopwords.
        index_file (str): The path to save/load the index.
        use_existing_index (bool): Whether to use an existing index if available.
        use_parallel (bool): Whether to use parallel processing for indexing.
        query_file (str): The file containing terms to query.
        results_file (str): The file to save query results.

    Raises:
        AttributeError: If the arguments are not provided and the script is run interactively.
    """
    if documents_dir is None or stopwords_file is None or index_file is None or query_file is None or results_file is None: 
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
        parser.add_argument('--query_file', default='queries.txt',
                          help='File containing boolean queries')
        parser.add_argument('--results_file', default='results.txt',
                          help='File to save query results')
        
        args = parser.parse_args()
        documents_dir = args.documents_dir
        stopwords_file = args.stopwords_file
        index_file = args.index_file
        use_existing_index = args.use_existing
        use_parallel = not args.no_parallel
        query_file = args.query_file
        results_file = args.results_file

    initialize_nltk()
    display_banner()
    stopwords = InvertedIndex.load_stopwords(stopwords_file)
    index = InvertedIndex(documents_dir=documents_dir, index_file=index_file,
                          use_existing_index=use_existing_index, use_parallel=use_parallel,
                          stopwords=stopwords)
    process_queries(index, query_file, results_file)

if __name__ == "__main__":
    main()