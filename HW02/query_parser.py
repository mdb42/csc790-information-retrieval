"""
CSC 790 Information Retrieval
Homework 02: Building a Basic Search System
Author: Matthew Branson
Date: 02/26/2025

Description:
This program is a basic search system that uses a modified version of the inverted index from
Homework 1 to retrieve mathching documents given a list of user queries. The program reads boolean
queries from a file and provides the results for all possible combinations of the query terms using
AND and OR operators. Aside from using itertools to combinatorially expand the query terms with 
AND/OR operators, the retrieval implementation is otherwise only slightly modified from the prototype 
query_parser.py file that I offered for demonstration in Homework 1.
Additionally, just as an exercise for my own understanding, I have implemented tf-idf scoring for
term weighting and cosine similarity for ranking the retrieved documents.

Note: I actually only just now saw that you are providing a list of special characters, and so the
usage of that file is something I just added in the last ten minutes before submission. I hope that
doesn't break anything!
"""

import io
import nltk
import itertools
import argparse
import logging
from inverted_index import InvertedIndex
import math
from collections import Counter

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

def load_special_chars(special_chars_file):
    """
    Load special characters to be removed from text.
    
    Args:
        special_chars_file (str): Path to the file containing special characters to remove.
        
    Returns:
        set: A set of special characters to remove during processing.
    """
    logging.info(f"Loading special characters from '{special_chars_file}'...")
    special_chars = set()
    try:
        with open(special_chars_file, encoding="utf-8") as file:
            special_chars = {line.strip() for line in file if line.strip()}
        logging.info(f"Loaded {len(special_chars)} special characters to remove.")
    except FileNotFoundError:
        logging.error(f"Special characters file not found: {special_chars_file}")
    except Exception as e:
        logging.error(f"Error loading special characters: {str(e)}")
    return special_chars

#################################################################
# Query Parser
#################################################################

class QueryParser:
    """
    Recursive descent parser for boolean queries with NOT, AND, OR operators.
    
    Implements operator precedence: NOT > AND > OR. Parentheses can override precedence.
    Query processing matches the inverted index's tokenization/stemming rules.
    
    Attributes:
        index (InvertedIndex): Prebuilt search index
        stopwords (set): Terms to ignore in queries
        stemmer (SnowballStemmer): Stemmer matching index's preprocessing
        all_docs (set): Complete document set for NOT operations
        tokens (list): Tokenized query components
        current (int): Current parse position in tokens
    """
    def __init__(self, index, stopwords, stemmer, all_docs):
        """Initialize parser with index configuration and document universe."""
        self.index = index
        self.stopwords = stopwords
        self.stemmer = stemmer
        self.all_docs = all_docs
        self.tokens = []
        self.current = 0

    def parse(self, query):
        """
        Main parse entry point. Returns matching document IDs.
        
        Args:
            query: Raw input string with boolean operators
            
        Returns:
            set: Document IDs satisfying the query
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
        """"
        Parses an expression in the query string and returns the set of document IDs that match it.
        
        Returns:
            set: A set of document IDs that match the expression.
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
        """
        Parses a factor in the query string and returns the set of document IDs that match it.

        Returns:
            set: A set of document IDs that match the factor.
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
            if term in self.index.index:
                return set(self.index.index[term].keys())
            return set()

    def is_operator(self, token):
        """
        Checks if a token is a boolean operator (AND, OR, NOT).

        Args:
            token (str): The token to check.

        Returns:
            bool: True if the token is a boolean operator, False otherwise.
        """
        return token.lower() in {'and', 'or', 'not'}

#################################################################
# Query Processing
#################################################################

def generate_operator_combinations(terms):
    """
    Generates all possible combinations of AND/OR operators for a list of query terms.
    
    Args:
        terms: List of query terms (non-empty, 2-4 terms)
        
    Returns:
        list: All valid boolean combinations for term analysis
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

def process_queries(index, queries_file, results_file, verbose=False, special_chars=None):
    """
    Process query file and write ranked results.
    
    Implementation Notes:
    1. For each query:
        a. Generate all AND/OR combinations
        b. Boolean retrieval for each combination
        c. Rank results using TF-IDF cosine similarity
    2. Write results with/without scores based on verbose flag
    
    Args:
        index: Prebuilt InvertedIndex
        queries_file: Path to queries (one per line)
        results_file: Output path for results
        verbose: Show ranking scores when True
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
            
            # Process query terms once (for ranking)
            processed_terms = process_query_terms(query, index.stopwords, index.stemmer, special_chars)
            
            for combo in combinations:
                # Get matching documents using Boolean retrieval
                matching_doc_ids = boolean_retrieve(index, combo)
                
                output.write(f"\n============= Results for: {combo} ===============\n")
                
                if not matching_doc_ids:
                    output.write("No results found.\n")
                else:
                    # Rank results using tf-idf when we have matches
                    doc_scores = []
                    if processed_terms:
                        # Get tf-idf scores for all documents in the Boolean result set
                        all_scores = calculate_document_rankings(index, processed_terms)
                        # Filter to only include docs from the Boolean result
                        doc_scores = [(doc_id, score) for doc_id, score in all_scores 
                                     if doc_id in matching_doc_ids]
                    else:
                        # If no processed terms, just use the Boolean results with no scoring
                        doc_scores = [(doc_id, 0.0) for doc_id in sorted(matching_doc_ids)]
                    
                    # Output results with scores depending on verbosity
                    if verbose:
                        for rank, (doc_id, score) in enumerate(doc_scores, 1):
                            filename = index.reverse_doc_id_map[doc_id]
                            output.write(f"{rank}. {filename} (Score: {score:.4f})\n")
                    else:
                        # Simple output format as required by the assignment
                        for doc_id, _ in doc_scores:
                            filename = index.reverse_doc_id_map[doc_id]
                            output.write(f"{filename}\n")
                
                output.write("=" * 50)
            
            # Print to console
            print(output.getvalue())
            # Write to file, since it may overflow the console
            results.write(output.getvalue())
        logging.info(f"Query results written to '{results_file}'.")

def process_query_terms(query, stopwords, stemmer, special_chars=None):
    """
    Processes query terms by tokenizing, normalizing, and stemming them.
    
    Args:
        query (str): The query string to process.
        stopwords (set): A set of stopwords to ignore.
        stemmer (nltk.stem.SnowballStemmer): A stemmer to use for term normalization.
        special_chars (set): A set of special characters to remove.
    
    Returns:
        list: A list of processed query terms.
    """
    # Remove special characters from query if specified
    if special_chars:
        for char in special_chars:
            query = query.replace(char, ' ')
    
    tokens = nltk.word_tokenize(query)
    processed_terms = []
    for word in tokens:
        lower_word = word.lower()
        if lower_word.isalpha() and lower_word not in stopwords:
            processed_terms.append(stemmer.stem(lower_word))
    return processed_terms

def boolean_retrieve(index, query_str):
    """
    Retrieves documents that match a boolean query expression.
    
    Args:
        index (InvertedIndex): The inverted index to use for query processing.
        query_str (str): The boolean query expression to evaluate.
        
    Returns:
        set: A set of document IDs that match the query expression.
    """
    stopwords = index.stopwords
    stemmer = index.stemmer
    all_docs = set(index.doc_id_map.values())
    parser = QueryParser(index, stopwords, stemmer, all_docs)
    try:
        return parser.parse(query_str)
    except Exception as e:
        logging.error(f"Could not parse query expression '{query_str}': {e}")
        return set()

def calculate_document_rankings(index, query_terms, use_cosine=True):
    """
    Compute TF-IDF cosine similarity scores.
    
    Implementation Notes:
    - IDF calculated as log(N/df) 
    - Query vector uses TF-IDF weights (count * IDF)
    - Document vectors normalized by L2 norm
    - Missing terms contribute zero to vectors
    
    Args:
        index: InvertedIndex with term frequencies
        query_terms: Stemmed, filtered terms from query
        candidate_docs: Pre-filtered documents from boolean retrieval
        
    Returns:
        list: (doc_id, score) tuples sorted descending
    """

    # Total number of documents in collection (N in the IDF formula)
    n_docs = len(index.doc_id_map)
    unique_terms = set(query_terms)
    idf = {}

    # Calculate IDF for each unique term in the query
    for term in unique_terms:
        df = len(index.index.get(term, {}))  # document frequency
        idf[term] = math.log10(n_docs / df) if df else 0.0

    # Find all documents containing any query term to avoid scanning the entire collection
    candidate_docs = set()
    for term in query_terms:
        candidate_docs.update(index.index.get(term, {}).keys())

    # Compute query vector weights using TF-IDF
    query_vec = Counter(query_terms)  # term frequencies in query
    query_weights = {term: query_vec[term] * idf.get(term, 0.0) for term in unique_terms}

    if use_cosine:
        # Precompute document vector norms for cosine similarity
        doc_norms = {}
        for doc_id in candidate_docs:
            doc_norm = 0
            for term in unique_terms:
                if term in index.index and doc_id in index.index[term]:
                    # Get tf-idf weight for this term in this document
                    weight = index.index[term][doc_id] * idf.get(term, 0.0)
                    doc_norm += weight ** 2  # square the weight and accumulate
            doc_norms[doc_id] = math.sqrt(doc_norm)  # square root of sum of squares

        # Compute query vector norm
        query_norm = math.sqrt(sum(w ** 2 for w in query_weights.values()))

    # Compute scores for each candidate document
    doc_scores = []
    for doc_id in candidate_docs:
        # Calculate dot product between query and document vectors
        dot_product = sum(
            index.index.get(term, {}).get(doc_id, 0)  # tf of term in document
            * idf.get(term, 0.0)                      # idf of term
            * query_weights[term]                     # weight in query vector
            for term in unique_terms
        )

        if use_cosine and doc_norms[doc_id] > 0 and query_norm > 0:
            # Measure the cosine of the angle between vectors q and d
            score = dot_product / (doc_norms[doc_id] * query_norm)
        else:
            # Without normalization, use the dot product
            # This is just basic tf-idf scoring without accounting for document length
            score = dot_product

        doc_scores.append((doc_id, score))

    # Sort by score (descending) and then by doc_id (ascending)
    doc_scores.sort(key=lambda x: (-x[1], x[0]))
    return doc_scores

#################################################################
# Main Function - Build/Load and Query an Inverted Index
#################################################################


def main(documents_dir=None, stopwords_file=None, special_chars_file=None,
         index_file=None, use_existing_index=False, use_parallel=True, 
         query_file=None, results_file=None, verbose=False):
    """
    Main function to build/load an inverted index and process boolean queries.
    
    Args:
        documents_dir (str): Directory containing documents to index.
        stopwords_file (str): File containing stopwords.
        index_file (str): Path to save/load the index.
        use_existing_index (bool): Use existing index if available.
        use_parallel (bool): Enable parallel processing.
        query_file (str): File containing boolean queries.
        results_file (str): File to save query results.
        verbose (bool): Enable verbose output with scores.
    """
    if documents_dir is None or stopwords_file is None or index_file is None or query_file is None or results_file is None or verbose is None or special_chars_file is None:
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
        parser.add_argument('--verbose', '-v', action='store_true',
                   help='Enable verbose output with scores')
        parser.add_argument('--special_chars_file', default='special-chars.txt',
                        help='File containing special characters to retain')
        
        args = parser.parse_args()
        documents_dir = args.documents_dir
        stopwords_file = args.stopwords_file
        index_file = args.index_file
        use_existing_index = args.use_existing
        use_parallel = not args.no_parallel
        query_file = args.query_file
        results_file = args.results_file
        special_chars = load_special_chars(special_chars_file) if special_chars_file else set()

    initialize_nltk()
    display_banner()
    stopwords = InvertedIndex.load_stopwords(stopwords_file)
    index = InvertedIndex(documents_dir=documents_dir, index_file=index_file,
                          use_existing_index=use_existing_index, use_parallel=use_parallel,
                          stopwords=stopwords, special_chars=special_chars)
    input("Press Enter to continue...") # Since the results may likely overflow your console
    process_queries(index, query_file, results_file, verbose=args.verbose, special_chars=special_chars)

if __name__ == "__main__":
    main()