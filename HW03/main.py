"""
CSC 790 Information Retrieval
Homework 3: Vector Space Model for Document Similarity
Author: Matthew Branson
Date: March 14, 2025

This program implements a Vector Space Model (VSM) for finding similar documents
in a corpus. It features multiple optimized implementations that are automatically
selected based on document volume and available system resources.
"""

import os
import argparse
import sys
from src.profiler import Profiler
from src.index import IndexFactory
from src.vsm import VSMFactory
from src.utils import display_detailed_statistics, display_dependencies, load_config


def parse_arguments():
    """
    Parse command-line arguments and integrate with configuration file settings.
    
    Returns:
        argparse.Namespace: The parsed command-line arguments with config file integration
    """
    config = load_config()
    parser = argparse.ArgumentParser(description='Vector space model for document similarity.')
    parser.add_argument('--config', default='config.json', 
                        help='Path to configuration file')
    parser.add_argument('--documents_dir', default=config['documents_dir'], 
                        help=f"Directory containing documents to index (default: {config['documents_dir']})")
    parser.add_argument('--stopwords_file', default=config['stopwords_file'], 
                        help=f"File containing stopwords (default: {config['stopwords_file']})")
    parser.add_argument('--special_chars_file', default=config['special_chars_file'], 
                        help=f"File containing special characters to remove (default: {config['special_chars_file']})")
    parser.add_argument('--index_file', default=config['index_file'], 
                        help=f"Path to save/load the index (default: {config['index_file']})")
    parser.add_argument('--use_existing', action='store_true', 
                        help='Use existing index if available')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Number of top similar document pairs to display (if not provided, will prompt)')
    parser.add_argument('--index_mode', choices=['auto', 'standard', 'parallel'], default=config['index_mode'],
                        help=f"Index implementation to use (default: {config['index_mode']})")
    parser.add_argument('--vsm_mode', 
                        choices=['auto', 'standard', 'parallel', 'sparse'], 
                        default=config['vsm_mode'],
                        help=f"VSM implementation to use (default: {config['vsm_mode']})")
    parser.add_argument('--parallel_index_threshold', type=int, default=config['parallel_index_threshold'],
                       help=f"Document threshold for parallel index (default: {config['parallel_index_threshold']})")
    parser.add_argument('--parallel_vsm_threshold', type=int, default=config['parallel_vsm_threshold'],
                       help=f"Document threshold for Parallel VSM (default: {config['parallel_vsm_threshold']})")
    parser.add_argument('--parallelize_weights', action='store_true',
                       help='Parallelize weight computation in ParallelVSM (for benchmarking)')
    parser.add_argument('--export_json', default=None,
                        help='Export index to JSON file (specify filename)')
    parser.add_argument('--stats', action='store_true',
                        help='Display detailed index statistics')
    parser.add_argument('--check_deps', action='store_true',
                        help='Check and display available dependencies')
    
    args = parser.parse_args()
    
    # If a non-default config file is specified and exists, load it and apply values
    # for any parameters not explicitly provided on the command line
    if args.config != 'config.json' and os.path.exists(args.config):
        new_config = load_config(args.config)
        
        # Get default values from argparse
        parser_defaults = {action.dest: action.default for action in parser._actions}
        
        # Determine which arguments were explicitly provided on the command line
        provided_args = set()
        for i, arg in enumerate(sys.argv[1:]):
            if arg.startswith('--') and '=' not in arg and i+1 < len(sys.argv[1:]) and not sys.argv[i+2].startswith('--'):
                provided_args.add(arg[2:])
            elif arg.startswith('--') and '=' in arg:
                provided_args.add(arg.split('=')[0][2:])
        
        # Apply config file values only for arguments not explicitly provided
        for key, value in new_config.items():
            if key in parser_defaults and key not in provided_args:
                setattr(args, key, value)
    
    return args

def get_valid_int_input(prompt):
    """
    Prompt the user for a positive integer and validate the input.
    
    Args:
        prompt (str): The prompt to display to the user
        
    Returns:
        int: A validated positive integer
    """
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
            print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def display_banner():
    """Display required banner."""
    print("=" * 55)
    print("=" * 16 + " CSC790-IR Homework 03 " + "=" * 16)
    print("First Name: Matthew")
    print("Last Name : Branson")
    print("=" * 55)

def display_vocabulary_statistics(index):
    """
    Display vocabulary statistics including count and most frequent terms.
    
    Args:
        index: The document index object with vocabulary information
    """
    print(f"The number of unique words is: {index.vocab_size}")
    print("The top 10 most frequent words are:")
    for i, (term, freq) in enumerate(index.get_most_frequent_terms(n=10), 1):
        print(f"    {i}. {term} ({freq:,})")
    print("=" * 55)

def display_similar_documents(vsm, k):
    """
    Display the top k similar document pairs using different weighting schemes.
    
    Args:
        vsm: The vector space model object
        k (int): The number of top similar document pairs to display
    """
    print("The top k closest documents are:")
    weighting_schemes = [
        ("1. Using tf", "tf"),
        ("2. Using tfidf", "tfidf"),
        ("3. Using wfidf", "sublinear")
    ]
    
    for label, weighting in weighting_schemes:
        print(f"{label}:")
        similar_docs = vsm.find_similar_documents(k=k, weighting=weighting)
        if not similar_docs:
            print("    No similar document pairs found.")
        else:
            for doc1, doc2, sim in similar_docs:
                print(f"    {doc1}, {doc2} with similarity of {sim:.2f}")

def main():
    """
    Main function to run the vector space model analysis.
    """
    args = parse_arguments()
    
    # Check dependencies if requested
    if args.check_deps:
        display_dependencies()

    # Use provided top_k argument or prompt the user if not provided
    if args.top_k is None:
        k = get_valid_int_input("\nEnter the number of top similar document pairs (k): ")
    else:
        k = args.top_k

    display_banner()
    
    # Initialize performance profiler
    profiler = Profiler()
    profiler.start_global_timer()

    # Try to load existing index if requested
    if args.use_existing and os.path.exists(args.index_file):
        with profiler.timer("Index Loading"):
            from src.index.standard_index import StandardIndex
            try:
                index = StandardIndex.load(args.index_file)
            except Exception as e:
                print(f"Error loading index from {args.index_file}: {e}")
                print("Building new index instead...")
                args.use_existing = False
    
    # Build a new index if needed
    if not args.use_existing or not os.path.exists(args.index_file):
        index = IndexFactory.create_index(
            documents_dir=args.documents_dir,
            stopwords_file=args.stopwords_file,
            special_chars_file=args.special_chars_file,
            profiler=profiler,
            mode=args.index_mode,
            parallel_threshold=args.parallel_index_threshold
        )
        index.build_index()

        # Save the index for future use
        with profiler.timer("Index Saving"):
            index.save(args.index_file)

    # Export index to JSON if requested
    if args.export_json:
        with profiler.timer("JSON Export"):
            index.export_json(args.export_json)
            print(f"Index exported to {args.export_json}")

    # Create vector space model
    vsm = VSMFactory.create_vsm(
        index, 
        args.vsm_mode, 
        profiler,
        parallel_threshold=args.parallel_vsm_threshold,
        parallelize_weights=args.parallelize_weights
    )

    # Display statistics
    display_vocabulary_statistics(index)

    if args.stats:
        display_detailed_statistics(index)

    # Display similar document pairs
    display_similar_documents(vsm, k)

    # Generate and display performance report
    print("\n" + profiler.generate_report(
        doc_count=index.doc_count,
        vocab_size=index.vocab_size,
        filename="performance.log"
    ))

if __name__ == "__main__":
    main()