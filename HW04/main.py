"""
CSC 790 Information Retrieval

Homework 4: Binary Independence Model (BIM)
Author: Matthew Branson

Precisely powerful probabilistic paradigms: providing proficient, performant
passage pursuit via painstakingly parameterized probability patterns. Purposefully
parsing prose, picking perfect pieces, presenting prime publications promptly. 
Previously popular patterns pale; probabilistic processing predominates!

This project is brought to you by the letter 'P' and the number 4.
"""

import os
import argparse
import sys
from src.profiler import Profiler
from src.index import IndexFactory
from src.bim import RetrievalBIM

from src.utils import ( display_vocabulary_statistics,
                        display_detailed_statistics,
                        load_config)

def parse_arguments():
    """
    Parse command-line arguments and integrate with configuration file settings.
    
    Returns:
        argparse.Namespace: The parsed command-line arguments with config file integration
    """
    config = load_config()
    parser = argparse.ArgumentParser(description='Vector space model for document similarity.')

    # General Options
    parser.add_argument('--config', default='config.json', 
                        help='Path to configuration file')
    parser.add_argument('--query_files', nargs='+', default=config['query_files'],
                    help=f"Files containing queries to evaluate (default: {config['query_files']})")
    parser.add_argument('--labels_files', nargs='+', default=config['labels_files'],
                        help=f"Files containing relevance labels (default: {config['labels_files']})")
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
    parser.add_argument('--export_json', default=None,
                        help='Export index to JSON file (specify filename)')
    parser.add_argument('--stats', action='store_true',
                        help='Display detailed index statistics')
    
    # Indexing Options
    parser.add_argument('--index_mode', choices=['auto', 'standard', 'parallel'], default=config['index_mode'],
                        help=f"Index implementation to use (default: {config['index_mode']})")    
    parser.add_argument('--parallel_index_threshold', type=int, default=config['parallel_index_threshold'],
                       help=f"Document threshold for parallel index (default: {config['parallel_index_threshold']})")
    
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

def display_banner():
    """Display required banner."""
    print("=" * 55)
    print("=" * 16 + " CSC790-IR Homework 04 " + "=" * 16)
    print("First Name: Matthew")
    print("Last Name : Branson")
    print("=" * 55)

def process_queries(model, query_files, label_files, profiler):
    """
    Process queries from files and display results.
    
    Args:
        model: Binary Independence Model to use for scoring
        query_files (list): List of paths to query files
        label_files (list): List of paths to relevance label files
        profiler (Profiler): Performance profiler
    """
    if isinstance(query_files, str):
        query_files = [query_files]
    if isinstance(label_files, str):
        label_files = [label_files]
    
    with profiler.timer("Total Query Processing"):
        for i, (query_file, label_file) in enumerate(zip(query_files, label_files), 1):
            # Load relevance labels for this query
            model.load_relevance_labels(label_file)
            
            try:
                # Read query from file
                with open(query_file, 'r') as f:
                    query = f.read().strip()
                
                if not query:
                    profiler.log_message(f"Warning: Empty query in file {query_file}")
                    continue
                
                # Process query and get results
                with profiler.timer(f"Query {i} Total"):
                    results = model.search(query)
                
                # Display results
                print("\n"+"="*19+f" Query {i} " +"="*24)
                if results:
                    for doc_id, score in results:
                        # Get relevance label
                        label = model.get_relevance_label(doc_id)
                        print(f"RSV{{ {doc_id} }} = {score:.2f}\t{label}")
                else:
                    print("No results found")
                    
            except FileNotFoundError:
                profiler.log_message(f"Error: Could not find file {query_file}")
            except Exception as e:
                profiler.log_message(f"Error processing query {i}: {e}")

def main():
    """
    Main function to run the vector space model analysis.
    """
    args = parse_arguments()

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

    # Create BIM retrieval model
    model = RetrievalBIM(index, profiler=profiler)    

    # Process queries and display results
    process_queries(model, args.query_files, args.labels_files, profiler)

    # Display detailed index statistics if requested
    if args.stats:
        display_vocabulary_statistics(index)
        display_detailed_statistics(index)

    # Generate and display performance report
    print("\n" + profiler.generate_report(
        doc_count=index.doc_count,
        vocab_size=index.vocab_size,
        filename="performance.log"
    ))

if __name__ == "__main__":
    main()