# main.py
import os
import argparse
from src.performance_monitoring import Profiler
from src.index import IndexFactory
from src.vsm import VSMFactory
from src.utils import display_detailed_statistics, display_dependencies

def parse_arguments():
    from src.utils import load_config, save_config
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
    parser.add_argument('--index_mode', choices=['auto', 'standard', 'parallel'], default=config['index_mode'],
                        help=f"Index implementation to use (default: {config['index_mode']})")
    parser.add_argument('--vsm_mode', 
                        choices=['auto', 'standard', 'parallel', 'hybrid', 'sparse', 'experimental'], 
                        default=config['vsm_mode'],
                        help=f"VSM implementation to use (default: {config['vsm_mode']})")
    parser.add_argument('--parallel_index_threshold', type=int, default=config['parallel_index_threshold'],
                       help=f"Document threshold for parallel index (default: {config['parallel_index_threshold']})")
    parser.add_argument('--hybrid_vsm_threshold', type=int, default=config['hybrid_vsm_threshold'],
                       help=f"Document threshold for hybrid VSM (default: {config['hybrid_vsm_threshold']})")
    parser.add_argument('--chunk_size', type=int, default=config.get('chunk_size', 1000),
                       help=f"Chunk size for experimental VSM (default: {config.get('chunk_size', 1000)})")
    parser.add_argument('--export_json', default=None,
                        help='Export index to JSON file (specify filename)')
    parser.add_argument('--stats', action='store_true',
                        help='Display detailed index statistics')
    parser.add_argument('--check_deps', action='store_true',
                        help='Check and display available dependencies')
    parser.add_argument('--save_config', action='store_true',
                        help='Save current settings to config file')
    
    args = parser.parse_args()
    
    if args.config != 'config.json' and os.path.exists(args.config):
        new_config = load_config(args.config)
        
        parser_defaults = {action.dest: action.default for action in parser._actions}
        
        import sys
        provided_args = set()
        for i, arg in enumerate(sys.argv[1:]):
            if arg.startswith('--') and '=' not in arg and i+1 < len(sys.argv[1:]) and not sys.argv[i+2].startswith('--'):
                provided_args.add(arg[2:])
            elif arg.startswith('--') and '=' in arg:
                provided_args.add(arg.split('=')[0][2:])
        
        for key, value in new_config.items():
            if key in parser_defaults and key not in provided_args:
                setattr(args, key, value)
    
    if args.save_config:
        current_config = {
            'documents_dir': args.documents_dir,
            'stopwords_file': args.stopwords_file,
            'special_chars_file': args.special_chars_file,
            'index_file': args.index_file,
            'index_mode': args.index_mode,
            'vsm_mode': args.vsm_mode,
            'parallel_index_threshold': args.parallel_index_threshold,
            'hybrid_vsm_threshold': args.hybrid_vsm_threshold,
            'experimental_vsm_threshold': args.experimental_vsm_threshold,
            'chunk_size': args.chunk_size
        }
        save_config(current_config, args.config)
        print(f"Configuration saved to {args.config}")
    
    return args

def get_valid_int_input(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
            print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def display_banner():
    print("=" * 55)
    print("=" * 16 + " CSC790-IR Homework 03 " + "=" * 16)
    print("First Name: Matthew")
    print("Last Name : Branson")
    print("=" * 55)

def display_vocabulary_statistics(index):
    print(f"The number of unique words is: {index.vocab_size}")
    print("The top 10 most frequent words are:")
    # Numbered display requirement makes this O(k) but the underlying heapq operation is O(n log k)
    for i, (term, freq) in enumerate(index.get_most_frequent_terms(n=10), 1):
        print(f"    {i}. {term} ({freq:,})")
    print("=" * 56)

def display_similar_documents(vsm, k):
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
    args = parse_arguments()
    
    if args.check_deps:
        display_dependencies()

    k = get_valid_int_input("\nEnter the number of top similar document pairs (k): ")
    display_banner()
    
    profiler = Profiler()
    profiler.start_global_timer()

    if args.use_existing and os.path.exists(args.index_file):
        with profiler.timer("Index Loading"):
            from src.index.standard_index import StandardIndex
            try:
                index = StandardIndex.load(args.index_file)
            except Exception as e:
                print(f"Error loading index from {args.index_file}: {e}")
                print("Building new index instead...")
                args.use_existing = False
    
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

        with profiler.timer("Index Saving"):
            index.save(args.index_file)

    if args.export_json:
        with profiler.timer("JSON Export"):
            index.export_json(args.export_json)
            print(f"Index exported to {args.export_json}")

    vsm = VSMFactory.create_vsm(
        index, 
        args.vsm_mode, 
        profiler,
        hybrid_threshold=args.hybrid_vsm_threshold
    )

    display_vocabulary_statistics(index)

    if args.stats:
        display_detailed_statistics(index)

    display_similar_documents(vsm, k)

    print("\n"+profiler.generate_report(
        doc_count=index.doc_count,
        vocab_size=index.vocab_size,
        filename="performance.log"
    ))

if __name__ == "__main__":
    main()