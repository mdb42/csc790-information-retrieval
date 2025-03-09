# main.py
import os
import argparse
from src.performance_monitoring import Profiler
from src.index.factory import IndexFactory
from src.vsm.factory import VSMFactory

def parse_arguments():
    parser = argparse.ArgumentParser(description='Vector space model for document similarity.')
    parser.add_argument('--documents_dir', default='documents', 
                        help='Directory containing documents to index')
    parser.add_argument('--stopwords_file', default='stopwords.txt', 
                        help='File containing stopwords')
    parser.add_argument('--special_chars_file', default='special_chars.txt', 
                        help='File containing special characters to remove')
    parser.add_argument('--index_file', default='index.pkl', 
                        help='Path to save/load the index')
    parser.add_argument('--use_existing', action='store_true', 
                        help='Use existing index if available')
    parser.add_argument('--index_mode', choices=['auto', 'standard', 'parallel'], default='auto',
                        help='Index implementation to use')
    parser.add_argument('--vsm_mode', choices=['auto', 'standard', 'parallel', 'hybrid', 'sparse'], default='auto',
                        help='VSM implementation to use')
    parser.add_argument('--parallel_index_threshold', type=int, default=IndexFactory.DEFAULT_PARALLEL_DOC_THRESHOLD,
                       help=f'Document threshold for parallel index (default: {IndexFactory.DEFAULT_PARALLEL_DOC_THRESHOLD})')
    parser.add_argument('--hybrid_vsm_threshold', type=int, default=VSMFactory.DEFAULT_HYBRID_DOC_THRESHOLD,
                       help=f'Document threshold for hybrid vs parallel VSM (default: {VSMFactory.DEFAULT_HYBRID_DOC_THRESHOLD})')
    parser.add_argument('--export_json', default=None,
                        help='Export index to JSON file (specify filename)')
    parser.add_argument('--stats', action='store_true',
                        help='Display detailed index statistics')
    parser.add_argument('--check_deps', action='store_true',
                        help='Check and display available dependencies')
    return parser.parse_args()

def display_dependencies():
    """Display available dependencies."""
    vsm_deps = VSMFactory.check_dependencies()
    mp_available = IndexFactory.check_multiprocessing()
    
    print("\n=== Available Dependencies ===")
    print(f"NumPy: {'Available' if vsm_deps['numpy'] else 'Not Available'}")
    print(f"SciPy Sparse: {'Available' if vsm_deps['scipy.sparse'] else 'Not Available'}")
    print(f"Scikit-learn Metrics: {'Available' if vsm_deps['sklearn.metrics'] else 'Not Available'}")
    print(f"Multiprocessing: {'Available' if mp_available else 'Not Available'}")
    
    print("\n=== Recommended Implementations ===")
    if vsm_deps['scipy.sparse'] and vsm_deps['sklearn.metrics']:
        print("VSM: Sparse (optimal)")
    elif mp_available:
        print("VSM: Hybrid/Parallel (good)")
    else:
        print("VSM: Standard (basic)")
        
    print(f"Index: {'Parallel' if mp_available else 'Standard'} (based on dataset size)")
    print("=" * 61)

def display_banner():
    print("=" * 61)
    print("=" * 19 + " CSC790-IR Homework 03 " + "=" * 19)
    print("First Name: Matthew")
    print("Last Name : Branson")
    print("=" * 61)

def display_vocabulary_statistics(index):
    print(f"\nThe number of unique words is: {index.vocab_size}")
    print("The top 10 most frequent words are:")
    for i, (term, freq) in enumerate(index.get_most_frequent_terms(n=10), 1):
        print(f"    {i}. {term} ({freq:,})")
    print("=" * 61)

def display_detailed_statistics(index):
    """Display detailed statistics about the index."""
    stats = index.get_statistics()
    
    print("\n=== Index Statistics ===")
    print(f"Total Documents: {stats['document_count']:,}")
    print(f"Vocabulary Size: {stats['vocabulary_size']:,}")
    print(f"Average Document Length: {stats['avg_doc_length']:.2f} terms")
    print(f"Max Document Length: {stats['max_doc_length']:,} terms")
    print(f"Min Document Length: {stats['min_doc_length']:,} terms")
    print(f"Average Term Frequency: {stats['avg_term_freq']:.2f}")
    print(f"Average Document Frequency: {stats['avg_doc_freq']:.2f}")
    
    print("\n=== Memory Usage ===")
    for key, value in stats['memory_usage'].items():
        if value > 1024 * 1024 * 1024:
            formatted = f"{value / (1024 * 1024 * 1024):.2f} GB"
        elif value > 1024 * 1024:
            formatted = f"{value / (1024 * 1024):.2f} MB"
        elif value > 1024:
            formatted = f"{value / 1024:.2f} KB"
        else:
            formatted = f"{value:,} bytes"
        
        print(f"{key}: {formatted}")
    
    print("\n"+"=" * 61)

def get_valid_int_input(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
            print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def display_similar_documents(vsm, k):
    print("\nThe top k closest documents are:")
    weighting_schemes = [
        ("1. Using tf", "tf"),
        ("2. Using tfidf", "tfidf"),
        ("3. Using wfidf", "sublinear")
    ]
    
    for label, weighting in weighting_schemes:
        print(f"\n{label}:")
        similar_docs = vsm.find_similar_documents(k=k, weighting=weighting)
        
        if not similar_docs:
            print("    No similar document pairs found.")
        else:
            for doc1, doc2, sim in similar_docs:
                print(f"    {doc1}, {doc2} with similarity of {sim:.2f}")

def main():
    args = parse_arguments()
    profiler = Profiler()
    profiler.start_global_timer()
    
    if args.check_deps:
        display_dependencies()

    if args.use_existing and os.path.exists(args.index_file):
        with profiler.timer("Index Loading"):
            from src.index.standard_index import StandardIndex
            index = StandardIndex.load(args.index_file)
    else:
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

    profiler.pause_global_timer()

    display_banner()
    display_vocabulary_statistics(index)

    if args.stats:
        display_detailed_statistics(index)

    # Get user input
    k = get_valid_int_input("\nEnter the number of top similar document pairs (k): ")

    profiler.resume_global_timer()

    # Display similar documents
    display_similar_documents(vsm, k)

    # Time reporting
    print("\n"+profiler.generate_report(
        doc_count=index.doc_count,
        vocab_size=index.vocab_size,
        filename="performance.log"
    ))

if __name__ == "__main__":
    main()