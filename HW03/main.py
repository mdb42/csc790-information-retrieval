# main.py
import os
import argparse
from src.performance_monitoring import Profiler
from src.text_processor import TextProcessor
from src.index.memory_index import MemoryIndex
from src.vsm.standard_vsm import StandardVSM

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
    parser.add_argument('--parallel', action='store_true', 
                        help='Enable parallel processing')
    return parser.parse_args()

def display_banner():
    print("=" * 61)
    print("=" * 19 + " CSC790-IR Homework 03 " + "=" * 19)
    print("First Name: Matthew")
    print("Last Name : Branson")
    print("=" * 61)

def display_vocabulary_statistics(index: MemoryIndex):
    print(f"\nThe number of unique words is: {index.vocab_size}")
    print("The top 10 most frequent words are:")
    for i, (term, freq) in enumerate(index.get_most_frequent_terms(n=10), 1):
        print(f"    {i}. {term} ({freq:,})")
    print("=" * 61)

def get_valid_int_input(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
            print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def find_and_display_similar_documents(vsm, k):
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

    if args.use_existing and os.path.exists(args.index_file):
        # Load existing index
        with profiler.timer("Index Loading"):
            index = MemoryIndex.load(args.index_file)
    else:
        # Process documents and build new index
        text_processor = TextProcessor(args.documents_dir, args.stopwords_file, args.special_chars_file, args.parallel)
        with profiler.timer("Document Processing"):
            filenames, term_freqs = text_processor.process_documents()
        
        index = MemoryIndex()
        with profiler.timer("Index Building"):
            for filename, freq_dict in zip(filenames, term_freqs):
                index.add_document(freq_dict, filename)
        
        index.save(args.index_file)
    
    # Vector Space Model
    vsm = StandardVSM(index, profiler)
    vsm.build_model()

    profiler.pause_global_timer()

    # Display Information
    display_banner()
    display_vocabulary_statistics(index)

    # Get user input
    k = get_valid_int_input("\nEnter the number of top similar document pairs (k): ")

    profiler.resume_global_timer()

    # Find similar documents
    find_and_display_similar_documents(vsm, k)

    # Final reporting
    log_report = profiler.write_log_file(
        filename="performance.log",
        doc_count=index.doc_count,
        vocab_size=index.vocab_size
    )

    print("\n"+log_report)

if __name__ == "__main__":
    main()
