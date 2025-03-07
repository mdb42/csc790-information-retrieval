# main.py
import argparse
from src.text_processor import TextProcessor
from src.vector_space_model import VectorSpaceModel
from src.performance_monitoring import Profiler

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

def display_vocabulary_statistics(vsm):
    print(f"\nThe number of unique words is: {vsm.vocab_size}")
    print("The top 10 most frequent words are:")
    for i, (term, freq) in enumerate(vsm.get_most_frequent_terms(n=10), 1):
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

    text_processor = TextProcessor(
        documents_dir=args.documents_dir,
        stopwords_file=args.stopwords_file,
        special_chars_file=args.special_chars_file,
        parallel=args.parallel
    )
    
    vsm = VectorSpaceModel(
        text_processor, 
        index_file=args.index_file, 
        use_existing_index=args.use_existing,
        profiler=profiler
    )

    display_banner()
    display_vocabulary_statistics(vsm)
    k = get_valid_int_input("\nEnter the number of top similar document pairs (k): ")
    find_and_display_similar_documents(vsm, k)

    total_time = profiler.get_global_time()
    profiler.write_log_file(
        filename="performance.log",
        doc_count=vsm.doc_count,
        vocab_size=vsm.vocab_size,
        total_time=total_time
    )
    print(f"\nTotal execution time: {total_time:.4f} seconds")

if __name__ == "__main__":
    main()
