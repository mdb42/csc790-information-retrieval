import os
from HW01.HW01 import InvertedIndex, build_inverted_index
# 50 points extra credit for parallel processing and multi-threading

# Quiz Friday

# Term Frequency: Number of times a term appears in a document
# Document Frequency: Number of documents a term appears in

# With Incidence Matrix
# Boolean Retrieval Model forming vector (Brutus AND Ceasar BUT NOT Calpurnia)



def main():
    """
      1) Build or load the index
      2) Show index size
      3) Prompt user for queries
      4) Return matching documents
    """
    # Directory containing your documents
    documents_dir = "HW01//documents"

    # Just a few common stop words. Will find a better option later.
    stop_words = {"the", "a", "for", "to", "is", "are", "in", "of", "and"}
    stop_words_path = "HW01//stop_words.txt"

    # Decide whether to build or load from file
    index_file_path = "HW01//saved_index.pkl"
    use_existing_index = False

    if use_existing_index and os.path.exists(index_file_path):
        print("[+] Loading existing index from file...")
        inv_index = InvertedIndex()
        inv_index.load(index_file_path)
    else:
        print("[+] Building a new index from the documents directory...")
        inv_index = build_inverted_index(documents_dir, stop_words=stop_words)
        print("[+] Saving the new index to disk...")
        inv_index.save(index_file_path)

    # Display the size of the index
    size_in_bytes = inv_index.get_index_size_in_bytes()
    print(f"[+] Inverted Index built/loaded. Size in memory: {size_in_bytes} bytes")

    while True:
        query_str = input("\nEnter a Boolean query (or type 'exit' to quit): ")
        if query_str.lower() == 'exit':
            break

        matching_doc_ids = inv_index.boolean_retrieve(query_str)
        if not matching_doc_ids:
            print("No documents matched your query.")
        else:
            matching_filenames = [inv_index.reverse_doc_id_map[doc_id] 
                                  for doc_id in sorted(matching_doc_ids)]
            print(f"Documents matching '{query_str}':")
            for fname in matching_filenames:
                print(f"  - {fname}")

if __name__ == "__main__":
    main()