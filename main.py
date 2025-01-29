from HW01 import document_indexer

def main():
    params = {
        "stopwords_dir": "HW01//stopwords.txt",
        "documents_dir": "HW01//documents",
        "index_file": "HW01//saved_index.pkl",
        "query_file": "HW01//query_list.txt",
        "use_existing_index": False,
        "use_parallel": True,
    }
    
    document_indexer.run(**params)

if __name__ == "__main__":
    main()