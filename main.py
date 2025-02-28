from engine.src.core import SearchEngine

def main():    
    engine = SearchEngine()
    engine.crawl()
    print("Crawling complete!")

if __name__ == "__main__":
    main()