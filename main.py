from engine.src.core import SearchEngine

def main():
    config = {
        'seed_urls': ['https://example.com', 'https://anothersite.org'],
        'max_pages': 100
    }
    
    engine = SearchEngine(config)
    
    engine.crawl()
    
    print("Crawling complete!")

if __name__ == "__main__":
    main()