"""
Core module that orchestrates the search engine components.
Initializes the crawler and manages the data collection phase.
"""

import logging
import os
import time

from .crawler import Crawler
from .constants import DATA_DIR, CRAWL_DIR, LOG_DIR
    
class SearchEngine:
    """
    Main class that coordinates the search engine components.
    Currently implements the crawling phase with hooks for future indexing and retrieval.
    """
    
    def __init__(self, config=None):
        """
        Initialize the search engine with configuration.
        
        Args:
            config (dict, optional): Configuration dictionary with optional keys:
                - seed_urls: List of URLs to start crawling from
                - crawl_delay: Seconds to wait between requests to the same domain
                - max_pages: Maximum number of pages to crawl
                - max_depth: Maximum link depth to crawl
                - user_agent: Custom user agent string for the crawler
        """
        self.config = config or {}
        
        # Ensure data directories exist
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(CRAWL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(LOG_DIR, f"search_engine_{time.strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('searchengine')
        
        # Initialize crawler (but don't start it yet)
        self.crawler = None
        self.logger.info("Search engine initialized")

    def crawl(self, seed_urls=None, max_pages=100, max_depth=3, delay=1.0):
        """
        Start the crawling process to collect documents.
        
        Args:
            seed_urls (list): List of URLs to start crawling from
            max_pages (int): Maximum number of pages to crawl
            max_depth (int): Maximum link depth to crawl
            delay (float): Seconds to wait between requests to the same domain
            
        Returns:
            int: Number of pages successfully crawled
        """
        # Use provided parameters or fall back to config values
        seed_urls = seed_urls or self.config.get('seed_urls', ['https://en.wikipedia.org/wiki/Information_retrieval'])
        max_pages = max_pages or self.config.get('max_pages', 100)
        max_depth = max_depth or self.config.get('max_depth', 3)
        delay = delay or self.config.get('crawl_delay', 1.0)
        
        self.logger.info(f"Starting crawl with {len(seed_urls)} seed URLs")
        self.logger.info(f"Crawl parameters: max_pages={max_pages}, max_depth={max_depth}, delay={delay}")
        
        # Initialize crawler with appropriate configuration
        self.crawler = Crawler(
            base_urls=seed_urls,
            output_dir=CRAWL_DIR,
            delay=delay,
            max_pages=max_pages,
            max_depth=max_depth,
            user_agent=self.config.get('user_agent', 'SearchEngineBot/1.0 (Educational Project)')
        )
        
        # Start crawling and time the process
        start_time = time.time()
        pages_crawled = self.crawler.start()
        elapsed_time = time.time() - start_time
        
        # Log crawling statistics
        self.logger.info(f"Crawling completed: {pages_crawled} pages in {elapsed_time:.2f} seconds")
        self.logger.info(f"Average crawl rate: {pages_crawled/elapsed_time:.2f} pages/second")
        
        return pages_crawled
    
    def get_crawl_stats(self):
        """
        Get statistics about the most recent crawl.
        
        Returns:
            dict: Statistics about the crawl or None if no crawl has been performed
        """
        if not self.crawler:
            self.logger.warning("No crawler initialized, cannot provide stats")
            return None
        
        return {
            'pages_crawled': len(self.crawler.visited_urls),
            'unique_domains': len(self.crawler.domain_last_access),
            'robots_checked': len(self.crawler.robots_cache),
            'crawl_failures': self.crawler.failure_count
        }

def main():
    """
    Main entry point to run the search engine.
    """
    # Example configuration
    config = {
        'seed_urls': [
            'https://en.wikipedia.org/wiki/Information_retrieval',
            'https://en.wikipedia.org/wiki/Search_engine_technology',
            'https://en.wikipedia.org/wiki/Web_crawler'
        ],
        'max_pages': 50,
        'max_depth': 2,
        'crawl_delay': 1.0
    }
    
    # Initialize and run the search engine
    engine = SearchEngine(config)
    
    # Start crawling
    pages_crawled = engine.crawl()
    
    # Print crawl statistics
    stats = engine.get_crawl_stats()
    if stats:
        print("\nCrawl Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    print(f"\nCrawling complete! Collected {pages_crawled} pages.")
    print(f"Documents saved to: {CRAWL_DIR}")


if __name__ == "__main__":
    main()