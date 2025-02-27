"""
Web crawler module for search engine.
Handles URL discovery, content fetching, and initial processing.
"""

import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
import logging
import os
from urllib.robotparser import RobotFileParser
from collections import deque
import hashlib

class Crawler:
    """
    Web crawler that fetches and processes web pages while respecting robots.txt.
    
    Features:
    - Polite crawling with rate limiting
    - robots.txt compliance
    - URL normalization
    - Content extraction
    - Basic duplicate detection
    """
    
    def __init__(self, base_urls, output_dir="data/crawl", 
                 delay=1.0, max_pages=100, max_depth=3,
                 user_agent="SearchEngineBot/1.0 (Educational Project)"):
        """
        Initialize the crawler with starting URLs and configuration.
        
        Args:
            base_urls (list): List of seed URLs to start crawling from
            output_dir (str): Directory to store crawled content
            delay (float): Minimum delay between requests to the same domain (seconds)
            max_pages (int): Maximum number of pages to crawl
            max_depth (int): Maximum link depth to crawl
            user_agent (str): User agent string to use for HTTP requests
        """
        self.base_urls = [self._normalize_url(url) for url in base_urls]
        self.output_dir = output_dir
        self.delay = delay
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.user_agent = user_agent
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Track crawled URLs and content hashes to avoid duplicates
        self.visited_urls = set()
        self.content_hashes = set()
        
        # Track last access time per domain for rate limiting
        self.domain_last_access = {}
        
        # Cache for robots.txt parsers
        self.robots_cache = {}
        
        # Track crawl failures
        self.failure_count = 0
        
        # Setup logging
        self.logger = logging.getLogger('crawler')
    
    def start(self):
        """
        Start the crawling process from the base URLs.
        
        Returns:
            int: Number of pages successfully crawled
        """
        self.logger.info(f"Starting crawl with seeds: {self.base_urls}")
        self.logger.info(f"Using user agent: {self.user_agent}")
        
        # Queue of (url, depth) pairs to crawl
        url_queue = deque([(url, 0) for url in self.base_urls])
        pages_crawled = 0
        
        while url_queue and pages_crawled < self.max_pages:
            url, depth = url_queue.popleft()
            
            # Check if we've reached max depth
            if depth > self.max_depth:
                continue
            
            # Skip if already visited
            if url in self.visited_urls:
                continue
            
            # Check if we're allowed to crawl this URL
            if not self._can_fetch(url):
                self.logger.info(f"Skipping {url} (robots.txt disallowed)")
                continue
            
            # Respect rate limiting for the domain
            self._respect_rate_limits(url)
            
            # Fetch the page
            try:
                html, status_code = self._fetch_url(url)
                
                # Skip if fetch unsuccessful
                if status_code != 200 or not html:
                    self.logger.warning(f"Failed to fetch {url} (status: {status_code})")
                    self.failure_count += 1
                    continue
                
                # Check for duplicate content
                content_hash = hashlib.md5(html.encode('utf-8')).hexdigest()
                if content_hash in self.content_hashes:
                    self.logger.info(f"Skipping {url} (duplicate content)")
                    self.visited_urls.add(url)  # Mark as visited anyway
                    continue
                
                # Process the page
                self.content_hashes.add(content_hash)
                self.visited_urls.add(url)
                
                # Extract and save content
                title, text, links = self._process_html(html, url)
                self._save_page(url, title, text, html)
                
                pages_crawled += 1
                self.logger.info(f"Crawled ({pages_crawled}/{self.max_pages}): {url}")
                
                # Add new links to the queue
                for link in links:
                    if link not in self.visited_urls:
                        url_queue.append((link, depth + 1))
                
            except Exception as e:
                self.logger.error(f"Error crawling {url}: {str(e)}")
                self.failure_count += 1
        
        self.logger.info(f"Crawl completed. Pages crawled: {pages_crawled}, Failures: {self.failure_count}")
        return pages_crawled
    
    def _fetch_url(self, url):
        """
        Fetch the content of a URL.
        
        Args:
            url (str): URL to fetch
            
        Returns:
            tuple: (html_content, status_code)
        """
        headers = {
            'User-Agent': self.user_agent
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        return response.text, response.status_code
    
    def _process_html(self, html, base_url):
        """
        Process HTML to extract title, text, and links.
        
        Args:
            html (str): HTML content to process
            base_url (str): Base URL for resolving relative links
            
        Returns:
            tuple: (title, text, links)
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else ""
        
        # Extract text (simple version - combining various text elements)
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'div'])
        text = ' '.join([elem.get_text().strip() for elem in text_elements if elem.get_text().strip()])
        
        # Extract links
        links = []
        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']
            
            # Skip anchor links and javascript
            if link.startswith('#') or link.startswith('javascript:'):
                continue
                
            # Convert to absolute URL
            absolute_link = urllib.parse.urljoin(base_url, link)
            normalized_link = self._normalize_url(absolute_link)
            
            # Only add HTTP/HTTPS links from the same domain or explicitly allowed domains
            if normalized_link.startswith(('http://', 'https://')):
                # Optional: add domain filtering here if you want to stay within certain domains
                links.append(normalized_link)
        
        return title, text, links
    
    def _normalize_url(self, url):
        """
        Normalize a URL by removing fragments and some query parameters.
        
        Args:
            url (str): URL to normalize
            
        Returns:
            str: Normalized URL
        """
        # Skip invalid URLs
        if not url or not isinstance(url, str):
            return ""
            
        try:
            # Parse URL
            parsed = urllib.parse.urlparse(url)
            
            # Remove fragment
            parsed = parsed._replace(fragment='')
            
            # Remove common tracking parameters
            query_params = urllib.parse.parse_qs(parsed.query)
            for param in ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content']:
                if param in query_params:
                    del query_params[param]
            
            # Rebuild query string
            query_string = urllib.parse.urlencode(query_params, doseq=True)
            parsed = parsed._replace(query=query_string)
            
            # Reassemble URL
            normalized = urllib.parse.urlunparse(parsed)
            
            # Remove trailing slash for consistency
            if normalized.endswith('/'):
                normalized = normalized[:-1]
                
            return normalized
        except Exception as e:
            self.logger.warning(f"Error normalizing URL {url}: {str(e)}")
            return ""
    
    def _respect_rate_limits(self, url):
        """
        Ensure we respect rate limits for a domain.
        
        Args:
            url (str): URL to check
        """
        domain = urllib.parse.urlparse(url).netloc
        
        # Check if we need to wait
        if domain in self.domain_last_access:
            last_access = self.domain_last_access[domain]
            time_since_last = time.time() - last_access
            
            if time_since_last < self.delay:
                wait_time = self.delay - time_since_last
                self.logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for {domain}")
                time.sleep(wait_time)
        
        # Update last access time
        self.domain_last_access[domain] = time.time()
    
    def _can_fetch(self, url):
        """
        Check if we're allowed to fetch a URL according to robots.txt.
        
        Args:
            url (str): URL to check
            
        Returns:
            bool: True if allowed, False otherwise
        """
        try:
            parsed_url = urllib.parse.urlparse(url)
            domain = parsed_url.netloc
            
            # Check cache first
            if domain not in self.robots_cache:
                robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"
                rp = RobotFileParser()
                rp.set_url(robots_url)
                
                try:
                    rp.read()
                    self.robots_cache[domain] = rp
                except Exception as e:
                    self.logger.warning(f"Error fetching robots.txt for {domain}: {str(e)}")
                    # Assume allowed if we can't fetch robots.txt
                    return True
            
            # Check if we can fetch
            return self.robots_cache[domain].can_fetch(self.user_agent, url)
        except Exception as e:
            self.logger.warning(f"Error checking robots.txt for {url}: {str(e)}")
            return True  # Default to allowed on error
    
    def _save_page(self, url, title, text, html):
        """
        Save a crawled page to disk.
        
        Args:
            url (str): URL of the page
            title (str): Title of the page
            text (str): Extracted text content
            html (str): Raw HTML content
        """
        try:
            # Create a filename based on the URL
            url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
            
            # Save metadata and text
            metadata = {
                'url': url,
                'title': title,
                'crawl_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save text content
            text_path = os.path.join(self.output_dir, f"{url_hash}.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"URL: {url}\n")
                f.write(f"TITLE: {title}\n")
                f.write(f"CRAWL_TIME: {metadata['crawl_time']}\n")
                f.write("\n" + "="*80 + "\n\n")
                f.write(text)
            
            # Save HTML content
            html_path = os.path.join(self.output_dir, f"{url_hash}.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html)
                
            return True
        except Exception as e:
            self.logger.error(f"Error saving page {url}: {str(e)}")
            return False