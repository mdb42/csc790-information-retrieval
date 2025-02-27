"""
Constants and configuration values for the search engine.
"""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESOURCES_DIR = os.path.join(BASE_DIR, 'resources')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Subdirectories
CRAWL_DIR = os.path.join(DATA_DIR, 'crawl')
INDEX_DIR = os.path.join(DATA_DIR, 'index')

# Crawler settings
DEFAULT_CRAWL_DELAY = 1.0  # seconds
DEFAULT_USER_AGENT = 'SearchEngineBot/1.0 (Educational Project)'

# Indexer settings
STOPWORDS_FILE = os.path.join(RESOURCES_DIR, 'stopwords.txt')
SPECIAL_CHARS_FILE = os.path.join(RESOURCES_DIR, 'special-chars.txt')

# Retriever settings
DEFAULT_TOP_K = 10