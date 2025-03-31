# src/utils.py
"""
Utility Functions
Author: Matthew Branson
Date: March 31, 2025

This module provides utility functions for loading and saving configuration,
formatting memory sizes, displaying detailed index statistics, and checking dependencies.
"""
import json
import os
from typing import Dict

# Define a single default configuration dictionary
DEFAULT_CONFIG = {
    "query_files": ["query1.txt", "query2.txt"],
    "labels_files": ["file_label_query1.txt", "file_label_query2.txt"],
    "documents_dir": "documents",
    "stopwords_file": "stopwords.txt",
    "special_chars_file": "special_chars.txt",
    "index_file": "index.pkl",
    "index_mode": "auto",
    "parallel_index_threshold": 5000
}

def load_config(config_file='config.json') -> Dict:
    """
    Load configuration from a JSON file, falling back to defaults if not found or invalid.
    
    Args:
        config_file (str): Path to the configuration file
    
    Returns:
        dict: The loaded or default configuration
    """
    if not os.path.exists(config_file):
        return _create_default_config(config_file)
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Merge with defaults to ensure all keys exist
        return {**DEFAULT_CONFIG, **config}
    except Exception as e:
        print(f"Warning: Error loading config file {config_file}: {e}")
        print("Using default configuration.")
        return _create_default_config(config_file)

def save_config(config: Dict, config_file='config.json'):
    """
    Save the current configuration to a JSON file.
    
    Args:
        config (dict): The configuration to save
        config_file (str): Path to the configuration file
    """
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving configuration to {config_file}: {e}")

def _create_default_config(config_file='config.json') -> Dict:
    """
    Create a default configuration file if it doesn't exist.
    
    Args:
        config_file (str): Path to the configuration file
        
    Returns:
        dict: The default configuration
    """
    try:
        save_config(DEFAULT_CONFIG, config_file)
        print(f"Created default configuration file: {config_file}")
    except Exception as e:
        print(f"Warning: Could not create default configuration file: {e}")
    
    return DEFAULT_CONFIG

def check_multiprocessing() -> bool:
    """
    Check if multiprocessing is available and functional on the system.
    
    Returns:
        bool: True if multiprocessing is available and functional, False otherwise
    """
    try:
        import multiprocessing
        with multiprocessing.Pool(1) as _:
            pass
        return True
    except (ImportError, OSError, ValueError):
        return False

def display_vocabulary_statistics(index):
    """
    Display vocabulary statistics including count and most frequent terms.
    
    Args:
        index: The document index object with vocabulary information
    """
    print(f"The number of unique words is: {index.vocab_size}")
    print("The top 10 most frequent words are:")
    for i, (term, freq) in enumerate(index.get_most_frequent_terms(n=10), 1):
        print(f"    {i}. {term} ({freq:,})")
    print("=" * 55)

def display_detailed_statistics(index):
    """
    Display detailed statistics about the index.
    
    Args:
        index (BaseIndex): The index instance to analyze
    """
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
        print(f"{key}: {format_memory_size(value)}")
    
    print("\n" + "=" * 56)

def format_memory_size(value: int) -> str:
    """
    Format memory size to a human-readable format.
    
    Args:
        value (int): Memory size in bytes
        
    Returns:
        str: Formatted memory size with units
    """
    if value > 1024 ** 3:
        return f"{value / (1024 ** 3):.2f} GB"
    elif value > 1024 ** 2:
        return f"{value / (1024 ** 2):.2f} MB"
    elif value > 1024:
        return f"{value / 1024:.2f} KB"
    else:
        return f"{value:,} bytes"