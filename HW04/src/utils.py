# src/utils.py
"""
Utility Functions
Author: Matthew Branson
Date: March 14, 2025

This module provides utility functions for loading and saving configuration,
formatting memory sizes, displaying detailed index statistics, and checking dependencies.
"""
import json
import os
from typing import Dict

def load_config(config_file='config.json'):
    """Load configuration from JSON file, falling back to defaults if not found.
    
    Args:
        config_file (str): Path to the configuration file
    
    Returns:
        dict: The loaded configuration
    """
    if not os.path.exists(config_file):
        return ensure_config_exists(config_file)
        
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Fill in any missing keys with defaults
        default_config = {
            'queries_file': 'queries.txt',
            'documents_dir': 'documents',
            'stopwords_file': 'stopwords.txt',
            'special_chars_file': 'special_chars.txt',
            'index_file': 'index.pkl',
            'bm25_k1': 1.2,
            'bm25_b': 0.75,
            'index_mode': 'auto',
            "parallel_index_threshold": 5000
        }
        
        # Ensure all needed keys exist by combining with defaults
        merged_config = {**default_config, **config}
        return merged_config
    except Exception as e:
        print(f"Warning: Error loading config file {config_file}: {e}")
        print("Using default configuration.")
        return ensure_config_exists(config_file)

def save_config(config, config_file='config.json'):
    """Save current configuration to JSON file.
    
    Args:
        config (dict): The configuration to save
        config_file (str): Path to the configuration file
    """
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

def ensure_config_exists(config_file='config.json'):
    """
    Check if a configuration file exists, and create a default one if it doesn't.
    
    Args:
        config_file (str): Path to the configuration file
        
    Returns:
        dict: The configuration (either loaded or newly created)
    """
    if os.path.exists(config_file):
        return load_config(config_file)
    
    # Default configuration
    default_config = {
        'queries_file': 'queries.txt',
        'documents_dir': 'documents',
        'stopwords_file': 'stopwords.txt',
        'special_chars_file': 'special_chars.txt',
        'index_file': 'index.pkl',
        'bm25_k1': 1.2,
        'bm25_b': 0.75,
        'index_mode': 'auto',
        "parallel_index_threshold": 5000
    }
    
    # Save the default configuration
    try:
        save_config(default_config, config_file)
        print(f"Created default configuration file: {config_file}")
    except Exception as e:
        print(f"Warning: Could not create default configuration file: {e}")
    
    return default_config

def check_multiprocessing():
        """
        Check if multiprocessing is available and functional on the system.
        
        This method attempts to create a multiprocessing pool to verify
        that parallel processing is actually available.
        
        Returns:
            bool: True if multiprocessing is available and functional, False otherwise
        """
        try:
            import multiprocessing
            try:
                # Try to create a process pool to verify multiprocessing works
                with multiprocessing.Pool(1) as p:
                    pass
                return True
            except (ImportError, OSError, ValueError):
                # Various errors that can occur if multiprocessing is unavailable
                return False
        except ImportError:
            # Multiprocessing module not available
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
    """Display detailed statistics about the index.
    
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
    
    print("\n"+"=" * 56)

def format_memory_size(value):
    """Format memory size to human-readable format.
    
    Args:
        value (int): Memory size in bytes
        
    Returns:
        str: Formatted memory size with units
    """
    if value > 1024 * 1024 * 1024:
        return f"{value / (1024 * 1024 * 1024):.2f} GB"
    elif value > 1024 * 1024:
        return f"{value / (1024 * 1024):.2f} MB"
    elif value > 1024:
        return f"{value / 1024:.2f} KB"
    else:
        return f"{value:,} bytes"    
