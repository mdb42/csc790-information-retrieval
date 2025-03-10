# src/utils.py
import json
import os
from src.index.factory import IndexFactory
from src.vsm.factory import VSMFactory

def load_config(config_file='config.json'):
    """Load configuration from JSON file, falling back to defaults if not found."""
    if not os.path.exists(config_file):
        return ensure_config_exists(config_file)
        
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Fill in any missing keys with defaults
        default_config = {
            'documents_dir': 'documents',
            'stopwords_file': 'stopwords.txt',
            'special_chars_file': 'special_chars.txt',
            'index_file': 'index.pkl',
            'index_mode': 'auto',
            'vsm_mode': 'auto',
            'parallel_index_threshold': IndexFactory.DEFAULT_PARALLEL_DOC_THRESHOLD,
            'hybrid_vsm_threshold': VSMFactory.DEFAULT_HYBRID_DOC_THRESHOLD
        }
        
        # Ensure all needed keys exist by combining with defaults
        merged_config = {**default_config, **config}
        return merged_config
    except Exception as e:
        print(f"Warning: Error loading config file {config_file}: {e}")
        print("Using default configuration.")
        return ensure_config_exists(config_file)

def save_config(config, config_file='config.json'):
    """Save current configuration to JSON file."""
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
        'documents_dir': 'documents',
        'stopwords_file': 'stopwords.txt',
        'special_chars_file': 'special_chars.txt',
        'index_file': 'index.pkl',
        'index_mode': 'auto',
        'vsm_mode': 'auto',
        'parallel_index_threshold': IndexFactory.DEFAULT_PARALLEL_DOC_THRESHOLD,
        'hybrid_vsm_threshold': VSMFactory.DEFAULT_HYBRID_DOC_THRESHOLD
    }
    
    # Save the default configuration
    try:
        save_config(default_config, config_file)
        print(f"Created default configuration file: {config_file}")
    except Exception as e:
        print(f"Warning: Could not create default configuration file: {e}")
    
    return default_config

def format_memory_size(value):
    """Format memory size to human-readable format."""
    if value > 1024 * 1024 * 1024:
        return f"{value / (1024 * 1024 * 1024):.2f} GB"
    elif value > 1024 * 1024:
        return f"{value / (1024 * 1024):.2f} MB"
    elif value > 1024:
        return f"{value / 1024:.2f} KB"
    else:
        return f"{value:,} bytes"    

def display_detailed_statistics(index):
    """Display detailed statistics about the index."""
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

def display_dependencies():
    """Display available dependencies."""
    vsm_deps = VSMFactory.check_dependencies()
    mp_available = IndexFactory.check_multiprocessing()
    
    print("\n=== Available Dependencies ===")
    print(f"NumPy: {'Available' if vsm_deps['numpy'] else 'Not Available'}")
    print(f"SciPy Sparse: {'Available' if vsm_deps['scipy.sparse'] else 'Not Available'}")
    print(f"Scikit-learn Metrics: {'Available' if vsm_deps['sklearn.metrics'] else 'Not Available'}")
    print(f"Multiprocessing: {'Available' if mp_available else 'Not Available'}")
    
    print("\n=== Recommended Implementations ===")
    if vsm_deps['scipy.sparse'] and vsm_deps['sklearn.metrics']:
        print("VSM: Sparse (optimal)")
    elif mp_available:
        print("VSM: Hybrid/Parallel (good)")
    else:
        print("VSM: Standard (basic)")
    
    # Update index recommendation to mention dataset size threshold
    if mp_available:
        print(f"Index: Parallel (for datasets > {IndexFactory.DEFAULT_PARALLEL_DOC_THRESHOLD} documents)")
        print(f"       Standard (for smaller datasets)")
    else:
        print(f"Index: Standard (multiprocessing unavailable)")
    
    print("=" * 56)
