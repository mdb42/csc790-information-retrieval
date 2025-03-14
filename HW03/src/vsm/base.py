# src/vsm/base_vsm.py
"""
Base Vector Space Model Interface
Author: Matthew Branson
Date: March 14, 2025

This module defines the abstract base class for all Vector Space Model (VSM) 
implementations. It establishes the common interface and data structures that
concrete VSM implementations must provide.
"""
from typing import List, Tuple, Dict, Any
from src.profiler import Profiler
from src.index import BaseIndex


class BaseVSM:
    """
    Abstract base class for Vector Space Model implementations.
    
    This class defines the common interface and data structures for all VSM
    implementations. It provides storage for term weights and document magnitudes
    under different weighting schemes.
    
    Attributes:
        index (BaseIndex): Document index containing term frequencies and document info
        profiler (Profiler): Performance monitoring utility for timing operations
        weights (Dict): Nested dictionary storing term weights by scheme and document ID
                       {scheme: {doc_id: {term: weight}}}
        magnitudes (Dict): Nested dictionary storing document magnitudes by scheme
                          {scheme: {doc_id: magnitude}}
    """
    def __init__(self, index: BaseIndex, profiler: Profiler = None):
        """
        Initialize the base Vector Space Model.
        
        Args:
            index (BaseIndex): Document index containing the collection
            profiler (Profiler, optional): Performance profiler for timing operations
        """
        self.index = index
        self.profiler = profiler or Profiler()
        
        # Initialize weight dictionaries for different weighting schemes
        # Each scheme maps document IDs to term weight dictionaries
        self.weights = {
            'tf': {},       # Term frequency weights
            'tfidf': {},    # TF-IDF weights
            'sublinear': {} # Sublinear (logarithmic) TF-IDF weights
        }
        
        # Initialize magnitude dictionaries for different weighting schemes
        # Each scheme maps document IDs to vector magnitudes
        self.magnitudes = {
            'tf': {},       # Magnitudes for term frequency vectors
            'tfidf': {},    # Magnitudes for TF-IDF vectors
            'sublinear': {} # Magnitudes for sublinear TF-IDF vectors
        }
        
    def build_model(self):
        """
        Build the vector space model by computing term weights and document magnitudes.
        
        The three standard weighting schemes are:
        - tf: Raw term frequency
        - tfidf: Term frequency * inverse document frequency
        - sublinear: (1 + log(tf)) * idf (dampened term frequency)
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement build_model()")
    
    def find_similar_documents(self, k: int = 10, weighting: str = 'tf') -> List[Tuple[str, str, float]]:
        """
        Find the top k most similar document pairs using the specified weighting.
        
        Args:
            k (int): Number of top document pairs to return
            weighting (str): Weighting scheme to use ('tf', 'tfidf', or 'sublinear')
            
        Returns:
            List[Tuple[str, str, float]]: List of (doc1, doc2, similarity) tuples
                                         sorted by similarity in descending order
                                         
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement find_similar_documents()")