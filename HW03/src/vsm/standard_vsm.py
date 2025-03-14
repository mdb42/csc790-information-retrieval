# src/vsm/standard_vsm.py
"""
Standard Vector Space Model
Author: Matthew Branson
Date: March 14, 2025

This module implements a single-threaded, optimized version of the Vector Space Model
for document similarity calculations. It's designed for small to medium document collections 
where the overhead of parallelization would outweigh its benefits.
"""
import math
import heapq
from typing import List, Tuple, Dict, Any
from src.profiler import Profiler
from src.index import BaseIndex
from src.vsm import BaseVSM


class StandardVSM(BaseVSM):
    """
    Standard Vector Space Model implementation using sequential processing.
    
    Attributes:
        idf_values (Dict[str, float]): Inverse document frequency values for all terms
    """
    def __init__(self, index: BaseIndex, profiler: Profiler = None):
        """
        Initialize the StandardVSM with an index.
        
        Args:
            index (BaseIndex): The document index
            profiler (Profiler, optional): Performance profiler for timing operations
        """
        super().__init__(index, profiler)
        self.idf_values = {}
        self._compute_idf_values()
        self.build_model()

    def _compute_idf_values(self):
        """
        Compute inverse document frequency (IDF) values for all terms in the index.
        
        IDF = log10(total_docs / doc_frequency)
        """
        total_docs = self.index.doc_count
        self.idf_values = {
            term: math.log10(total_docs / max(len(docs), 1))
            for term, docs in self.index.term_doc_freqs.items()
        }

    def build_model(self):
        """
        Build the vector space model by computing term weights and document magnitudes.
        
        This method computes three different weighting schemes in a single pass:
        - tf: Term frequency
        - tfidf: Term frequency * inverse document frequency
        - sublinear: (1 + log10(tf)) * idf (dampens the effect of high term frequencies)
        """
        with self.profiler.timer("Weight Precomputation"):
            for doc_id in self.index.doc_term_freqs:
                try:
                    term_counts = self.index.doc_term_freqs[doc_id]
                    
                    # Initialize weight dictionaries for the three schemes
                    tf = {}
                    tfidf = {}
                    sublinear = {}
                    
                    # Initialize squared magnitudes for vector normalization
                    tf_mag_sq = 0
                    tfidf_mag_sq = 0
                    sublinear_mag_sq = 0
                    
                    # Process each term in the document, calculating weights
                    # and updating magnitude components in a single pass
                    for term, freq in term_counts.items():
                        # Compute tf weighting
                        tf[term] = freq
                        tf_mag_sq += freq * freq
                        
                        # Get idf value for the term
                        idf = self.idf_values.get(term, 0)
                        
                        # Compute tfidf weighting
                        tfidf_val = freq * idf
                        tfidf[term] = tfidf_val
                        tfidf_mag_sq += tfidf_val * tfidf_val
                        
                        # Compute sublinear (wf-idf) weighting
                        sublinear_val = (1 + math.log10(freq)) * idf if freq > 0 else 0
                        sublinear[term] = sublinear_val
                        sublinear_mag_sq += sublinear_val * sublinear_val
                    
                    # Store the weights for each scheme
                    self.weights['tf'][doc_id] = tf
                    self.weights['tfidf'][doc_id] = tfidf
                    self.weights['sublinear'][doc_id] = sublinear
                    
                    # Compute and store magnitude values (sqrt of sum of squares)
                    self.magnitudes['tf'][doc_id] = math.sqrt(tf_mag_sq)
                    self.magnitudes['tfidf'][doc_id] = math.sqrt(tfidf_mag_sq)
                    self.magnitudes['sublinear'][doc_id] = math.sqrt(sublinear_mag_sq)
                except Exception as e:
                    self.profiler.log_message(f"Error processing document {doc_id}: {e}")

    def _compute_similarity(self, doc1: int, doc2: int, weighting: str) -> float:
        """
        Compute the cosine similarity between two documents using the specified weighting.
        
        Args:
            doc1 (int): Index of the first document
            doc2 (int): Index of the second document
            weighting (str): Weighting scheme to use ('tf', 'tfidf', or 'sublinear')
            
        Returns:
            float: Cosine similarity between the documents (0.0 to 1.0)
        """
        try:
            vec1 = self.weights[weighting][doc1]
            vec2 = self.weights[weighting][doc2]
            
            # Get common terms using efficient dictionary view intersection
            common_terms = vec1.keys() & vec2.keys()
            
            if not common_terms:
                return 0.0
                
            # Calculate dot product of common terms
            dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
            mag1 = self.magnitudes[weighting][doc1]
            mag2 = self.magnitudes[weighting][doc2]
            
            # Calculate cosine similarity
            return dot_product / (mag1 * mag2) if mag1 * mag2 != 0 else 0.0
        except KeyError:
            # Handle case where a document doesn't have weights or magnitude
            return 0.0

    def find_similar_documents(self, k=10, weighting='tf') -> List[Tuple[str, str, float]]:
        """
        Find the top k most similar document pairs using the specified weighting.
        
        This method computes pairwise similarities between all documents using a
        batched approach to improve cache locality. It maintains a min-heap of
        the top k most similar document pairs.
        
        Note:
            This has O(n²) complexity, which is unavoidable for pairwise similarity
            calculations. Optimizations like batching and early skipping help reduce
            the constant factor, but the quadratic complexity remains.
        
        Args:
            k (int): Number of top document pairs to return
            weighting (str): Weighting scheme to use ('tf', 'tfidf', or 'sublinear')
            
        Returns:
            List[Tuple[str, str, float]]: List of (doc1, doc2, similarity) tuples
                                         sorted by similarity in descending order
        """
        top_similarities = []
        total_docs = self.index.doc_count
        
        # Choose batch size to balance memory usage and cache efficiency
        # Square root of total docs is a good heuristic, bounded for very small or large collections
        batch_size = min(1000, max(100, int(math.sqrt(total_docs))))
        
        with self.profiler.timer(f"Similarity Calculation ({weighting})"):
            # Process documents in batches to improve cache locality
            for batch_start_i in range(0, total_docs, batch_size):
                batch_end_i = min(batch_start_i + batch_size, total_docs)
                
                # For each document i in the current batch
                for i in range(batch_start_i, batch_end_i):
                    # Skip documents with zero magnitude (no terms or all terms filtered out)
                    if i not in self.magnitudes[weighting] or self.magnitudes[weighting][i] == 0:
                        continue
                    
                    vec_i = self.weights[weighting][i]
                    mag_i = self.magnitudes[weighting][i]
                    
                    # Compare with all documents j > i (upper triangular matrix)
                    for j in range(i+1, total_docs):
                        # Skip documents with zero magnitude
                        if j not in self.magnitudes[weighting] or self.magnitudes[weighting][j] == 0:
                            continue
                        
                        vec_j = self.weights[weighting][j]
                        mag_j = self.magnitudes[weighting][j]
                        
                        # Fast check for common terms using dictionary view intersection
                        common_terms = vec_i.keys() & vec_j.keys()
                        
                        # Skip if no common terms (dot product would be zero)
                        if not common_terms:
                            continue
                        
                        # Calculate cosine similarity
                        dot_product = sum(vec_i[term] * vec_j[term] for term in common_terms)
                        sim = dot_product / (mag_i * mag_j) if mag_i * mag_j != 0 else 0.0
                        
                        # Use a min-heap to efficiently track top k similarities
                        if sim > 0:
                            if len(top_similarities) < k:
                                heapq.heappush(top_similarities, (sim, i, j))
                            elif sim > top_similarities[0][0]:
                                heapq.heappushpop(top_similarities, (sim, i, j))
        
        # Convert document indices to filenames and sort by similarity (descending)
        return [
            (self.index.filenames[i], self.index.filenames[j], sim)
            for sim, i, j in sorted(top_similarities, reverse=True)
        ]