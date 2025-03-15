# src/vsm/sparse_vsm.py
"""
Sparse Vector Space Model
Author: Matthew Branson
Date: March 14, 2025

This module implements a highly optimized Vector Space Model using sparse matrices
and numerical libraries (NumPy, SciPy, scikit-learn). This implementation offers
substantial performance improvements over both standard and parallel implementations,
especially for medium to large document collections and is recommended for all
practical use cases.
"""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from src.retrieval.vsm import BaseVSM
from src.profiler import Profiler
from src.index import BaseIndex


class SparseVSM(BaseVSM):
    """
    Sparse Vector Space Model implementation using numerical libraries.
    
    For efficient storage and computation, it uses Compressed Sparse Row (CSR) matrix
    format, leveraging vectorized operations instead of Python loops, and optimized
    cosine similarity calculation using BLAS operations.
    
    Attributes:
        term_to_idx (dict): Mapping from terms to column indices in the sparse matrix
        vocab_size (int): Number of unique terms in the vocabulary
        doc_term_matrix (csr_matrix): Sparse matrix of term frequencies
        tfidf_matrix (csr_matrix): Sparse matrix of TF-IDF weights
        sublinear_matrix (csr_matrix): Sparse matrix of sublinear (log) TF-IDF weights
        idf (ndarray): Array of IDF values for each term
    """
    def __init__(self, index: BaseIndex, profiler: Profiler = None):
        """
        Initialize the SparseVSM with an index.
        
        Args:
            index (BaseIndex): The document index
            profiler (Profiler, optional): Performance profiler for timing operations
        """
        super().__init__(index, profiler)
        self._build_optimized_sparse_matrix()
        self._precompute_optimized_weights()

    def _build_optimized_sparse_matrix(self):
        """
        Build a sparse matrix representation of the document-term frequencies.
        
        This method creates a Compressed Sparse Row (CSR) matrix where:
        - Each row represents a document
        - Each column represents a term in the vocabulary
        - Each cell value represents the term frequency
        
        CSR format is chosen because:
        - It's memory efficient for sparse matrices (only stores non-zero elements)
        - It's optimized for row-wise operations and matrix-vector operations
        - It's the preferred format for scikit-learn's cosine_similarity function
        
        Raises:
            RuntimeError: If matrix construction fails
        """
        with self.profiler.timer("Sparse Matrix Construction"):
            try:
                # Create mapping from terms to column indices
                self.term_to_idx = {term: idx for idx, term in enumerate(self.index.term_doc_freqs)}
                self.vocab_size = len(self.term_to_idx)
                
                # Prepare data for sparse matrix construction using coordinate format (COO)
                # which is later converted to CSR for efficiency
                doc_ids = []        # Row indices (document IDs)
                col_indices = []    # Column indices (term indices)
                data = []           # Values (term frequencies)
                
                # Collect data for all documents
                for doc_id, term_counts in self.index.doc_term_freqs.items():
                    if not term_counts:
                        continue
                        
                    terms = list(term_counts.keys())
                    # Add document ID for each term (repeat doc_id for each term in document)
                    doc_ids.extend([doc_id] * len(terms))
                    
                    try:
                        # Map terms to column indices
                        col_indices.extend([self.term_to_idx[term] for term in terms])
                    except KeyError as e:
                        raise ValueError(f"Term {e} found in document {doc_id} but not in term_doc_freqs") from e
                        
                    # Add term frequencies
                    data.extend(term_counts.values())
                
                # Construct the sparse matrix in CSR format
                # CSR is more efficient than COO for arithmetic operations
                self.doc_term_matrix = csr_matrix(
                    (data, (doc_ids, col_indices)),
                    shape=(self.index.doc_count, self.vocab_size),
                    dtype=np.float32  # 32-bit float is sufficient for TF values and saves memory
                )
            except Exception as e:
                raise RuntimeError(f"Failed to build sparse matrix: {e}") from e

    def _precompute_optimized_weights(self):
        """
        Precompute TF-IDF and sublinear TF-IDF weights as sparse matrices.
        
        Raises:
            RuntimeError: If weight computation fails
        """
        with self.profiler.timer("Optimized Weight Precomputation"):
            try:
                doc_count = self.index.doc_count
                
                # Compute IDF values for all terms
                # Extract document frequencies from the index
                dfs = np.array([len(docs) for docs in self.index.term_doc_freqs.values()], dtype=np.float32)
                
                # IDF = log10(N/df) where N is the total number of documents
                # np.maximum ensures we don't divide by zero
                self.idf = np.log10(doc_count / np.maximum(dfs, 1.0))
                
                # Compute TF-IDF by element-wise multiplication of TF matrix with IDF values
                # scipy.sparse.csr_matrix.multiply performs element-wise multiplication with a vector
                self.tfidf_matrix = self.doc_term_matrix.multiply(self.idf)
                
                # Compute sublinear TF-IDF using log(TF) instead of raw TF
                # 1. Calculate log(TF) for all non-zero elements in the TF matrix
                # 2. Create a new CSR matrix with the log values
                # 3. Multiply by IDF to get the sublinear TF-IDF
                
                # log(TF) for all non-zero elements
                # Use np.maximum to ensure we don't take log of zero (with a small epsilon)
                log_data = 1 + np.log10(np.maximum(self.doc_term_matrix.data, 1e-10)) # Not quite zero to avoid -inf
                
                # Create new CSR matrix with the same structure but log(TF) values
                self.sublinear_matrix = csr_matrix(
                    (log_data, self.doc_term_matrix.indices, self.doc_term_matrix.indptr),
                    shape=self.doc_term_matrix.shape
                ).multiply(self.idf)
            except Exception as e:
                raise RuntimeError(f"Failed to precompute weights: {e}") from e

    def _optimized_cosine_similarity(self, matrix):
        """
        Compute pairwise cosine similarities between all documents.
        
        Args:
            matrix (csr_matrix): Document-term matrix with the desired weighting
            
        Returns:
            csr_matrix: Sparse matrix of pairwise cosine similarities
        """
        with self.profiler.timer("Matrix Normalization + Similarity"):
            return cosine_similarity(matrix, dense_output=False)

    def find_similar_documents(self, k=10, weighting='tf') -> List[Tuple[str, str, float]]:
        """
        Find the top k most similar document pairs using the specified weighting.
        
        The approach here is different from the other VSM implementations:
        - Instead of iterating through document pairs, we compute all similarities at once
        - We leverage optimized numerical libraries for much better performance
        - We use numpy's argpartition for efficient top-k selection
        
        Args:
            k (int): Number of top document pairs to return
            weighting (str): Weighting scheme to use ('tf', 'tfidf', or 'sublinear')
            
        Returns:
            List[Tuple[str, str, float]]: List of (doc1, doc2, similarity) tuples
                                         sorted by similarity in descending order
        """
        with self.profiler.timer(f"Similarity Calculation ({weighting})"):
            try:
                # Select the appropriate matrix based on weighting scheme
                if weighting == 'tf':
                    matrix = self.doc_term_matrix
                elif weighting == 'tfidf':
                    matrix = self.tfidf_matrix
                elif weighting == 'sublinear':
                    matrix = self.sublinear_matrix
                else:
                    raise ValueError(f"Unknown weighting scheme: {weighting}")

                # Compute pairwise cosine similarities
                similarity_matrix = self._optimized_cosine_similarity(matrix)
                
                # Convert to coordinate format (COO) for easier filtering
                # COO format gives us direct access to row indices, column indices, and data
                similarity_matrix = similarity_matrix.tocoo()
                
                # Extract upper triangular part (where row < col)
                # This avoids duplicate pairs and self-similarities
                # Note: Using scipy's triu would be ideal but it requires CSC matrix format,
                # which would require an additional conversion. This approach is faster.
                mask = similarity_matrix.row < similarity_matrix.col
                rows = similarity_matrix.row[mask]
                cols = similarity_matrix.col[mask]
                sims = similarity_matrix.data[mask]
                
                # Select top k similarities
                if len(sims) > k:
                    # Use argpartition for efficient partial sorting (O(n) complexity)
                    # This is much faster than full sorting when we only need top k
                    idx = np.argpartition(sims, -k)[-k:]
                    
                    # Sort just the top k elements to get the final order
                    top_indices = idx[np.argsort(-sims[idx])]
                else:
                    # If we have fewer than k similarities, sort them all
                    top_indices = np.argsort(-sims)[:min(k, len(sims))]
                
                # Convert indices to document pairs and return
                return [
                    (self.index.filenames[rows[i]], 
                    self.index.filenames[cols[i]], 
                    sims[i]
                    ) for i in top_indices
                ]
            except Exception as e:
                self.profiler.log_message(f"Error finding similar documents: {e}")
                return []