# src/vsm/sparse_vsm.py
"""
Scenario:
 - numpy, scipy, and sklearn available (presumably true with anaconda)
 - you saved all your linear algebra class notes just for this moment

New:
- more robust error handling 

I did try to prune the sparse matrix, bypass unnecessary calculations, but it's
actually faster to perform something unnecessary with C and Fortran under the
hood than it is to attempt to do nothing in Python.
 """
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from src.vsm import BaseVSM
from src.performance_monitoring import Profiler
from src.index import BaseIndex

class SparseVSM(BaseVSM):
    def __init__(self, index: BaseIndex, profiler: Profiler = None):
        super().__init__(index, profiler)
        self._build_optimized_sparse_matrix()
        self._precompute_optimized_weights()

    def _build_optimized_sparse_matrix(self):
        with self.profiler.timer("Sparse Matrix Construction"):
            try:
                self.term_to_idx = {term: idx for idx, term in enumerate(self.index.term_doc_freqs)}
                self.vocab_size = len(self.term_to_idx)
                
                doc_ids = []
                col_indices = []
                data = []
                
                for doc_id, term_counts in self.index.doc_term_freqs.items():
                    if not term_counts:
                        continue
                        
                    terms = list(term_counts.keys())
                    doc_ids.extend([doc_id] * len(terms))
                    
                    try:
                        col_indices.extend([self.term_to_idx[term] for term in terms])
                    except KeyError as e:
                        raise ValueError(f"Term {e} found in document {doc_id} but not in term_doc_freqs") from e
                        
                    data.extend(term_counts.values())
                
                self.doc_term_matrix = csr_matrix(
                    (data, (doc_ids, col_indices)),
                    shape=(self.index.doc_count, self.vocab_size),
                    dtype=np.float32
                )
            except Exception as e:
                raise RuntimeError(f"Failed to build sparse matrix: {e}") from e

    def _precompute_optimized_weights(self):
        with self.profiler.timer("Optimized Weight Precomputation"):
            try:
                doc_count = self.index.doc_count
                
                dfs = np.array([len(docs) for docs in self.index.term_doc_freqs.values()], dtype=np.float32)
                self.idf = np.log10(doc_count / np.maximum(dfs, 1.0))
                
                self.tfidf_matrix = self.doc_term_matrix.multiply(self.idf)
                
                log_data = 1 + np.log10(np.maximum(self.doc_term_matrix.data, 1e-10))
                self.sublinear_matrix = csr_matrix(
                    (log_data, self.doc_term_matrix.indices, self.doc_term_matrix.indptr),
                    shape=self.doc_term_matrix.shape
                ).multiply(self.idf)
            except Exception as e:
                raise RuntimeError(f"Failed to precompute weights: {e}") from e

    def _optimized_cosine_similarity(self, matrix):
        with self.profiler.timer("Matrix Normalization + Similarity"):
            return cosine_similarity(matrix, dense_output=False)

    def find_similar_documents(self, k=10, weighting='tf') -> List[Tuple[str, str, float]]:
        with self.profiler.timer(f"Similarity Calculation ({weighting})"):
            try:
                if weighting == 'tf':
                    matrix = self.doc_term_matrix
                elif weighting == 'tfidf':
                    matrix = self.tfidf_matrix
                elif weighting == 'sublinear':
                    matrix = self.sublinear_matrix
                else:
                    raise ValueError(f"Unknown weighting scheme: {weighting}")

                similarity_matrix = self._optimized_cosine_similarity(matrix)
                similarity_matrix = similarity_matrix.tocoo()
                
                # I tried triu for optimized filtering but only available for CSC matrices
                mask = similarity_matrix.row < similarity_matrix.col
                rows = similarity_matrix.row[mask]
                cols = similarity_matrix.col[mask]
                sims = similarity_matrix.data[mask]
                
                if len(sims) > k:
                    idx = np.argpartition(sims, -k)[-k:]
                    top_indices = idx[np.argsort(-sims[idx])]
                else:
                    top_indices = np.argsort(-sims)[:min(k, len(sims))]
                
                return [
                    (self.index.filenames[rows[i]], 
                    self.index.filenames[cols[i]], 
                    sims[i]
                    ) for i in top_indices
                ]
            except Exception as e:
                self.profiler.log_message(f"Error finding similar documents: {e}")
                return []