# src/vsm/sparse_vsm.py
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from src.vsm.base_vsm import BaseVSM
from src.performance_monitoring import Profiler
from src.index.memory_index import MemoryIndex

class SparseVSM(BaseVSM):
    def __init__(self, index: MemoryIndex, profiler: Profiler = None):
        super().__init__(index, profiler)
        self._build_optimized_sparse_matrix()
        self._precompute_optimized_weights()

    def _build_optimized_sparse_matrix(self):
        with self.profiler.timer("Optimized Sparse Matrix Construction"):
            self.term_to_idx = {term: idx for idx, term in enumerate(self.index.term_doc_freqs)}
            self.vocab_size = len(self.term_to_idx)
            
            doc_ids = []
            col_indices = []
            data = []
            
            for doc_id, term_counts in self.index.doc_term_freqs.items():
                terms = list(term_counts.keys())
                doc_ids.extend([doc_id] * len(terms))
                col_indices.extend([self.term_to_idx[term] for term in terms])
                data.extend(term_counts.values())
            
            self.doc_term_matrix = csr_matrix(
                (data, (doc_ids, col_indices)),
                shape=(self.index.doc_count, self.vocab_size),
                dtype=np.float32
            )

    def _precompute_optimized_weights(self):
        with self.profiler.timer("Optimized Weight Precomputation"):
            doc_count = self.index.doc_count
            dfs = np.array([len(docs) for docs in self.index.term_doc_freqs.values()], dtype=np.float32)
            self.idf = np.log10(doc_count / np.maximum(dfs, 1.0))
            
            self.tfidf_matrix = self.doc_term_matrix.multiply(self.idf)
            
            log_data = 1 + np.log10(self.doc_term_matrix.data)
            self.sublinear_matrix = csr_matrix(
                (log_data, self.doc_term_matrix.indices, self.doc_term_matrix.indptr),
                shape=self.doc_term_matrix.shape
            ).multiply(self.idf)

    def _optimized_cosine_similarity(self, matrix):
        with self.profiler.timer("Matrix Normalization + Similarity"):
            # sklearn's cosine_similarity handles normalization internally
            return cosine_similarity(matrix, dense_output=False)

    def find_similar_documents(self, k=10, weighting='tf') -> List[Tuple[str, str, float]]:
        with self.profiler.timer(f"Similarity Calculation ({weighting})"):
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
            mask = similarity_matrix.row < similarity_matrix.col
            rows = similarity_matrix.row[mask]
            cols = similarity_matrix.col[mask]
            sims = similarity_matrix.data[mask]
            
            if len(sims) > k:
                idx = np.argpartition(sims, -k)[-k:]
                top_indices = idx[np.argsort(-sims[idx])]
            else:
                top_indices = np.argsort(-sims)
            
            return [
                (self.index.filenames[rows[i]], 
                self.index.filenames[cols[i]], 
                sims[i]
            ) for i in top_indices[:k]]