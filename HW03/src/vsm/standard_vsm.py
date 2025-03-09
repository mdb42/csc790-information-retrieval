# src/vsm/standard_vsm.py
"""
Scenario:
- standard library only
- no parallelization
"""
import math
import heapq
from typing import List, Tuple
from src.performance_monitoring import Profiler
from src.index import MemoryIndex
from src.vsm import BaseVSM

class StandardVSM(BaseVSM):
    def __init__(self, index: MemoryIndex, profiler: Profiler = None):
        super().__init__(index, profiler)
        self.idf_values = {}
        self._compute_idf_values()

    def _compute_idf_values(self):
        total_docs = self.index.doc_count
        self.idf_values = {
            term: math.log10(total_docs / max(len(docs), 1))
            for term, docs in self.index.term_doc_freqs.items()
        }

    def build_model(self):
        with self.profiler.timer("Weight Precomputation"):
            for doc_id in self.index.doc_term_freqs:
                term_counts = self.index.doc_term_freqs[doc_id]
                
                tf = {}
                tfidf = {}
                sublinear = {}
                
                for term, freq in term_counts.items():
                    tf[term] = freq
                    idf = self.idf_values.get(term, 0)
                    tfidf[term] = freq * idf
                    sublinear[term] = (1 + math.log10(freq)) * idf if freq > 0 else 0
                
                self.weights['tf'][doc_id] = tf
                self.weights['tfidf'][doc_id] = tfidf
                self.weights['sublinear'][doc_id] = sublinear
                
                # Precompute magnitudes
                self.magnitudes['tf'][doc_id] = math.sqrt(sum(v**2 for v in tf.values()))
                self.magnitudes['tfidf'][doc_id] = math.sqrt(sum(v**2 for v in tfidf.values()))
                self.magnitudes['sublinear'][doc_id] = math.sqrt(sum(v**2 for v in sublinear.values()))

    def _compute_similarity(self, doc1: int, doc2: int, weighting: str) -> float:
        vec1 = self.weights[weighting][doc1]
        vec2 = self.weights[weighting][doc2]
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        if not common_terms:
            return 0.0
            
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        mag1 = self.magnitudes[weighting][doc1]
        mag2 = self.magnitudes[weighting][doc2]
        
        return dot_product / (mag1 * mag2) if mag1 * mag2 != 0 else 0.0

    def find_similar_documents(self, k=10, weighting='tf') -> List[Tuple[str, str, float]]:
        """
        O(n²) complexity - Unavoidable for pairwise similarity calculations
        """
        similarities = []
        total_docs = self.index.doc_count
        
        with self.profiler.timer(f"Similarity Calculation ({weighting})"):
            for i in range(total_docs):
                for j in range(i+1, total_docs):
                    sim = self._compute_similarity(i, j, weighting)
                    if sim > 0:
                        similarities.append((
                            self.index.filenames[i],
                            self.index.filenames[j],
                            sim
                        ))
        
        return heapq.nlargest(k, similarities, key=lambda x: x[2])