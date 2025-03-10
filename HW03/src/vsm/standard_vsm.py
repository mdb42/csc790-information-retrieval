# src/vsm/standard_vsm.py
"""
Scenario:
- standard library only
- no parallelization

New:
- merged separate loops for calculating weights and magnitudes
- replaced explicit set creation (set(vec1.keys()) & set(vec2.keys())) with dictionary views (vec1.keys() & vec2.keys()) for common term calculation
- added checks to skip documents with zero magnitude early in the process
- used a min-heap for tracking only the top-k results
- batch processing to improve cache locality
"""
import math
import heapq
from typing import List, Tuple, Dict, Any
from src.performance_monitoring import Profiler
from src.index import BaseIndex
from src.vsm import BaseVSM

class StandardVSM(BaseVSM):
    def __init__(self, index: BaseIndex, profiler: Profiler = None):
        super().__init__(index, profiler)
        self.idf_values = {}
        self._compute_idf_values()
        self.build_model()

    def _compute_idf_values(self):
        total_docs = self.index.doc_count
        self.idf_values = {
            term: math.log10(total_docs / max(len(docs), 1))
            for term, docs in self.index.term_doc_freqs.items()
        }

    def build_model(self):
        with self.profiler.timer("Weight Precomputation"):
            for doc_id in self.index.doc_term_freqs:
                try:
                    term_counts = self.index.doc_term_freqs[doc_id]
                    
                    tf = {}
                    tfidf = {}
                    sublinear = {}
                    tf_mag_sq = 0
                    tfidf_mag_sq = 0
                    sublinear_mag_sq = 0
                    
                    for term, freq in term_counts.items():
                        tf[term] = freq
                        tf_mag_sq += freq * freq
                        
                        idf = self.idf_values.get(term, 0)
                        
                        tfidf_val = freq * idf
                        tfidf[term] = tfidf_val
                        tfidf_mag_sq += tfidf_val * tfidf_val
                        
                        sublinear_val = (1 + math.log10(freq)) * idf if freq > 0 else 0
                        sublinear[term] = sublinear_val
                        sublinear_mag_sq += sublinear_val * sublinear_val
                    
                    self.weights['tf'][doc_id] = tf
                    self.weights['tfidf'][doc_id] = tfidf
                    self.weights['sublinear'][doc_id] = sublinear
                    
                    self.magnitudes['tf'][doc_id] = math.sqrt(tf_mag_sq)
                    self.magnitudes['tfidf'][doc_id] = math.sqrt(tfidf_mag_sq)
                    self.magnitudes['sublinear'][doc_id] = math.sqrt(sublinear_mag_sq)
                except Exception as e:
                    self.profiler.log_message(f"Error processing document {doc_id}: {e}")

    def _compute_similarity(self, doc1: int, doc2: int, weighting: str) -> float:
        try:
            vec1 = self.weights[weighting][doc1]
            vec2 = self.weights[weighting][doc2]
            
            common_terms = vec1.keys() & vec2.keys()
            
            if not common_terms:
                return 0.0
                
            dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
            mag1 = self.magnitudes[weighting][doc1]
            mag2 = self.magnitudes[weighting][doc2]
            
            return dot_product / (mag1 * mag2) if mag1 * mag2 != 0 else 0.0
        except KeyError:
            return 0.0

    def find_similar_documents(self, k=10, weighting='tf') -> List[Tuple[str, str, float]]:
        """
        O(n²) complexity - Unavoidable for pairwise similarity calculations
        """
        top_similarities = []
        total_docs = self.index.doc_count
        batch_size = min(1000, max(100, int(math.sqrt(total_docs))))
        
        with self.profiler.timer(f"Similarity Calculation ({weighting})"):
            for batch_start_i in range(0, total_docs, batch_size):
                batch_end_i = min(batch_start_i + batch_size, total_docs)
                
                for i in range(batch_start_i, batch_end_i):
                    if i not in self.magnitudes[weighting] or self.magnitudes[weighting][i] == 0:
                        continue
                    
                    vec_i = self.weights[weighting][i]
                    mag_i = self.magnitudes[weighting][i]
                    
                    for j in range(i+1, total_docs):
                        if j not in self.magnitudes[weighting] or self.magnitudes[weighting][j] == 0:
                            continue
                        
                        vec_j = self.weights[weighting][j]
                        mag_j = self.magnitudes[weighting][j]
                        
                        common_terms = vec_i.keys() & vec_j.keys()
                        
                        if not common_terms:
                            continue
                        
                        dot_product = sum(vec_i[term] * vec_j[term] for term in common_terms)
                        sim = dot_product / (mag_i * mag_j) if mag_i * mag_j != 0 else 0.0
                        
                        if sim > 0:
                            if len(top_similarities) < k:
                                heapq.heappush(top_similarities, (sim, i, j))
                            elif sim > top_similarities[0][0]:
                                heapq.heappushpop(top_similarities, (sim, i, j))
        
        return [
            (self.index.filenames[i], self.index.filenames[j], sim)
            for sim, i, j in sorted(top_similarities, reverse=True)
        ]