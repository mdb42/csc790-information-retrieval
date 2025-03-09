# src/vsm/hybrid_vsm.py
"""
Scenario:
- standard library only
- parallelization only where it measurably benefits performance (similarities)
"""
import math
import heapq
import multiprocessing
from src.vsm import BaseVSM

def _compute_similarities_for_doc(args):
    i, weight_vectors, magnitudes, total_docs = args
    vec_i = weight_vectors[i]
    mag_i = magnitudes[i]
    results = []
    for j in range(i + 1, total_docs):
        vec_j = weight_vectors[j]
        mag_j = magnitudes[j]
        common_terms = set(vec_i.keys()) & set(vec_j.keys())
        if not common_terms:
            continue
        dot = sum(vec_i[t] * vec_j[t] for t in common_terms)
        sim = dot / (mag_i * mag_j) if mag_i * mag_j != 0 else 0.0
        if sim > 0:
            results.append((i, j, sim))
    return results

class HybridVSM(BaseVSM):
    def __init__(self, index, profiler=None):
        super().__init__(index, profiler)
        self.idf_values = {}
        self._compute_idf_values()

    def _compute_idf_values(self):
        total_docs = self.index.doc_count
        # Compute idf values for each term
        self.idf_values = {
            term: math.log10(total_docs / max(len(docs), 1))
            for term, docs in self.index.term_doc_freqs.items()
        }

    def build_model(self):
        with self.profiler.timer("Weight Precomputation (Serial)"):
            for doc_id, term_counts in self.index.doc_term_freqs.items():
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
                self.magnitudes['tf'][doc_id] = math.sqrt(sum(v**2 for v in tf.values()))
                self.magnitudes['tfidf'][doc_id] = math.sqrt(sum(v**2 for v in tfidf.values()))
                self.magnitudes['sublinear'][doc_id] = math.sqrt(sum(v**2 for v in sublinear.values()))

    def find_similar_documents(self, k=10, weighting='tf'):
        similarities = []
        total_docs = self.index.doc_count
        weight_vectors = [self.weights[weighting][i] for i in range(total_docs)]
        magnitudes = [self.magnitudes[weighting][i] for i in range(total_docs)]
        with self.profiler.timer(f"Parallel Similarity Calculation ({weighting})"):
            tasks = [
                (i, weight_vectors, magnitudes, total_docs)
                for i in range(total_docs)
            ]
            with multiprocessing.Pool() as pool:
                results = pool.map(_compute_similarities_for_doc, tasks)
            for res in results:
                similarities.extend(res)
        doc_pairs = [
            (self.index.filenames[i], self.index.filenames[j], sim)
            for i, j, sim in similarities
        ]
        return heapq.nlargest(k, doc_pairs, key=lambda x: x[2])
