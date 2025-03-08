# src/vsm/parallel_vsm.py
"""
Scenario:
- standard library only
- parallelization where possible

=== Timing Breakdown ===
Document Processing: 1.9740s
Index Building: 0.0102s
Parallel Weight Precomputation: 3.5328s
Parallel Similarity Calculation (tf): 4.2469s
Parallel Similarity Calculation (tfidf): 4.2716s
Parallel Similarity Calculation (sublinear): 4.3384s

Tracked Operations Total: 18.3739s
"""
import math
import heapq
import multiprocessing
from src.vsm.base import BaseVSM

def _compute_doc_weights(args):
    doc_id, term_counts, idf_values = args
    tf = {}
    tfidf = {}
    sublinear = {}
    for term, freq in term_counts.items():
        tf[term] = freq
        idf = idf_values.get(term, 0)
        tfidf[term] = freq * idf
        sublinear[term] = (1 + math.log10(freq)) * idf if freq > 0 else 0
    mag_tf = math.sqrt(sum(v * v for v in tf.values()))
    mag_tfidf = math.sqrt(sum(v * v for v in tfidf.values()))
    mag_sublinear = math.sqrt(sum(v * v for v in sublinear.values()))
    return (doc_id, tf, tfidf, sublinear, mag_tf, mag_tfidf, mag_sublinear)

def _compute_similarities_for_doc(args):
    i, weight_vectors, magnitudes, total_docs = args
    vec_i = weight_vectors[i]
    mag_i = magnitudes[i]
    results = []
    for j in range(i + 1, total_docs):
        vec_j = weight_vectors[j]
        mag_j = magnitudes[j]
        # Find common terms between documents i and j.
        common = set(vec_i.keys()) & set(vec_j.keys())
        if not common:
            continue
        dot = sum(vec_i[t] * vec_j[t] for t in common)
        sim = dot / (mag_i * mag_j) if mag_i * mag_j != 0 else 0.0
        if sim > 0:
            results.append((i, j, sim))
    return results

class ParallelVSM(BaseVSM):
    def __init__(self, index, profiler=None):
        super().__init__(index, profiler)
        self.idf_values = {}
        self._compute_idf_values()

    def _compute_idf_values(self):
        total_docs = self.index.doc_count
        # Compute the inverse document frequency for each term.
        self.idf_values = {
            term: math.log10(total_docs / max(len(docs), 1))
            for term, docs in self.index.term_doc_freqs.items()
        }

    def build_model(self):
        with self.profiler.timer("Parallel Weight Precomputation"):
            # Prepare tasks for each document.
            tasks = [
                (doc_id, term_counts, self.idf_values)
                for doc_id, term_counts in self.index.doc_term_freqs.items()
            ]
            with multiprocessing.Pool() as pool:
                results = pool.map(_compute_doc_weights, tasks)
            # Save computed weights and magnitudes.
            for doc_id, tf, tfidf, sublinear, mag_tf, mag_tfidf, mag_sublinear in results:
                self.weights['tf'][doc_id] = tf
                self.weights['tfidf'][doc_id] = tfidf
                self.weights['sublinear'][doc_id] = sublinear
                self.magnitudes['tf'][doc_id] = mag_tf
                self.magnitudes['tfidf'][doc_id] = mag_tfidf
                self.magnitudes['sublinear'][doc_id] = mag_sublinear

    def find_similar_documents(self, k=10, weighting='tf'):
        similarities = []
        total_docs = self.index.doc_count
        # Convert the weight dictionaries and magnitudes into ordered lists (assumed doc_ids 0 to n-1).
        weight_vectors = [self.weights[weighting][i] for i in range(total_docs)]
        magnitudes = [self.magnitudes[weighting][i] for i in range(total_docs)]
        with self.profiler.timer(f"Parallel Similarity Calculation ({weighting})"):
            # Create a task for each document index i.
            tasks = [
                (i, weight_vectors, magnitudes, total_docs)
                for i in range(total_docs)
            ]
            with multiprocessing.Pool() as pool:
                # Each task returns a list of similarity tuples.
                results = pool.map(_compute_similarities_for_doc, tasks)
            for res in results:
                similarities.extend(res)
        # Convert document indices to filenames.
        doc_pairs = [
            (self.index.filenames[i], self.index.filenames[j], sim)
            for i, j, sim in similarities
        ]
        return heapq.nlargest(k, doc_pairs, key=lambda x: x[2])
