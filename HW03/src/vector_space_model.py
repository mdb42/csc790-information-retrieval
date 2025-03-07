# src/vector_space_model.py
import math
import pickle
import os
import heapq

class VectorSpaceModel:
    def __init__(self, parser, index_file=None, use_existing_index=False, profiler=None):
        self.parser = parser
        self.index_file = index_file
        self.use_existing_index = use_existing_index
        self.filenames = []
        self.term_doc_freqs = {}
        self.doc_term_freqs = {}
        self.idf_values = {}
        if profiler is None:
            from src.performance_monitoring import Profiler
            profiler = Profiler()
        self.profiler = profiler

        if use_existing_index and index_file and os.path.exists(index_file):
            self.load_index(index_file)
        else:
            self._build_index()
            if index_file:
                self.save_index(index_file)

    def _build_index(self):
        with self.profiler.timer("Document Parsing"):
            self.filenames, term_freqs = self.parser.process_documents()

        self.doc_count = len(self.filenames)
        self.term_doc_freqs = {}
        self.doc_term_freqs = {}

        for doc_id, term_counts in enumerate(term_freqs):
            self.doc_term_freqs[doc_id] = term_counts
            for term, freq in term_counts.items():
                if term not in self.term_doc_freqs:
                    self.term_doc_freqs[term] = {}
                self.term_doc_freqs[term][doc_id] = freq

        self.vocab_size = len(self.term_doc_freqs)
        self._compute_idf_values()
        self._precompute_weights()
        self.profiler.log(f"Indexed {self.doc_count} documents with {self.vocab_size} unique terms.")

    def _compute_idf_values(self):
        self.idf_values = {
            term: math.log10(self.doc_count / max(len(doc_freqs), 1))
            for term, doc_freqs in self.term_doc_freqs.items()
        }

    def _precompute_weights(self):
        self.weights = {'tf': {}, 'tfidf': {}, 'sublinear': {}}
        self.magnitudes = {'tf': {}, 'tfidf': {}, 'sublinear': {}}

        for doc_id, term_counts in self.doc_term_freqs.items():
            tf_weights = {}
            tfidf_weights = {}
            sublinear_weights = {}

            for term, freq in term_counts.items():
                tf_weights[term] = freq
                idf = self.idf_values.get(term, 0)
                tfidf_weights[term] = freq * idf
                sublinear_weights[term] = (1 + math.log10(freq)) * idf if freq > 0 else 0

            self.weights['tf'][doc_id] = tf_weights
            self.weights['tfidf'][doc_id] = tfidf_weights
            self.weights['sublinear'][doc_id] = sublinear_weights

            self.magnitudes['tf'][doc_id] = math.sqrt(sum(w**2 for w in tf_weights.values()))
            self.magnitudes['tfidf'][doc_id] = math.sqrt(sum(w**2 for w in tfidf_weights.values()))
            self.magnitudes['sublinear'][doc_id] = math.sqrt(sum(w**2 for w in sublinear_weights.values()))

    def _compute_vector_similarity(self, doc1_id, doc2_id, weighting='tf'):
        weights1 = self.weights[weighting][doc1_id]
        weights2 = self.weights[weighting][doc2_id]

        common_terms = set(weights1.keys()) & set(weights2.keys())
        if not common_terms:
            return 0.0

        dot_product = sum(weights1[term] * weights2[term] for term in common_terms)
        magnitude1 = self.magnitudes[weighting][doc1_id]
        magnitude2 = self.magnitudes[weighting][doc2_id]

        return dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 != 0 else 0.0

    def find_similar_documents(self, k=10, weighting='tf'):
        with self.profiler.timer(f"Similarity calculation ({weighting})"):
            similarities = []

            for i in range(self.doc_count):
                for j in range(i + 1, self.doc_count):
                    sim = self._compute_vector_similarity(i, j, weighting)
                    if sim > 0:
                        similarities.append((i, j, sim))

            top_pairs = heapq.nlargest(k, similarities, key=lambda x: x[2])
            self.profiler.log(f"Computed top-{k} similar documents using {weighting} weighting.")
            return [(self.filenames[doc1_id], self.filenames[doc2_id], similarity) for doc1_id, doc2_id, similarity in top_pairs]

    def get_most_frequent_terms(self, n=10):
        term_totals = {term: sum(freqs.values()) for term, freqs in self.term_doc_freqs.items()}
        return sorted(term_totals.items(), key=lambda x: x[1], reverse=True)[:n]

    def save_index(self, filepath):
        with self.profiler.timer("Saving Index"):
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'filenames': self.filenames,
                    'term_doc_freqs': self.term_doc_freqs,
                    'doc_term_freqs': self.doc_term_freqs,
                    'idf_values': self.idf_values
                }, f)

    def load_index(self, filepath):
        with self.profiler.timer("Loading Index"):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.filenames = data['filenames']
                self.term_doc_freqs = data['term_doc_freqs']
                self.doc_term_freqs = data['doc_term_freqs']
                self.idf_values = data['idf_values']
                self.vocab_size = len(self.term_doc_freqs)


