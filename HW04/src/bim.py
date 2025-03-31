import math
from typing import List, Tuple, Set, Dict
from contextlib import nullcontext

class RetrievalBIM:
    """
    Binary Independence Model for probabilistic information retrieval.
    """
    
    def __init__(self, index, profiler=None):
        self.index = index
        self.profiler = profiler
        self.relevance_labels: Dict[str, int] = {}
        self.doc_count = self.index.doc_count
    
    def load_relevance_labels(self, filepath: str):
        try:
            with open(filepath.strip().strip('"\''), 'r') as f:
                self.relevance_labels = {
                    line.split(',')[0].strip(): int(line.split(',')[1].strip())
                    for line in f if ',' in line
                }
        except Exception as e:
            print(f"Error loading relevance labels from {filepath}: {e}")
            self.relevance_labels = {}
    
    def get_relevance_label(self, filename: str) -> int:
        normalized_filename = filename.rsplit('.', 1)[0]
        return self.relevance_labels.get(normalized_filename, "?")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        with self.profiler.timer("BIM Search") if self.profiler else nullcontext():
            processed_query = self._process_query(query)
            if not processed_query:
                return []
            
            candidate_docs = self._get_candidate_documents(processed_query)
            if not candidate_docs:
                return []
            
            document_scores = [
                (self.index.filenames[doc_id], self._compute_rsv(processed_query, doc_id))
                for doc_id in candidate_docs
            ]
            return sorted(document_scores, key=lambda x: x[1], reverse=True)[:k]
    
    def _compute_rsv(self, query_terms: List[str], doc_id: int) -> float:
        doc_terms = self.index.doc_term_freqs.get(doc_id, {})
        return sum(
            self._calculate_term_weight(term)
            for term in query_terms if term in doc_terms
        )
    
    def _calculate_term_weight(self, term: str) -> float:
        df = len(self.index.term_doc_freqs.get(term, {})) # Document frequency
        N = self.doc_count # Total number of documents
        s, S = 0.5, 1.0 # Default values for s and S
        p_t = s / S # Probability of term t in the collection
        u_t = (df + 0.5) / (N + 1) # Probability of term t in the document
        if u_t == 1.0: # Avoid log(0)
            u_t = 0.9999 # Adjust to avoid log(0)
        return math.log10((p_t * (1 - u_t)) / ((1 - p_t) * u_t)) # RSV(t, d)
    
    def _get_candidate_documents(self, query_terms: List[str]) -> Set[int]:
        return {
            doc_id
            for term in query_terms
            if term in self.index.term_doc_freqs
            for doc_id in self.index.term_doc_freqs[term].keys()
        }
    
    def _process_query(self, query: str) -> List[str]:
        try:
            return self.index._preprocess_text(query)
        except Exception as e:
            print(f"Error processing query: {e}")
            return []