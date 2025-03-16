# src/retrieval_bm25.py
import math
import heapq
from typing import List, Tuple, Dict, Set
from src.profiler import Profiler
from src.index import BaseIndex

class RetrievalBM25:
    def __init__(self, index: BaseIndex, profiler: Profiler = None, k1=1.2, b=0.75):
        self.index = index
        self.profiler = profiler
        self.k1 = k1 # term frequency saturation parameter
        self.b = b # length normalization parameter
        
        self.doc_lengths, self.avg_doc_length = self.index.get_document_lengths()
        
        self.idf_values = self._compute_idf_values()
    
    def _compute_idf_values(self):
        """
        Uses the classic IDF formula:
        IDF(t) = log(N/df_t)
        
        Note: Some BM25 implementations use a smoothed version:
        IDF(t) = log((N - df_t + 0.5) / (df_t + 0.5))
        
        It seems preferred, but I don't see it in the slides?
        """
        total_docs = max(1, self.index.doc_count)  # Avoid division by zero
        idf_values = {}
        
        for term, docs in self.index.term_doc_freqs.items():
            df = len(docs)
            if df > 0:  # Avoid division by zero
                idf_values[term] = math.log10(total_docs / df)
            else:
                idf_values[term] = 0
        
        return idf_values
    
    def score_document(self, query_terms: List[str], doc_id: int) -> float:
        # Quick check if document exists
        if doc_id not in self.doc_lengths:
            return 0.0
            
        score = 0.0
        doc_term_freqs = self.index.doc_term_freqs.get(doc_id, {})
        doc_length = self.doc_lengths[doc_id]
        
        # Avoid division by zero
        if self.avg_doc_length == 0:
            normalized_length = 1.0
        else:
            normalized_length = doc_length / self.avg_doc_length
        
        for term in query_terms:
            if term not in doc_term_freqs:
                continue
                
            tf = doc_term_freqs[term]
            # Only compute score for terms with positive IDF
            idf = self.idf_values.get(term, 0)
            if idf <= 0:
                continue
                
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * normalized_length)
            # Avoid division by zero
            if denominator == 0:
                continue
                
            term_score = idf * (numerator / denominator)
            score += term_score
            
        return score
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        if self.profiler:
            with self.profiler.timer(f"BM25 Search"):
                return self._execute_search(query, k)
        else:
            return self._execute_search(query, k)
    
    def _execute_search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        # Process the query
        processed_query = self._process_query(query)
        
        if not processed_query:
            return []
        
        # Find candidate documents (those containing at least one query term)
        candidate_docs = self._get_candidate_documents(processed_query)
        
        if not candidate_docs:
            return []
        
        # Score candidate documents
        document_scores = []
        for doc_id in candidate_docs:
            score = self.score_document(processed_query, doc_id)
            if score > 0:
                # Use negative score for min-heap to get top-k
                if len(document_scores) < k:
                    heapq.heappush(document_scores, (score, doc_id))
                elif score > document_scores[0][0]:
                    heapq.heappushpop(document_scores, (score, doc_id))
        
        # Sort by score (descending)
        document_scores.sort(reverse=True)
        
        # Convert document IDs to filenames
        results = [(self.index.filenames[doc_id], score) 
                  for score, doc_id in document_scores]
        
        return results
    
    def _get_candidate_documents(self, query_terms: List[str]) -> Set[int]:
        candidate_docs = set()
        for term in query_terms:
            if term in self.index.term_doc_freqs:
                candidate_docs.update(self.index.term_doc_freqs[term].keys())
        return candidate_docs
    
    def _process_query(self, query: str) -> List[str]:
        try:
            # Use the index's preprocessing method for consistency
            return self.index._preprocess_text(query)
        except Exception as e:
            if self.profiler:
                self.profiler.log_message(f"Error processing query: {e}")
            # Return empty list on error
            return []