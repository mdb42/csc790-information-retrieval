# src/vsm/parallel_vsm.py
"""
Parallel Vector Space Model
Author: Matthew Branson
Date: March 14, 2025

This module implements a parallelized version of the Vector Space Model for document similarity
calculations. It's optimized for medium to large document collections. The optional
parallelize_weights flag can be used to parallelize the weight computation step, but it's
typically not beneficial under normal circumstances.
"""
import math
import heapq
import multiprocessing
from typing import List, Tuple, Dict, Any
from src.profiler import Profiler
from src.index import BaseIndex
from src.vsm import BaseVSM


def _compute_weights_for_doc(term_counts: Dict[str, int],
                             idf_values: Dict[str, float]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Compute the weights for a single document across all schemes (tf, tfidf, sublinear)
    and calculate their corresponding magnitudes.

    Args:
        term_counts: Dict mapping terms to their frequency in the document.
        idf_values: Dict mapping terms to their inverse document frequency values.
    
    Returns:
        A tuple:
            - weights: Dict mapping each scheme to its term-to-weight dict.
            - magnitudes: Dict mapping each scheme to the computed magnitude.
    """
    tf = {}
    tfidf = {}
    sublinear = {}
    tf_sq = tfidf_sq = sublinear_sq = 0
    
    for term, freq in term_counts.items():
        # Term Frequency (TF)
        tf[term] = freq
        tf_sq += freq * freq
        
        # TF-IDF
        idf = idf_values.get(term, 0)
        tfidf_val = freq * idf
        tfidf[term] = tfidf_val
        tfidf_sq += tfidf_val * tfidf_val
        
        # Sublinear TF-IDF
        sublinear_val = (1 + math.log10(freq)) * idf if freq > 0 else 0
        sublinear[term] = sublinear_val
        sublinear_sq += sublinear_val * sublinear_val

    weights = {"tf": tf, "tfidf": tfidf, "sublinear": sublinear}
    magnitudes = {
        "tf": math.sqrt(tf_sq),
        "tfidf": math.sqrt(tfidf_sq),
        "sublinear": math.sqrt(sublinear_sq)
    }
    return weights, magnitudes


def _compute_doc_weights(args):
    """
    Compute document weights and magnitudes either for a single document or for a batch.
    
    Args:
        args: If first element is a list, then it's a batch call: 
              (doc_ids, doc_term_freqs, idf_values)
              Otherwise, it's a single document call: (doc_id, term_counts, idf_values)
              
    Returns:
        For batch: a dict containing weights and magnitudes for each scheme.
        For single document: a tuple containing (doc_id, tf, tfidf, sublinear, mag_tf, mag_tfidf, mag_sublinear)
    """
    if isinstance(args[0], list):
        # Batch processing (Hypothetically helpful if you have an absurd amount of documents and cores)
        doc_ids, doc_term_freqs, idf_values = args
        batch_weights = {'tf': {}, 'tfidf': {}, 'sublinear': {}}
        batch_magnitudes = {'tf': {}, 'tfidf': {}, 'sublinear': {}}
        
        for doc_id in doc_ids:
            weights, mags = _compute_weights_for_doc(doc_term_freqs[doc_id], idf_values)
            for scheme in weights:
                batch_weights[scheme][doc_id] = weights[scheme]
                batch_magnitudes[scheme][doc_id] = mags[scheme]
        return {'weights': batch_weights, 'magnitudes': batch_magnitudes}
    else:
        # Single document processing (Recommended for all realistic use cases)
        doc_id, term_counts, idf_values = args
        weights, mags = _compute_weights_for_doc(term_counts, idf_values)
        return (doc_id, weights['tf'], weights['tfidf'], weights['sublinear'],
                mags['tf'], mags['tfidf'], mags['sublinear'])


def _cosine_similarity(vec_i: Dict[str, float], mag_i: float,
                       vec_j: Dict[str, float], mag_j: float) -> float:
    """
    Compute cosine similarity between two document vectors.

    Args:
        vec_i: Term weights for document i
        mag_i: Magnitude of document i
        vec_j: Term weights for document j
        mag_j: Magnitude of document j

    Returns:
        Cosine similarity score between the two documents.
    """
    if mag_i == 0 or mag_j == 0:
        return 0.0
    common_terms = vec_i.keys() & vec_j.keys()
    if not common_terms:
        return 0.0
    dot = sum(vec_i[t] * vec_j[t] for t in common_terms)
    return dot / (mag_i * mag_j)


def _compute_similarities_for_chunk(args):
    """
    Compute similarity scores for a chunk of document pairs.
    
    Args:
        args: A tuple containing:
            - chunk_id: Identifier for this chunk
            - start_i/end_i: Start/end indices for the first set of documents
            - weights_i/magnitudes_i: Weight vectors and magnitudes for first set
            - start_j/end_j: Start/end indices for the second set of documents
            - weights_j/magnitudes_j: Weight vectors and magnitudes for second set
            - filenames: List of document filenames
            
    Returns:
        List of (doc1, doc2, similarity) tuples.
    """
    (chunk_id, start_i, end_i, weights_i, magnitudes_i,
     start_j, end_j, weights_j, magnitudes_j, filenames) = args
    results = []
    
    for i_idx, i in enumerate(range(start_i, end_i)):
        if i_idx >= len(magnitudes_i) or magnitudes_i[i_idx] == 0:
            continue
        vec_i = weights_i[i_idx]
        mag_i = magnitudes_i[i_idx]
        
        # Ensure we only compare each pair once
        j_start = max(start_j, i + 1)
        j_idx_offset = j_start - start_j
        
        for j_idx, j in enumerate(range(j_start, end_j)):
            real_j_idx = j_idx + j_idx_offset
            if real_j_idx >= len(magnitudes_j) or magnitudes_j[real_j_idx] == 0:
                continue
            sim = _cosine_similarity(vec_i, mag_i, weights_j[real_j_idx], magnitudes_j[real_j_idx])
            if sim > 0:
                results.append((filenames[i], filenames[j], sim))
    return results


class ParallelVSM(BaseVSM):
    """
    Parallel Vector Space Model implementation using multiprocessing.
    
    This implementation divides the work into chunks that can be processed in parallel.

    Attributes:
        index (BaseIndex): Document index containing term frequencies and document info
        profiler (Profiler): Performance monitoring utility for timing operations
        num_processes (int): Number of parallel processes to use
        chunk_size (int): Number of documents to process in each chunk
        idf_values (Dict): Inverse document frequency values for each term
    """
    
    def __init__(self, index: BaseIndex, profiler: Profiler = None, 
                 num_processes: int = None, chunk_size: int = None,
                 parallelize_weights: bool = False):
        """
        Initialize the ParallelVSM.
        """
        super().__init__(index, profiler)
        self.num_processes = num_processes or max(1, multiprocessing.cpu_count() - 1)
        self.chunk_size = chunk_size
        self.idf_values = {}
        
        self._compute_idf_values()
        self.build_model(parallelize_weights=parallelize_weights)

    def _compute_idf_values(self):
        """
        Compute inverse document frequency (IDF) values.
        IDF = log10(total_docs / doc_frequency)
        """
        total_docs = self.index.doc_count
        self.idf_values = {
            term: math.log10(total_docs / max(len(docs), 1))
            for term, docs in self.index.term_doc_freqs.items()
        }

    def build_model(self, parallelize_weights=False):
        """
        Build the vector space model by computing term weights and magnitudes.

        Args:
            parallelize_weights (bool): Whether to parallelize weight computation
        """
        if parallelize_weights:
            if self.index.doc_count < 10000:
                self.profiler.log_message(
                    f"Warning: Parallel weight computation requested for small collection "
                    f"({self.index.doc_count} documents). This may not improve performance."
                )
            self._build_model_parallel()
        else:
            self._build_model_serial()

    def _build_model_serial(self):
        """
        Build the model using serial processing.
        """
        with self.profiler.timer("Weight Precomputation (Serial)"):
            try:
                for doc_id, term_counts in self.index.doc_term_freqs.items():
                    weights, mags = _compute_weights_for_doc(term_counts, self.idf_values)
                    for scheme in weights:
                        self.weights[scheme][doc_id] = weights[scheme]
                        self.magnitudes[scheme][doc_id] = mags[scheme]
            except Exception as e:
                self.profiler.log_message(f"Error in serial weight computation: {e}")

    def _build_model_parallel(self):
        """
        Build the model using parallel processing.
        """
        with self.profiler.timer("Weight Precomputation (Parallel)"):
            try:
                self.profiler.log_message(
                    f"Parallel weight computation started with {self.num_processes} processes "
                    f"for {self.index.doc_count} documents"
                )
                doc_batches = []
                batch_size = max(50, min(1000, self.index.doc_count // (self.num_processes * 2)))
                doc_ids = list(self.index.doc_term_freqs.keys())
                
                for i in range(0, len(doc_ids), batch_size):
                    batch_doc_ids = doc_ids[i:i+batch_size]
                    batch_data = (
                        batch_doc_ids,
                        {doc_id: self.index.doc_term_freqs[doc_id] for doc_id in batch_doc_ids},
                        self.idf_values
                    )
                    doc_batches.append(batch_data)
                
                self.profiler.log_message(f"Created {len(doc_batches)} document batches of size ~{batch_size}")
                
                num_processes = min(self.num_processes, len(doc_batches))
                with multiprocessing.Pool(processes=num_processes) as pool:
                    batch_results = pool.map(_compute_doc_weights, doc_batches)
                
                for batch_result in batch_results:
                    for scheme in ['tf', 'tfidf', 'sublinear']:
                        self.weights[scheme].update(batch_result['weights'][scheme])
                        self.magnitudes[scheme].update(batch_result['magnitudes'][scheme])
                
                self.profiler.log_message("Parallel weight computation completed successfully")
            except Exception as e:
                self.profiler.log_message(f"Error in parallel weight computation: {e}")
                self.profiler.log_message("Falling back to serial weight computation")
                self._fallback_build_model()

    def _fallback_build_model(self):
        """
        Fallback to serial weight computation if parallel fails.
        """
        self.profiler.log_message("Using fallback sequential weight computation")
        self._build_model_serial()

    def find_similar_documents(self, k=10, weighting='tf') -> List[Tuple[str, str, float]]:
        """
        Find the top k most similar document pairs using parallel processing.

        Args:
            k (int): Number of top document pairs to return
            weighting (str): Weighting scheme to use ('tf', 'tfidf', or 'sublinear')

        Returns:
            List of (doc1, doc2, similarity) tuples.
        """
        total_docs = self.index.doc_count
        if weighting not in self.weights:
            raise ValueError(f"Unknown weighting scheme: {weighting}")
        
        with self.profiler.timer(f"Chunked Parallel Similarity Calculation ({weighting})"):
            try:
                valid_docs = [i for i in range(total_docs) 
                              if i in self.magnitudes[weighting] and self.magnitudes[weighting][i] > 0]
                if len(valid_docs) <= 1:
                    return []
                
                if self.chunk_size is None:
                    self.chunk_size = min(2000, max(500, len(valid_docs) // (self.num_processes * 2)))
                    self.profiler.log_message(f"Using chunk size: {self.chunk_size} for {len(valid_docs)} valid documents")
                
                top_similarities = []
                chunk_tasks = []
                chunk_id = 0
                
                # Create chunk tasks by dividing valid document indices
                for chunk_start_i in range(0, len(valid_docs), self.chunk_size):
                    chunk_end_i = min(chunk_start_i + self.chunk_size, len(valid_docs))
                    chunk_docs_i = valid_docs[chunk_start_i:chunk_end_i]
                    
                    weights_i = [self.weights[weighting][i] for i in chunk_docs_i]
                    magnitudes_i = [self.magnitudes[weighting][i] for i in chunk_docs_i]
                    
                    for chunk_start_j in range(0, len(valid_docs), self.chunk_size):
                        chunk_end_j = min(chunk_start_j + self.chunk_size, len(valid_docs))
                        if valid_docs[chunk_end_j - 1] < valid_docs[chunk_start_i]:
                            continue
                        chunk_docs_j = valid_docs[chunk_start_j:chunk_end_j]
                        weights_j = [self.weights[weighting][j] for j in chunk_docs_j]
                        magnitudes_j = [self.magnitudes[weighting][j] for j in chunk_docs_j]
                        
                        chunk_tasks.append((
                            chunk_id,
                            valid_docs[chunk_start_i], valid_docs[chunk_end_i - 1] + 1, 
                            weights_i, magnitudes_i,
                            valid_docs[chunk_start_j], valid_docs[chunk_end_j - 1] + 1, 
                            weights_j, magnitudes_j,
                            self.index.filenames
                        ))
                        chunk_id += 1
                
                num_processes = min(self.num_processes, len(chunk_tasks))
                if num_processes > 0:
                    with multiprocessing.Pool(processes=num_processes) as pool:
                        chunk_results = pool.map(_compute_similarities_for_chunk, chunk_tasks)
                else:
                    chunk_results = []
                
                for result_set in chunk_results:
                    for doc1, doc2, sim in result_set:
                        if len(top_similarities) < k:
                            heapq.heappush(top_similarities, (sim, doc1, doc2))
                        elif sim > top_similarities[0][0]:
                            heapq.heappushpop(top_similarities, (sim, doc1, doc2))
                
                return [(doc1, doc2, sim) for sim, doc1, doc2 in sorted(top_similarities, reverse=True)]
            except Exception as e:
                self.profiler.log_message(f"Error in chunked similarity calculation: {e}")
                return self._fallback_find_similar_documents(k, weighting)

    def _fallback_find_similar_documents(self, k=10, weighting='tf') -> List[Tuple[str, str, float]]:
        """
        Fallback sequential similarity calculation.
        """
        self.profiler.log_message("Falling back to standard similarity calculation")
        similarities = []
        total_docs = self.index.doc_count
        
        for i in range(total_docs):
            if i not in self.magnitudes[weighting] or self.magnitudes[weighting][i] == 0:
                continue
            for j in range(i+1, total_docs):
                if j not in self.magnitudes[weighting] or self.magnitudes[weighting][j] == 0:
                    continue
                sim = _cosine_similarity(
                    self.weights[weighting][i], self.magnitudes[weighting][i],
                    self.weights[weighting][j], self.magnitudes[weighting][j]
                )
                if sim > 0:
                    if len(similarities) < k:
                        heapq.heappush(similarities, (sim, i, j))
                    elif sim > similarities[0][0]:
                        heapq.heappushpop(similarities, (sim, i, j))
        
        return [
            (self.index.filenames[i], self.index.filenames[j], sim)
            for sim, i, j in sorted(similarities, reverse=True)
        ]