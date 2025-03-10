# src/vsm/experimental_vsm.py
"""
Scenario:
- you're stranded on a desert island with nothing but a vanilla Python install
- you just happen to have 30k+ documents to analyze

This probably won't stay in the final build, but rather it served a good purpose.
My past implementations of the standard, parallel, and hybrid VSMs would exceed memory limits
when running on datasets of 30k+ documents. This experimental VSM was an attempt to address
that with a chunked similarity calculation approach that could handle large datasets
without running out of memory. There were also unexpected benefits in terms of performance,
so I adopted the chunked approach everywhere else I could.
"""
import math
import heapq
import multiprocessing
from typing import List, Tuple, Dict, Any
from src.performance_monitoring import Profiler
from src.index import BaseIndex
from src.vsm import BaseVSM

def _compute_similarities_for_chunk(args):
    chunk_id, start_i, end_i, weights_i, magnitudes_i, start_j, end_j, weights_j, magnitudes_j, filenames = args
    results = []
    
    for i_idx, i in enumerate(range(start_i, end_i)):
        if i_idx >= len(magnitudes_i) or magnitudes_i[i_idx] == 0:
            continue
            
        vec_i = weights_i[i_idx]
        mag_i = magnitudes_i[i_idx]
        
        j_start = max(start_j, i + 1)
        
        j_idx_offset = j_start - start_j
        
        for j_idx, j in enumerate(range(j_start, end_j)):
            real_j_idx = j_idx + j_idx_offset
            
            if real_j_idx >= len(magnitudes_j) or magnitudes_j[real_j_idx] == 0:
                continue
                
            vec_j = weights_j[real_j_idx]
            mag_j = magnitudes_j[real_j_idx]
            
            common_terms = vec_i.keys() & vec_j.keys()
            
            if not common_terms:
                continue

            dot = sum(vec_i[t] * vec_j[t] for t in common_terms)
            sim = dot / (mag_i * mag_j) if mag_i * mag_j != 0 else 0.0
            
            if sim > 0:
                results.append((
                    filenames[i], 
                    filenames[j], 
                    sim
                ))
    
    return results

class ExperimentalVSM(BaseVSM):    
    def __init__(self, index: BaseIndex, profiler: Profiler = None, 
                 num_processes: int = None, chunk_size: int = None):
        super().__init__(index, profiler)
        self.num_processes = num_processes or max(1, multiprocessing.cpu_count() - 1)
        self.chunk_size = chunk_size
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
        with self.profiler.timer("Weight Precomputation (Serial)"):
            try:
                for doc_id, term_counts in self.index.doc_term_freqs.items():
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
                        
                        if freq > 0:
                            sublinear_val = (1 + math.log10(freq)) * idf
                            sublinear[term] = sublinear_val
                            sublinear_mag_sq += sublinear_val * sublinear_val
                        else:
                            sublinear[term] = 0
                    
                    self.weights['tf'][doc_id] = tf
                    self.weights['tfidf'][doc_id] = tfidf
                    self.weights['sublinear'][doc_id] = sublinear
                    
                    self.magnitudes['tf'][doc_id] = math.sqrt(tf_mag_sq)
                    self.magnitudes['tfidf'][doc_id] = math.sqrt(tfidf_mag_sq)
                    self.magnitudes['sublinear'][doc_id] = math.sqrt(sublinear_mag_sq)
            except Exception as e:
                self.profiler.log_message(f"Error in weight computation: {e}")

    def find_similar_documents(self, k=10, weighting='tf') -> List[Tuple[str, str, float]]:
        total_docs = self.index.doc_count
        
        if weighting not in self.weights:
            raise ValueError(f"Unknown weighting scheme: {weighting}")
        
        with self.profiler.timer(f"Chunked Similarity Calculation ({weighting})"):
            try:
                valid_docs = [i for i in range(total_docs) 
                             if i in self.magnitudes[weighting] and self.magnitudes[weighting][i] > 0]
                
                if len(valid_docs) <= 1:
                    return []
                
                if self.chunk_size is None:
                    if total_docs < 3000:
                        self.chunk_size = total_docs
                    elif total_docs < 15000:
                        self.chunk_size = min(5000, max(1000, total_docs // (self.num_processes * 2)))
                    else:
                        self.chunk_size = min(2000, max(500, total_docs // (self.num_processes * 3)))
                    
                    self.profiler.log_message(f"Using chunk size: {self.chunk_size} for {total_docs} documents")
                
                top_similarities = []
                
                chunk_tasks = []
                chunk_id = 0
                
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
        self.profiler.log_message("Falling back to sequential chunked similarity calculation")
        similarities = []
        total_docs = self.index.doc_count
        
        chunk_size = 100
        
        for i in range(total_docs):
            if i not in self.magnitudes[weighting] or self.magnitudes[weighting][i] == 0:
                continue
                
            vec_i = self.weights[weighting][i]
            mag_i = self.magnitudes[weighting][i]
            
            for j in range(i+1, total_docs):
                if j not in self.magnitudes[weighting] or self.magnitudes[weighting][j] == 0:
                    continue
                    
                vec_j = self.weights[weighting][j]
                mag_j = self.magnitudes[weighting][j]
                
                common = vec_i.keys() & vec_j.keys()
                if not common:
                    continue
                    
                dot = sum(vec_i[t] * vec_j[t] for t in common)
                sim = dot / (mag_i * mag_j) if mag_i * mag_j != 0 else 0.0
                
                if sim > 0:
                    if len(similarities) < k:
                        heapq.heappush(similarities, (sim, i, j))
                    elif sim > similarities[0][0]:
                        heapq.heappushpop(similarities, (sim, i, j))
        
        return [
            (self.index.filenames[i], self.index.filenames[j], sim)
            for sim, i, j in sorted(similarities, reverse=True)
        ]