# src/vsm/base_vsm.py
from typing import List, Tuple
from src.performance_monitoring import Profiler
from src.index.memory_index import MemoryIndex

class BaseVSM:
    def __init__(self, index: MemoryIndex, profiler: Profiler = None):
        self.index = index
        self.profiler = profiler or Profiler()
        self.weights = {'tf': {}, 'tfidf': {}, 'sublinear': {}}
        self.magnitudes = {'tf': {}, 'tfidf': {}, 'sublinear': {}}
        
    def build_model(self): ...
    def find_similar_documents(self, k: int, weighting: str) -> List[Tuple[str, str, float]]: ...