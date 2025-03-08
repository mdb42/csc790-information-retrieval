# src/vsm/factory.py
import os
from typing import Optional
from src.index.memory_index import MemoryIndex
from src.performance_monitoring import Profiler
from src.vsm.base_vsm import BaseVSM

class VSMFactory:
    @staticmethod
    def create_vsm(index: MemoryIndex, mode: str = 'auto', profiler: Optional[Profiler] = None) -> BaseVSM:
        if profiler is None:
            from src.performance_monitoring import Profiler
            profiler = Profiler()

        has_scipy = False
        try:
            import scipy.sparse
            import sklearn.metrics.pairwise
            has_scipy = True
        except ImportError:
            pass
        
        if mode == 'auto':
            if has_scipy and index.doc_count > 100:
                mode = 'sparse'
            elif index.doc_count > 1000:
                mode = 'parallel'
            else:
                mode = 'standard'
        
        if mode == 'sparse' and has_scipy:
            from src.vsm.sparse_vsm import SparseVSM
            vsm = SparseVSM(index, profiler)
        # elif mode == 'parallel':
        #     from src.vsm.standard_vsm_parallel import StandardVSMParallel
        #     vsm = StandardVSMParallel(index, profiler)
        elif mode == 'standard':
            from src.vsm.standard_vsm import StandardVSM
            vsm = StandardVSM(index, profiler)
        else:
            # Fallback to standard
            from src.vsm.standard_vsm import StandardVSM
            vsm = StandardVSM(index, profiler)
            profiler.log(f"Requested mode '{mode}' not available, using standard implementation.")
        
        # Build the model
        vsm.build_model()
        return vsm