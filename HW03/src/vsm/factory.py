# src/vsm/factory.py
import os
from typing import Optional
from src.index.base import BaseIndex
from src.performance_monitoring import Profiler

class VSMFactory:

    DEFAULT_HYBRID_DOC_THRESHOLD = 15000
    
    @staticmethod
    def check_dependencies():
        dependencies = {
            "numpy": False,
            "scipy.sparse": False,
            "sklearn.metrics": False,
            "multiprocessing": False
        }
        
        try:
            import numpy
            dependencies["numpy"] = True
        except ImportError:
            pass
        
        try:
            import scipy.sparse
            dependencies["scipy.sparse"] = True
        except ImportError:
            pass
        
        try:
            import sklearn.metrics.pairwise
            dependencies["sklearn.metrics"] = True
        except ImportError:
            pass
        
        try:
            import multiprocessing
            try:
                with multiprocessing.Pool(1) as p:
                    pass
                dependencies["multiprocessing"] = True
            except:
                pass
        except ImportError:
            pass
        
        return dependencies
    
    @staticmethod
    def create_vsm(index: BaseIndex, mode: str = 'auto', profiler: Optional[Profiler] = None,
                  hybrid_threshold: int = None):
        if profiler is None:
            from src.performance_monitoring import Profiler
            profiler = Profiler()
        
        if hybrid_threshold is None:
            hybrid_threshold = VSMFactory.DEFAULT_HYBRID_DOC_THRESHOLD
        
        deps = VSMFactory.check_dependencies()
        
        has_sparse_deps = deps["numpy"] and deps["scipy.sparse"] and deps["sklearn.metrics"]
        
        has_parallel = deps["multiprocessing"]
        
        doc_count = index.doc_count
        
        if mode == 'auto':
            if has_sparse_deps:
                mode = 'sparse'
            elif has_parallel:
                if doc_count <= hybrid_threshold:
                    mode = 'hybrid'
                else:
                    mode = 'parallel'
            else:
                mode = 'standard'
        
        if mode == 'sparse' and has_sparse_deps:
            from src.vsm.sparse_vsm import SparseVSM
            return SparseVSM(index, profiler)
        elif mode == 'parallel' and has_parallel:
            from src.vsm.parallel_vsm import ParallelVSM
            return ParallelVSM(index, profiler)
        elif mode == 'hybrid' and has_parallel:
            from src.vsm.hybrid_vsm import HybridVSM
            return HybridVSM(index, profiler)
        else:
            if mode != 'standard' and mode != 'auto':
                print(f"Warning: {mode} VSM requested but dependencies not available. Using standard VSM.")
            from src.vsm.standard_vsm import StandardVSM
            return StandardVSM(index, profiler)