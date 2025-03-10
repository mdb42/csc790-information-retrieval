import os
from typing import Optional
from src.index.base import BaseIndex
from src.performance_monitoring import Profiler
"""
VSM Comparisons

| Document Count | Standard  | Hybrid   | Parallel | Experimental | Sparse  |
| -------------- | --------- | -------- | -------- | ------------ | ------- |
| 1000 Docs      | 2.8422    | 8.4243   | 11.9635  | 8.4299       | 1.0132  |
| 2000 Docs      | 9.8976    | 11.6419  | 15.1696  | 15.8947      | 1.9903  |
| 3000 Docs      | 18.5161   | 15.8393  | 19.0588  | 13.8868      | 3.0039  |
| 4000 Docs      | 31.8160   | 18.8228  | 22.3296  | 18.3230      | 4.0728  |
| 5000 Docs      | 48.5661   | 22.6129  | 25.7587  | 24.5038      | 4.9694  |
| 15000 Docs     | 449.5013  | 96.3591  | 101.2593 | 96.3827      | 20.9662 |
| 30000 Docs     | 1928.2886 | 365.8678 | 368.9252 | 396.8536     | 55.8942 |

VSM Parallelization Threshold Finetuning

| Fine-Tuning Datasets | Standard | Hybrid  |
| -------------------- | -------- | ------- |
| 2500 Docs            | 13.6522  | 14.2365 |
| 2600 Docs            | 15.6922  | 14.9760 |

"""

class VSMFactory:

    DEFAULT_HYBRID_DOC_THRESHOLD = 2500  # Switch from Standard to Hybrid at 2.5K docs
    
    VSM_CLASSES = {
        "sparse": "sparse_vsm.SparseVSM",
        "hybrid": "hybrid_vsm.HybridVSM",        
        "standard": "standard_vsm.StandardVSM",
        "parallel": "parallel_vsm.ParallelVSM",
        "experimental": "experimental_vsm.ExperimentalVSM"
    }
    
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
            except (ImportError, OSError, ValueError):
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
        
        original_mode = mode

        if mode == 'auto':
            if has_sparse_deps:
                mode = 'sparse'
            elif doc_count < hybrid_threshold:
                mode = 'standard'
            elif has_parallel:
                mode = 'hybrid'
            else:
                mode = 'standard'

            if profiler:
                profiler.log_message(f"Auto-selected VSM mode: {mode} (based on dependencies and doc_count={doc_count})")
        
        if mode == 'sparse' and not has_sparse_deps:
            if profiler:
                profiler.log_message(f"Warning: Sparse VSM requested but dependencies not available.")
            
            if doc_count < hybrid_threshold:
                mode = 'standard'
                profiler.log_message(f"Using Standard VSM instead (optimal for small datasets).")
            elif has_parallel:
                mode = 'hybrid'
                profiler.log_message(f"Using Hybrid VSM instead (optimal for medium/large datasets).")
            else:
                mode = 'standard'
                profiler.log_message(f"Using Standard VSM instead (no parallelization available).")
            
            print(f"Warning: Sparse VSM requested but dependencies not available. Using {mode} VSM.")
        elif (mode in ['parallel', 'hybrid', 'experimental']) and not has_parallel:
            if profiler:
                profiler.log_message(f"Warning: {mode} VSM requested but multiprocessing not available. Using standard VSM.")
            print(f"Warning: {mode} VSM requested but multiprocessing not available. Using standard VSM.")
            mode = 'standard'
        
        if original_mode != 'auto' and profiler:
            profiler.log_message(f"Using explicitly requested VSM mode: {mode}")
        
        try:
            selected_class = VSMFactory.VSM_CLASSES.get(mode, VSMFactory.VSM_CLASSES['standard'])
            module_name, class_name = selected_class.rsplit(".", 1)
            module = __import__(f"src.vsm.{module_name}", fromlist=[class_name])
            
            if profiler:
                profiler.log_message(f"Created VSM implementation: {module_name}.{class_name}")
            
            return getattr(module, class_name)(index, profiler)
        except ImportError as e:
            if profiler:
                profiler.log_message(f"Warning: Failed to import {mode} VSM implementation: {e}. Using standard VSM.")
            print(f"Warning: Failed to import {mode} VSM implementation: {e}. Using standard VSM.")
            from src.vsm.standard_vsm import StandardVSM
            return StandardVSM(index, profiler)