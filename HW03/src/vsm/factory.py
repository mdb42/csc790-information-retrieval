# src/vsm/factory.py
"""
Vector Space Model Factory
Author: Matthew Branson
Date: March 14, 2025

This module implements a factory pattern for creating appropriate Vector Space Model
implementations based on document count, available dependencies, and user preferences.
"""
import os
from typing import Optional, Dict
from src.index.base import BaseIndex
from src.vsm.base import BaseVSM
from src.profiler import Profiler


class VSMFactory:
    """
    Factory class for creating Vector Space Model implementations.
    
    This factory dynamically selects and instantiates the most appropriate 
    Vector Space Model implementation based on available dependencies,
    document count, and user preferences.
    
    Class Attributes:
        DEFAULT_PARALLEL_DOC_THRESHOLD (int): Document count threshold for switching 
                                             from StandardVSM to ParallelVSM
        VSM_CLASSES (Dict[str, str]): Mapping of implementation names to class paths
    """
    
    # Threshold for switching from Standard to Parallel VSM
    DEFAULT_PARALLEL_DOC_THRESHOLD = 2500  # Switch from Standard to Parallel at 2.5K docs
    
    # Mapping of implementation names to their class paths
    VSM_CLASSES = {
        "sparse": "sparse_vsm.SparseVSM",
        "standard": "standard_vsm.StandardVSM",
        "parallel": "parallel_vsm.ParallelVSM"
    }
    
    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """
        Check for availability of optional dependencies.

        Returns:
            Dict[str, bool]: Dictionary mapping dependency names to availability status
        """
        dependencies = {
            "numpy": False,
            "scipy.sparse": False,
            "sklearn.metrics": False,
            "multiprocessing": False
        }
        
        # Check for NumPy
        try:
            import numpy
            dependencies["numpy"] = True
        except ImportError:
            pass
        
        # Check for SciPy sparse
        try:
            import scipy.sparse
            dependencies["scipy.sparse"] = True
        except ImportError:
            pass
        
        # Check for scikit-learn metrics
        try:
            import sklearn.metrics.pairwise
            dependencies["sklearn.metrics"] = True
        except ImportError:
            pass
        
        # Check for multiprocessing
        try:
            import multiprocessing
            # Verify that we can actually create a process pool
            # (some environments may have the module but not support actual parallelism)
            try:
                with multiprocessing.Pool(1) as p:
                    pass
                dependencies["multiprocessing"] = True
            except (ImportError, OSError, ValueError):
                # OSError can happen in restricted environments
                # ValueError can occur if system resources are unavailable
                pass
        except ImportError:
            pass
        
        return dependencies
    
    @staticmethod
    def create_vsm(index: BaseIndex, mode: str = 'auto', profiler: Optional[Profiler] = None,
                parallel_threshold: int = None, parallelize_weights: bool = False) -> 'BaseVSM':
        """
        Create and return the appropriate Vector Space Model implementation.
        
        This method selects the VSM implementation based on:
        - Explicitly requested mode (if specified)
        - Available dependencies
        - Document count and threshold settings
        
        Args:
            index (BaseIndex): The document index to use
            mode (str): VSM implementation to use ('auto', 'standard', 'parallel', or 'sparse')
            profiler (Profiler, optional): Performance profiler for timing operations
            parallel_threshold (int, optional): Document count threshold for using ParallelVSM
                                            over StandardVSM
            parallelize_weights (bool, optional): Whether to parallelize weight computation 
                                            in ParallelVSM (default: False)
            
        Returns:
            BaseVSM: An instance of the selected VSM implementation
            
        Note:
            If the requested implementation is unavailable, this method falls back to
            the best available alternative and logs a warning.
        """
        # Create a profiler if none was provided
        if profiler is None:
            from HW03.src.profiler import Profiler
            profiler = Profiler()
        
        # Use default threshold if none specified
        if parallel_threshold is None:
            parallel_threshold = VSMFactory.DEFAULT_PARALLEL_DOC_THRESHOLD
        
        # Check available dependencies
        deps = VSMFactory.check_dependencies()
        
        # Determine if we have dependencies for specific implementations
        has_sparse_deps = deps["numpy"] and deps["scipy.sparse"] and deps["sklearn.metrics"]
        has_parallel = deps["multiprocessing"]
        
        # Get document count for threshold-based decisions
        doc_count = index.doc_count
        
        # Store original mode for logging purposes
        original_mode = mode

        # Auto-select mode based on dependencies and document count
        if mode == 'auto':
            if has_sparse_deps:
                # SparseVSM is always the fastest when dependencies are available
                mode = 'sparse'
            elif doc_count < parallel_threshold:
                # For small document sets, StandardVSM is more efficient
                mode = 'standard'
            elif has_parallel:
                # For larger document sets, use parallelization if available
                mode = 'parallel'
            else:
                # Fall back to StandardVSM if nothing else is available
                mode = 'standard'

            if profiler:
                profiler.log_message(f"Auto-selected VSM mode: {mode} (based on dependencies and doc_count={doc_count})")
        
        # Handle case where sparse was requested but dependencies aren't available
        if mode == 'sparse' and not has_sparse_deps:
            if profiler:
                profiler.log_message(f"Warning: Sparse VSM requested but dependencies not available.")
            
            # Fall back to the next best option based on document count
            if doc_count < parallel_threshold:
                mode = 'standard'
                profiler.log_message(f"Using Standard VSM instead (optimal for small datasets).")
            elif has_parallel:
                mode = 'parallel'
                profiler.log_message(f"Using Parallel VSM instead (optimal for medium/large datasets).")
            else:
                mode = 'standard'
                profiler.log_message(f"Using Standard VSM instead (no parallelization available).")
            
            print(f"Warning: Sparse VSM requested but dependencies not available. Using {mode} VSM.")
        # Handle case where parallel was requested but multiprocessing isn't available
        elif mode == 'parallel' and not has_parallel:
            if profiler:
                profiler.log_message(f"Warning: Parallel VSM requested but multiprocessing not available. Using standard VSM.")
            print(f"Warning: Parallel VSM requested but multiprocessing not available. Using standard VSM.")
            mode = 'standard'
        
        # Log if an explicit mode was requested (not auto)
        if original_mode != 'auto' and profiler:
            profiler.log_message(f"Using explicitly requested VSM mode: {mode}")
        
        # Instantiate the selected VSM implementation

        try:
            # Get the class path string for the selected mode
            selected_class = VSMFactory.VSM_CLASSES.get(mode, VSMFactory.VSM_CLASSES['standard'])
            
            # Split into module name and class name
            module_name, class_name = selected_class.rsplit(".", 1)
            
            # Import the module dynamically
            module = __import__(f"src.vsm.{module_name}", fromlist=[class_name])
            
            if profiler:
                profiler.log_message(f"Created VSM implementation: {module_name}.{class_name}")
            
            # If parallel mode is selected, pass the parallelize_weights parameter
            if mode == 'parallel':
                if profiler and parallelize_weights:
                    profiler.log_message(f"Weight computation will be parallelized (document count: {index.doc_count})")
                
                # Get the ParallelVSM class
                vsm_class = getattr(module, class_name)
                
                # Create the ParallelVSM instance with the parallelize_weights parameter
                return vsm_class(index, profiler, parallelize_weights=parallelize_weights)
            else:
                # For other VSM implementations, just instantiate normally
                return getattr(module, class_name)(index, profiler)
                
        except ImportError as e:
            # Handle import errors by falling back to StandardVSM
            if profiler:
                profiler.log_message(f"Warning: Failed to import {mode} VSM implementation: {e}. Using standard VSM.")
            print(f"Warning: Failed to import {mode} VSM implementation: {e}. Using standard VSM.")
            from src.vsm.standard_vsm import StandardVSM
            return StandardVSM(index, profiler)