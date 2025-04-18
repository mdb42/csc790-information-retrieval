# src/profiler.py
"""
Performance Profiling
Author: Matthew Branson
Date: March 14, 2025

This module provides tools for measuring and reporting execution time
and tracking diagnostic messages throughout the application. It offers
context managers for timing specific code blocks and methods for generating
comprehensive performance reports.
"""
import time
from io import StringIO


class Profiler:
    """
    Performance profiling utility for timing operations and logging messages.
    
    Attributes:
        timings (dict): Dictionary mapping task names to execution times
        start_time (float): Start timestamp of the global timer
        paused_time (float): Total time the global timer has been paused
        log_buffer (StringIO): Buffer for storing diagnostic messages
    """
    def __init__(self):
        """Initialize a new Profiler instance with empty timing and logging state."""
        self.timings = {}
        self.start_time = None
        self.paused_time = 0.0
        self.log_buffer = StringIO()

    def timer(self, task_name):
        """
        Create a context manager for timing a code block.
        
        Args:
            task_name (str): Name of the task to be timed
            
        Returns:
            Timer: A context manager that will time the enclosed code block
            
        Example:
            with profiler.timer("Data Processing"):
                process_data()
        """
        return Timer(task_name, self)

    def log_message(self, message):
        """
        Log a diagnostic message to the buffer.
        
        Args:
            message (str): Message to be logged
        """
        self.log_buffer.write(f"{message}\n")

    def start_global_timer(self):
        """
        Start the global execution timer.
        
        This method records the current time as the start point for
        measuring total execution time.
        """
        self.start_time = time.time()

    def pause_global_timer(self):
        """
        Pause the global execution timer.
        
        This method temporarily stops the global timer, preserving
        the elapsed time so far. Can be resumed with resume_global_timer().
        """
        if self.start_time is not None:
            self.paused_time += time.time() - self.start_time
            self.start_time = None

    def resume_global_timer(self):
        """
        Resume the global execution timer after pausing.
        
        This method restarts the global timer from where it was paused,
        maintaining the cumulative execution time.
        """
        if self.start_time is None:
            self.start_time = time.time() - self.paused_time
            self.paused_time = 0.0

    def get_global_time(self):
        """
        Get the total elapsed time from the global timer.
        
        Returns:
            float: Total elapsed time in seconds
        """
        if self.start_time is None:
            return self.paused_time
        return time.time() - self.start_time + self.paused_time
    
    def generate_report(self, doc_count: int, vocab_size: int, filename: str = None) -> str:
        """
        Generate a formatted performance report.
        
        This method creates a comprehensive report including logged messages,
        individual task timings, and overall execution time.
        
        Args:
            doc_count (int): Number of documents processed
            vocab_size (int): Size of the vocabulary
            filename (str, optional): If provided, write the report to this file
            
        Returns:
            str: Formatted performance report as a string
        """
        report = StringIO()

        report.write("=== Message Log ===\n")
        report.write(self.log_buffer.getvalue())
   
        report.write("=== Timing Breakdown ===\n")
        for task, duration in self.timings.items():
            report.write(f"{task}: {duration:.4f}s\n")
        
        tracked_total = sum(self.timings.values())
        report.write(f"\nTracked Operations Total: {tracked_total:.4f}s\n")
        
        report_content = report.getvalue()
        
        if filename:
            with open(filename, "w") as f:
                f.write(report_content)
        
        return report_content
    
    # Alias for backward compatibility
    log = log_message


class Timer:
    """
    Context manager for timing the execution of a code block.
    
    This class is used internally by the Profiler.timer() method to
    provide a convenient way to time specific code blocks.
    
    Attributes:
        task_name (str): Name of the task being timed
        profiler (Profiler): Reference to the parent Profiler instance
        start (float): Start timestamp when entering the context
    """
    def __init__(self, task_name, profiler):
        """
        Initialize a new Timer context manager.
        
        Args:
            task_name (str): Name of the task to be timed
            profiler (Profiler): Reference to the parent Profiler instance
        """
        self.task_name = task_name
        self.profiler = profiler

    def __enter__(self):
        """
        Enter the context and start timing.
        
        Returns:
            Timer: Self reference for context manager
        """
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """
        Exit the context, stop timing, and record the elapsed time.
        
        Args:
            *args: Exception information (if any) passed by the context manager
        """
        elapsed = time.time() - self.start
        self.profiler.timings[self.task_name] = elapsed