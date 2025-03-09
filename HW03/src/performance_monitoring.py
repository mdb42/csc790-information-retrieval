# src/performance_monitoring.py
import time
from io import StringIO

class Profiler:
    def __init__(self):
        self.timings = {}
        self.start_time = None
        self.paused_time = 0.0

    def timer(self, task_name):
        """Returns a context manager to time a code block."""
        return Timer(task_name, self)

    def log(self, message):
        """Logs a message to the buffer."""
        self.log_buffer.write(f"{message}\n")

    def start_global_timer(self):
        """Starts the global execution timer."""
        self.start_time = time.time()

    def pause_global_timer(self):
        if self.start_time is not None:
            self.paused_time += time.time() - self.start_time
            self.start_time = None

    def resume_global_timer(self):
        if self.start_time is None:
            self.start_time = time.time() - self.paused_time
            self.paused_time = 0.0

    def get_global_time(self):
        if self.start_time is None:
            return self.paused_time
        return time.time() - self.start_time + self.paused_time
    
    def generate_report(self, doc_count: int, vocab_size: int, filename: str = None) -> str:
        """Returns formatted performance report as string and optionally writes to a file"""
        report = StringIO()
        
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

class Timer:
    def __init__(self, task_name, profiler):
        self.task_name = task_name
        self.profiler = profiler

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        self.profiler.timings[self.task_name] = elapsed
