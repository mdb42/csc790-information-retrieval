# src/performance_monitoring.py
import time
import io

class Profiler:
    def __init__(self):
        self.timings = {}
        self.log_buffer = io.StringIO()
        self.start_time = None

    def timer(self, task_name):
        """Returns a context manager to time a code block."""
        return Timer(task_name, self)

    def log(self, message):
        """Logs a message to the buffer."""
        self.log_buffer.write(f"{message}\n")

    def start_global_timer(self):
        """Starts the global execution timer."""
        self.start_time = time.time()

    def get_global_time(self):
        """Returns the total execution time since start_global_timer was called."""
        return time.time() - self.start_time if self.start_time else 0.0

    def write_log_file(self, filename, doc_count, vocab_size, total_time):
        """Writes all collected data to a log file."""
        with open(filename, "w") as f:
            f.write("===== Performance Log =====\n")
            f.write(f"Total Documents Indexed: {doc_count}\n")
            f.write(f"Vocabulary Size: {vocab_size}\n")
            
            f.write("\n=== Timing Data ===\n")
            for task, duration in self.timings.items():
                f.write(f"{task}: {duration:.4f} seconds\n")
            f.write(f"Total Execution: {total_time:.4f} seconds\n")
            
            f.write("\n=== Log Messages ===\n")
            f.write(self.log_buffer.getvalue())

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
