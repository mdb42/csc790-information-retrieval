
import os
import time

class Logger:
    def __init__(self, level="INFO"):
        self.level = level
        self.log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.log")
        
    def info(self, message):
        if self.level in ["INFO", "DEBUG"]:
            with open(self.log_file, "a") as f:
                f.write(f"[INFO] {time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
            
    def error(self, message):
        with open(self.log_file, "a") as f:
            f.write(f"[ERROR] {time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

logging = Logger()

class Timer:
    def __init__(self, task_name):
        self.task_name = task_name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        logging.info(f"{self.task_name} took {elapsed:.4f} seconds")