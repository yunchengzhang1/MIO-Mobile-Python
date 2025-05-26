import time

class Timer:
    def __init__(self, label="Operation"):
        self.label = label
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        elapsed = self.end_time - self.start_time
        print(f"[TIMER] {self.label} took {elapsed:.4f} seconds.")

    def duration(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None