"""
Power profiling log using NVIDIA tegrastats binary.
Operates via subprocess and parses the output to 
extract power consumption data.
"""

# =================================================
#                   Imports
# =================================================
import subprocess
import time
from pathlib import Path


# =================================================
#                  PowerLog Class
# =================================================

class TegrastatsLogger:
    def __init__(self, log_path: Path, interval_ms: int=50):
        """
        Args:
            log_path (Path): Desired path of log file
            interval_ms (int): Sampling interval in milliseconds
        """
        self.log_file = log_path
        self.interval_ms = interval_ms
        self.process = None


    def __enter__(self):
        """
        Start the tegrastats subprocess in background
        """

        # Ensure Output Dir exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)


        # Start tegrastats subprocess
        cmd = [
                "tegrastats",
                "--interval", str(self.interval_ms),
                "--logfile", str(self.log_file)
                ]
        self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return self

    
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Kill the tegrastats process once inference finsishes
        """
        if self.process:
            self.process.terminate()
            self.process.wait()
