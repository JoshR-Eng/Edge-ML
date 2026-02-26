"""
Board power logging via tegrastats.

tegrastats is NVIDIA's hardware monitor for Jetson.  It reads actual sensor
values (not software estimates) at a fixed interval and writes them to a log
file.  We start it just before the timed inference loop and stop it immediately
after, so the log covers only the measurement window.

The raw log file is saved as-is for post-processing in Jupyter.  The relevant
metric is VDD_IN (total board input power in mW).  Example log line:

    02-25-2026 15:38:17 RAM 2000/7772MB ... VDD_IN 2772mW/2772mW ...
"""

import subprocess
from pathlib import Path


class PowerProfiler:
    """Wraps tegrastats to log board power for one measurement window."""

    def __init__(self):
        self._proc = None

    def start(self, logfile: Path) -> None:
        """Spawn tegrastats writing to logfile at 50 ms intervals."""
        logfile.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", "50", "--logfile", str(logfile)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        """Terminate tegrastats and wait for it to flush its final log lines."""
        if self._proc:
            self._proc.terminate()
            self._proc.wait()
            self._proc = None
