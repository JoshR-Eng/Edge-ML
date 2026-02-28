"""
DESCRIPTION:
    A benchmarking script to evalute power-accuracy-latency metrics for
    all .engine files within the suggested directory

"""

#===============================================================
#                        IMPORTS
#===============================================================

# Standard library
from pathlib import Path

# Custom modules
from src.benchmark.discover_files import find_engines




#===============================================================
#                         CONFIG
#===============================================================
TARGET_DIR = Path("models/v4")


#===============================================================
#                          MAIN
#===============================================================

engines = find_engines(TARGET_DIR)
print(engines)
