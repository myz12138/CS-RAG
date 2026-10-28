#!/usr/bin/env python3
"""Run Phase 1 structural retrieval."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.phase1.pipeline import main_1


if __name__ == "__main__":
    main_1()
