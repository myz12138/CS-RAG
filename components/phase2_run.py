#!/usr/bin/env python3
"""Phase-2 thin entrypoint.

Behavior is unchanged: delegates to the legacy implementation through the
modular pipeline package.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.phase2.pipeline import main_2


if __name__ == "__main__":
    main_2()
