#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "generate_data.py"), run_name="__main__")

