#!/usr/bin/env python3
"""
Shortcut to launch the autonomous BioML research loop.

Usage:
    python run_autonomous.py
    python run_autonomous.py --max-experiments 10
    python run_autonomous.py --config config/base.yaml -n 5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Delegate to train.py --auto
sys.argv = [sys.argv[0], "--auto"] + sys.argv[1:]

from train import main

main()
