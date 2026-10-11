"""Launcher so the lab runs from a checkout without `pip install -e .`.

    python lab.py --help
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from attention_lab.cli.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
