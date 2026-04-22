"""Dependency bootstrap helpers.

This module allows beginner users to run the project even if some third-party
libraries are not installed yet. It checks required modules and installs missing
ones using pip before continuing.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from typing import Mapping


def ensure_dependencies(requirements: Mapping[str, str]) -> None:
    """Install missing dependencies.

    Args:
        requirements: Mapping of module import name -> pip install spec.
                      Example: {"numpy": "numpy>=1.24"}
    """
    missing_specs = []
    for module_name, pip_spec in requirements.items():
        if importlib.util.find_spec(module_name) is None:
            missing_specs.append(pip_spec)

    if not missing_specs:
        return

    print("[Bootstrap] Missing dependencies detected:", ", ".join(missing_specs))
    print("[Bootstrap] Installing missing packages with pip...")
    cmd = [sys.executable, "-m", "pip", "install", *missing_specs]
    subprocess.check_call(cmd)
    print("[Bootstrap] Dependency installation completed.")
