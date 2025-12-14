#!/usr/bin/env python3
"""
Create and populate a project virtual environment at `.venv`.

Usage:
  python scripts/create_venv.py --requirements requirements.txt [--recreate]

This script is cross-platform and will:
 - create a `.venv` directory using the current Python (or `--python`)
 - upgrade pip/setuptools/wheel inside the venv
 - install from the provided `requirements.txt`
 - optionally verify imports
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time


def run(cmd, check=True, **kwargs):
    print("+ ", " ".join(cmd))
    return subprocess.run(cmd, check=check, **kwargs)


def venv_python_path(venv_dir: str = ".venv") -> str:
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def create_venv(python_exe: str, venv_dir: str = ".venv", recreate: bool = False):
    if recreate and os.path.exists(venv_dir):
        print(f"Removing existing venv: {venv_dir}")
        def _on_rm_error(func, path, exc_info):
            try:
                os.chmod(path, 0o777)
                func(path)
            except Exception as e:
                print(f"Failed to remove {path}: {e}")
                raise

        try:
            shutil.rmtree(venv_dir, onerror=_on_rm_error)
        except Exception as e:
            backup = f"{venv_dir}_backup_{int(time.time())}"
            print(f"Could not fully remove .venv due to: {e}. Renaming to {backup}")
            try:
                shutil.move(venv_dir, backup)
                print(f"Renamed existing venv to {backup}; continuing to create new venv")
            except Exception as me:
                print("Failed to rename .venv; please remove it manually and re-run this script:", me)
                sys.exit(1)

    if not os.path.exists(venv_dir):
        print(f"Creating virtual environment at {venv_dir} using {python_exe}")
        run([python_exe, "-m", "venv", venv_dir])
    else:
        print(f"Virtual environment already exists at {venv_dir}")


def install_requirements(venv_py: str, requirements: str):
    print("Upgrading pip, setuptools, wheel inside venv...")
    run([venv_py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    if requirements and os.path.exists(requirements):
        print(f"Installing from {requirements}...")
        run([venv_py, "-m", "pip", "install", "-r", requirements])
    else:
        print(f"Requirements file not found: {requirements}")


def verify_packages(venv_py: str, packages: list[str]):
    print("Verifying imports inside venv...")
    missing = []
    for pkg in packages:
        cmd = [venv_py, "-c", f"import {pkg}; print('{pkg}: OK')"]
        try:
            run(cmd)
        except subprocess.CalledProcessError:
            missing.append(pkg)

    if missing:
        print("Missing packages:", ", ".join(missing))
    else:
        print("All verification imports succeeded")


def print_activation(venv_dir: str = ".venv"):
    if os.name == "nt":
        print("\nTo activate in PowerShell:")
        print(f"  .\\{venv_dir}\\Scripts\\Activate.ps1")
        print("Or in cmd.exe:")
        print(f"  {venv_dir}\\Scripts\\activate.bat")
    else:
        print("\nTo activate in bash/zsh:")
        print(f"  source {venv_dir}/bin/activate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", help="Path to Python executable to create venv with (defaults to current interpreter)")
    ap.add_argument("--requirements", default="requirements.txt", help="Path to requirements file")
    ap.add_argument("--recreate", action="store_true", help="Remove existing .venv and recreate")
    ap.add_argument("--verify", action="store_true", help="Verify core packages imports after install")
    args = ap.parse_args()

    python_exe = args.python or sys.executable

    # If the active Python is very new (e.g., 3.14+) some binary wheels may not
    # be available (numpy/opencv). Try to locate a compatible Python (3.12/3.13/3.11)
    # via the Windows Python launcher `py` when a specific --python wasn't provided.
    if not args.python:
        ver = sys.version_info
        if ver.major == 3 and ver.minor >= 14:
            print(f"Warning: current Python is {ver.major}.{ver.minor}. Trying to locate Python 3.12/3.13/3.11 for better wheel support.")
            candidates = ["3.12", "3.13", "3.11"]
            for c in candidates:
                try:
                    res = subprocess.run(["py", f"-{c}", "-c", "import sys; print(sys.executable)"], capture_output=True, text=True)
                except Exception:
                    # `py` launcher not available or invocation failed
                    res = None
                if res and res.returncode == 0:
                    candidate_path = res.stdout.strip().splitlines()[-1]
                    if candidate_path:
                        print(f"Found Python {c} at: {candidate_path}. Will use this to create the venv.")
                        python_exe = candidate_path
                        break
            else:
                print("Could not find an alternate Python via the `py` launcher. You can re-run with --python pointing to Python 3.12 if available.")


    try:
        create_venv(python_exe, venv_dir=".venv", recreate=args.recreate)
    except subprocess.CalledProcessError as e:
        print("Failed to create virtualenv:", e)
        sys.exit(2)

    venv_py = venv_python_path(".venv")
    if not os.path.exists(venv_py):
        print("Could not find Python inside .venv; expected:", venv_py)
        sys.exit(3)

    try:
        install_requirements(venv_py, args.requirements)
    except subprocess.CalledProcessError as e:
        print("pip install failed:", e)
        sys.exit(4)

    if args.verify:
        verify_packages(venv_py, ["numpy", "plotly", "kaleido", "cv2", "tqdm"])  # cv2 is opencv-python

    print_activation(".venv")
    print("\nSetup complete.")


if __name__ == "__main__":
    main()
