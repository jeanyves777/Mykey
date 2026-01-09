"""
Build Standalone Trading Dashboard Executable
This script builds the complete standalone application
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def main():
    """Build the standalone executable"""

    # Paths
    project_root = Path(r'C:\Users\Jean-Yves\thevolumeainative')
    bridge_dir = project_root / 'trading_system' / 'NinjaTrader_Bridge'
    venv_python = project_root / '.venv' / 'Scripts' / 'python.exe'

    print("=" * 60)
    print("Building OANDA -> NinjaTrader Trading Dashboard")
    print("=" * 60)

    # Check if PyInstaller is installed
    print("\n[1/5] Checking PyInstaller installation...")
    try:
        result = subprocess.run(
            [str(venv_python), '-m', 'PyInstaller', '--version'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("PyInstaller not found. Installing...")
            subprocess.run([str(venv_python), '-m', 'pip', 'install', 'pyinstaller'], check=True)
        else:
            print(f"PyInstaller version: {result.stdout.strip()}")
    except Exception as e:
        print(f"Error: {e}")
        print("Installing PyInstaller...")
        subprocess.run([str(venv_python), '-m', 'pip', 'install', 'pyinstaller'], check=True)

    # Check if NinjaTraderBridge.exe exists
    print("\n[2/5] Checking NinjaTrader Bridge...")
    bridge_exe = bridge_dir / 'NinjaTraderBridge.exe'
    if not bridge_exe.exists():
        print("WARNING: NinjaTraderBridge.exe not found!")
        print("You may need to compile it first. See COMPILE_BRIDGE.md")
    else:
        print(f"Found: {bridge_exe}")

    # Clean previous builds
    print("\n[3/5] Cleaning previous builds...")
    dist_dir = bridge_dir / 'dist'
    build_dir = bridge_dir / 'build'

    if dist_dir.exists():
        shutil.rmtree(dist_dir)
        print(f"Removed: {dist_dir}")

    if build_dir.exists():
        shutil.rmtree(build_dir)
        print(f"Removed: {build_dir}")

    # Run PyInstaller
    print("\n[4/5] Building executable with PyInstaller...")
    spec_file = bridge_dir / 'trading_dashboard.spec'

    os.chdir(bridge_dir)

    result = subprocess.run(
        [str(venv_python), '-m', 'PyInstaller', str(spec_file), '--clean'],
        capture_output=False
    )

    if result.returncode != 0:
        print("\nBuild FAILED!")
        return 1

    # Verify output
    print("\n[5/5] Verifying build...")
    output_exe = dist_dir / 'TradingDashboard.exe'

    if output_exe.exists():
        size_mb = output_exe.stat().st_size / (1024 * 1024)
        print(f"\n{'=' * 60}")
        print("BUILD SUCCESSFUL!")
        print(f"{'=' * 60}")
        print(f"\nOutput: {output_exe}")
        print(f"Size: {size_mb:.1f} MB")
        print(f"\nTo run: double-click TradingDashboard.exe")
        print(f"\nIMPORTANT: On first run, go to Settings tab")
        print("and enter your OANDA API credentials!")
    else:
        print("\nBuild completed but executable not found!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
