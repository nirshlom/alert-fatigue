"""Script to diagnose and attempt to fix R DLL loading issues on Windows"""
import os
import sys
import subprocess

print("=" * 60)
print("R DLL Loading Issue Diagnostic and Fix Script")
print("=" * 60)
print()

# Check 1: Verify R installation
print("[1] Checking R installation...")
try:
    import rpy2.robjects as ro
    r_home = ro.r('R.home()')[0]
    print(f"    R Home: {r_home}")
    r_version = ro.r('R.version.string')[0]
    print(f"    R Version: {r_version}")
except Exception as e:
    print(f"    ERROR: Cannot connect to R: {e}")
    sys.exit(1)

# Check 2: Check if DLL exists
print("\n[2] Checking for stats.dll...")
dll_path = os.path.join(r_home, "library", "stats", "libs", "x64", "stats.dll")
if os.path.exists(dll_path):
    print(f"    Found: {dll_path}")
    print(f"    File size: {os.path.getsize(dll_path) / 1024:.2f} KB")
else:
    print(f"    ERROR: DLL not found at {dll_path}")
    sys.exit(1)

# Check 3: Try to load stats package directly
print("\n[3] Testing stats package loading...")
try:
    ro.r('library(stats)')
    print("    SUCCESS: stats package loaded")
except Exception as e:
    print(f"    ERROR: Cannot load stats package: {e}")
    print("\n    This is likely due to missing Visual C++ Redistributables.")
    print("    Solution: Install Microsoft Visual C++ Redistributables")
    print("    Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")

# Check 4: Try to reinstall Matrix package
print("\n[4] Attempting to reinstall Matrix package...")
try:
    print("    Uninstalling Matrix (if exists)...")
    ro.r('if ("Matrix" %in% rownames(installed.packages())) { remove.packages("Matrix") }')
    print("    Installing Matrix...")
    ro.r('install.packages("Matrix", repos="https://cran.rstudio.com/", dependencies=TRUE)')
    print("    SUCCESS: Matrix package reinstalled")
except Exception as e:
    print(f"    WARNING: Could not reinstall Matrix: {e}")

# Check 5: Test lme4 loading
print("\n[5] Testing lme4 package loading...")
try:
    ro.r('library(lme4)')
    print("    SUCCESS: lme4 package loaded successfully!")
    version = ro.r('packageVersion("lme4")')
    print(f"    lme4 version: {version[0]}")
except Exception as e:
    print(f"    ERROR: Cannot load lme4: {e}")
    print("\n" + "=" * 60)
    print("RECOMMENDED FIXES:")
    print("=" * 60)
    print("1. Install Microsoft Visual C++ Redistributables:")
    print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print()
    print("2. Reinstall R packages in R console:")
    print("   R")
    print("   > install.packages(c('Matrix', 'lme4'), dependencies=TRUE)")
    print()
    print("3. If still failing, reinstall R completely:")
    print("   - Uninstall R from Control Panel")
    print("   - Download and install latest R from: https://cran.r-project.org/")
    print("=" * 60)
    sys.exit(1)

print("\n" + "=" * 60)
print("SUCCESS: All checks passed! R and lme4 are working correctly.")
print("=" * 60)
