import os
import platform
import subprocess
import sys
import urllib.request

def install(package):
    """Install a Python package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def download_and_install_python():
    """Download and install Python 3.8.8 based on the operating system."""
    system = platform.system()
    python_url = ""
    installer_path = ""

    if system == "Windows":
        python_url = "https://www.python.org/ftp/python/3.8.8/python-3.8.8-amd64.exe"
        installer_path = "python-3.8.8-amd64.exe"
    elif system == "Linux":
        python_url = "https://www.python.org/ftp/python/3.8.8/Python-3.8.8.tgz"
        installer_path = "Python-3.8.8.tgz"
    elif system == "Darwin":  # macOS
        python_url = "https://www.python.org/ftp/python/3.8.8/python-3.8.8-macosx10.9.pkg"
        installer_path = "python-3.8.8-macosx10.9.pkg"
    else:
        print("Unsupported operating system. Please install Python 3.8.8 manually.")
        return False

    print(f"Downloading Python 3.8.8 installer from {python_url}...")
    urllib.request.urlretrieve(python_url, installer_path)
    print(f"Downloaded Python installer: {installer_path}")

    if system == "Windows":
        print("Running the Python installer...")
        subprocess.run([installer_path, "/quiet", "InstallAllUsers=1", "PrependPath=1"])
    elif system == "Linux":
        print("Extracting the Python source...")
        subprocess.run(["tar", "xvzf", installer_path])
        os.chdir("Python-3.8.8")
        subprocess.run(["./configure", "--enable-optimizations"])
        subprocess.run(["make"])
        subprocess.run(["sudo", "make", "altinstall"])
    elif system == "Darwin":
        print("Running the Python installer...")
        subprocess.run(["sudo", "installer", "-pkg", installer_path, "-target", "/"])
    
    print("Python 3.8.8 installation complete.")
    return True

def main():
    # Step 1: Check Python version
    if sys.version_info[:2] != (3, 8):
        print("Python 3.8.8 is not installed. Installing it now...")
        success = download_and_install_python()
        if not success:
            print("Failed to install Python. Please install it manually.")
            return

    # Step 2: Install necessary libraries
    libraries = [
        "pandas==1.2.4",       # Version compatible with Python 3.8.8
        "numpy==1.19.5",       # Version compatible with Python 3.8.8
        "joblib",              # Used for saving/loading models
        "psutil",              # Used for system resource monitoring
        "requests",            # Used for HTTP requests
    ]

    print("Installing required Python libraries...")
    for library in libraries:
        try:
            print(f"Installing {library}...")
            install(library)
            print(f"{library} installed successfully.")
        except Exception as e:
            print(f"Failed to install {library}: {e}")

if __name__ == "__main__":
    main()
