#!/usr/bin/env python3
"""
CropGuard AI - Desktop App Launcher
Starts FastAPI server and opens native window via pywebview.
"""

import sys
import os
import subprocess
import threading
import time
import signal
import atexit
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def install_requirements():
    """Install missing packages from requirements.txt."""
    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        return
    
    print("Checking dependencies...")
    try:
        import pkg_resources
        required = set(line.strip() for line in req_file.read_text().splitlines() 
                       if line.strip() and not line.startswith('#'))
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = required - installed
        
        if missing:
            print(f"Installing missing packages: {missing}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
            print("Dependencies installed.")
    except Exception as e:
        print(f"Warning: Could not verify/install dependencies: {e}")

def check_models():
    """Verify all required model files exist."""
    from inference import ModelManager
    mm = ModelManager(models_dir=PROJECT_ROOT / "models")
    health = mm.health_check()
    missing = [k for k, v in health.items() if not v]
    if missing:
        print("ERROR: Missing model files:")
        for m in missing:
            print(f"  - {mm.REQUIRED_MODELS[m]}")
        print("\nPlease train models first using:")
        print("  python scripts/train_efficientnet.py --model rice --epochs 20")
        print("  python scripts/train_efficientnet.py --model wheat --epochs 20")
        print("  python scripts/train_efficientnet.py --model crop --epochs 20")
        print("  python scripts/train_efficientnet.py --model wheat_stage --epochs 20")
        return False
    return True

def start_api_server():
    """Start the FastAPI server in a background thread."""
    import uvicorn
    import requests
    
    # Change to project root for proper imports
    os.chdir(PROJECT_ROOT)
    
    config = uvicorn.Config(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        log_level="warning",
        reload=False
    )
    server = uvicorn.Server(config)
    
    def run_server():
        server.run()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    # Wait for server to be ready
    for _ in range(30):
        try:
            resp = requests.get("http://127.0.0.1:8000/health", timeout=1)
            if resp.status_code == 200:
                print("API server ready at http://127.0.0.1:8000")
                return True
        except:
            pass
        time.sleep(0.5)
    
    print("WARNING: API server health check timeout")
    return False

def open_desktop_window():
    """Open the app in a native pywebview window."""
    try:
        import webview
        
        # Create window
        window = webview.create_window(
            "CropGuard AI",
            "http://127.0.0.1:8000",
            width=480,
            height=850,
            resizable=False,
            frameless=False,
            easy_drag=True,
            confirm_close=True
        )
        
        # Start the webview event loop (blocks)
        webview.start(debug=False)
        return True
        
    except ImportError:
        print("pywebview not installed, falling back to browser")
        return False
    except Exception as e:
        print(f"Failed to create native window: {e}")
        print("Falling back to browser...")
        return False

def open_in_browser():
    """Fallback: open in default browser."""
    import webbrowser
    webbrowser.open("http://127.0.0.1:8000")
    print("Opened in default browser. Press Ctrl+C to stop server.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")

def main():
    print("=" * 50)
    print("  CropGuard AI - Desktop App")
    print("=" * 50)
    
    # Install dependencies
    install_requirements()
    
    # Check models
    if not check_models():
        sys.exit(1)
    
    # Start API server
    print("Starting API server...")
    if not start_api_server():
        print("Failed to start API server")
        sys.exit(1)
    
    # Try native window, fallback to browser
    if not open_desktop_window():
        open_in_browser()

if __name__ == "__main__":
    main()