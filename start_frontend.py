#!/usr/bin/env python3
"""
Enhanced RAG Pipeline Frontend Startup Script
============================================

A comprehensive startup script that:
- Checks system requirements
- Starts the FastAPI backend
- Opens the browser automatically
- Provides real-time status updates
"""

import os
import sys
import time
import subprocess
import webbrowser
import requests
from pathlib import Path

def print_header():
    """Print startup header"""
    print("\n" + "="*60)
    print("  🚀 Enhanced RAG Pipeline Frontend Startup")
    print("="*60)

def print_status(message, status="INFO"):
    """Print formatted status message"""
    icons = {
        "INFO": "ℹ️",
        "SUCCESS": "✅", 
        "ERROR": "❌",
        "WARNING": "⚠️",
        "LOADING": "⏳"
    }
    icon = icons.get(status, "📝")
    print(f"{icon} {message}")

def check_requirements():
    """Check if all required files exist"""
    print_status("Running pre-flight checks...")
    
    required_files = [
        "backend.py",
        "index.html", 
        "static/style.css",
        "static/app.js",
        "data/sentiment_system.duckdb"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print_status("Missing required files:", "ERROR")
        for file_path in missing_files:
            print(f"   ❌ {file_path}")
        return False
    
    # Check database
    db_path = Path("data/sentiment_system.duckdb")
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print_status(f"Database found: {db_path} ({size_mb:.1f} MB)", "SUCCESS")
    
    # Check static files
    static_files = list(Path("static").glob("*")) if Path("static").exists() else []
    if static_files:
        print_status(f"Static files found: {len(static_files)} files", "SUCCESS")
    
    # Check for ML analysis directories
    analysis_dirs = [d for d in Path(".").iterdir() if d.is_dir() and "analysis" in d.name]
    if analysis_dirs:
        print_status(f"Analysis directories found: {len(analysis_dirs)} directories", "SUCCESS")
    
    print_status("Pre-flight checks completed", "SUCCESS")
    return True

def wait_for_backend(url="http://localhost:8000/api/health", timeout=30):
    """Wait for backend to be ready"""
    print_status("Waiting for backend to start...")
    
    for i in range(timeout):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print_status("Backend is ready!", "SUCCESS")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print_status(f"Waiting for backend... ({i+1}/{timeout})", "LOADING")
        time.sleep(1)
    
    print_status("Backend failed to start within timeout", "ERROR")
    return False

def start_backend(fast_mode=True):
    """Start the FastAPI backend"""
    backend_file = "backend_fast.py" if fast_mode else "backend.py"
    mode_text = "Fast Backend (Mock Agents)" if fast_mode else "Full Backend (Real Agents)"
    
    print_status(f"Starting {mode_text}...")
    
    # Check if uvicorn is available
    try:
        import uvicorn
    except ImportError:
        print_status("Installing uvicorn...", "LOADING")
        subprocess.run([sys.executable, "-m", "pip", "install", "uvicorn[standard]"], 
                      capture_output=True)
    
    # Start the backend
    try:
        # Use subprocess to run uvicorn
        backend_process = subprocess.Popen([
            sys.executable, backend_file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return backend_process
    except Exception as e:
        print_status(f"Failed to start backend: {e}", "ERROR")
        return None

def open_browser(url="http://localhost:8000"):
    """Open the frontend in browser"""
    print_status("Launching frontend...")
    try:
        webbrowser.open(url)
        print_status(f"Opening frontend: {url}", "SUCCESS")
        return True
    except Exception as e:
        print_status(f"Failed to open browser: {e}", "WARNING")
        print_status(f"Please manually open: {url}", "INFO")
        return False

def print_footer():
    """Print final status and instructions"""
    print("\n" + "="*60)
    print("🎉 Enhanced RAG Pipeline Dashboard is now running:")
    print("   🔗 Frontend: http://localhost:8000")
    print("   📖 API Docs: http://localhost:8000/docs") 
    print("   ❤️  Health: http://localhost:8000/api/health")
    print("="*60)
    print("\n💡 Usage Tips:")
    print("   1. 🎭 Try the Live Agent Prediction Theater (Mock Mode)")
    print("   2. 📊 Analyze setups with enhanced predictions")
    print("   3. 🔍 Use the portfolio scanner for opportunities")
    print("   4. 🎨 Explore visualizations and model performance")
    print("   5. 🧠 Check out the knowledge graph explorer")
    print("\n⚡ Fast Mode Info:")
    print("   - Mock agent predictions for instant startup")
    print("   - Real data from your database")
    print("   - To use real agents: python start_frontend.py --full")
    print("\n⚠️  Press Ctrl+C to stop the server")
    print("="*60)

def main():
    """Main startup routine"""
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced RAG Pipeline Frontend")
    parser.add_argument("--full", action="store_true", 
                       help="Use full backend with real agents (slower startup)")
    args = parser.parse_args()
    
    fast_mode = not args.full
    
    print_header()
    
    if fast_mode:
        print_status("Using Fast Mode - Mock agents for instant startup", "INFO")
    else:
        print_status("Using Full Mode - Real agents (slower startup)", "WARNING")
    
    # Check requirements
    if not check_requirements():
        print_status("Pre-flight checks failed. Please fix the issues above.", "ERROR")
        sys.exit(1)
    
    # Start backend
    backend_process = start_backend(fast_mode=fast_mode)
    if not backend_process:
        print_status("Failed to start backend server", "ERROR")
        sys.exit(1)
    
    # Wait for backend to be ready
    if not wait_for_backend():
        print_status("Backend is not responding", "ERROR")
        backend_process.terminate()
        sys.exit(1)
    
    # Open browser
    open_browser()
    
    # Print final status
    print_footer()
    
    # Keep running until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print_status("\nShutting down Enhanced RAG Pipeline Frontend...", "INFO")
        backend_process.terminate()
        backend_process.wait()
        print_status("Shutdown complete", "SUCCESS")

if __name__ == "__main__":
    main()