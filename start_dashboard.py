#!/usr/bin/env python3
"""
Single Command Dashboard Startup
================================

One command to start the entire Enhanced RAG Pipeline Dashboard.
Automatically handles backend startup, browser opening, and cleanup.
"""

import sys
import time
import webbrowser
import subprocess
import argparse
from pathlib import Path

def print_header():
    """Print startup header"""
    print("\n" + "="*60)
    print("🚀 Enhanced RAG Pipeline Dashboard")
    print("="*60)

def print_status(message, status="INFO"):
    """Print colored status message"""
    colors = {
        "INFO": "🔵",
        "SUCCESS": "✅", 
        "WARNING": "⚠️",
        "ERROR": "❌",
        "LOADING": "⏳"
    }
    print(f"{colors.get(status, '📋')} {message}")

def check_requirements():
    """Check if required files exist"""
    required_files = [
        "data/sentiment_system.duckdb",
        "index.html",
        "static/app.js", 
        "static/style.css"
    ]
    
    print_status("Running pre-flight checks...")
    
    for file_path in required_files:
        if Path(file_path).exists():
            print_status(f"✓ {file_path} found")
        else:
            print_status(f"✗ {file_path} missing", "ERROR")
            return False
    
    print_status("Pre-flight checks completed", "SUCCESS")
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced RAG Pipeline Dashboard")
    parser.add_argument("--full", action="store_true", 
                       help="Use full backend with real agents (slower startup)")
    args = parser.parse_args()
    
    # Choose backend
    backend_file = "backend.py" if args.full else "backend_fast.py"
    mode_name = "Full Mode (Real Agents)" if args.full else "Fast Mode (Mock Agents)"
    
    print_header()
    print_status(f"Starting {mode_name}")
    
    # Check requirements
    if not check_requirements():
        print_status("Please fix missing files above", "ERROR")
        sys.exit(1)
    
    # Start backend
    print_status(f"Starting backend server...")
    
    try:
        # Import and run the backend directly
        if args.full:
            print_status("Loading real agents - this may take 30+ seconds...", "WARNING")
            import backend
            app = backend.app
        else:
            print_status("Using fast backend with mock agents...", "SUCCESS") 
            import backend_fast
            app = backend_fast.app
        
        # Start uvicorn
        import uvicorn
        
        print_status("Backend server starting...", "LOADING")
        print_status("Opening browser in 3 seconds...")
        
        # Open browser after short delay
        def open_browser():
            time.sleep(3)
            try:
                webbrowser.open("http://localhost:8000")
                print_status("Browser opened to http://localhost:8000", "SUCCESS")
            except:
                print_status("Could not open browser automatically", "WARNING")
                print_status("Please open: http://localhost:8000", "INFO")
        
        import threading
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        # Print final instructions
        print("\n" + "="*60)
        print("🎉 Enhanced RAG Pipeline Dashboard")
        print("   🔗 URL: http://localhost:8000")
        print("   📖 API: http://localhost:8000/docs")
        print("   ❤️  Health: http://localhost:8000/api/health")
        print("="*60)
        print(f"   ⚡ Mode: {mode_name}")
        print("   📋 Features: Live Theater, Analysis, Scanner, Viz")
        print("   ⚠️  Press Ctrl+C to stop")
        print("="*60)
        
        # Start the server
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
        
    except KeyboardInterrupt:
        print_status("\nShutting down dashboard...", "INFO")
        print_status("Goodbye! 👋", "SUCCESS")
    except Exception as e:
        print_status(f"Error starting dashboard: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()