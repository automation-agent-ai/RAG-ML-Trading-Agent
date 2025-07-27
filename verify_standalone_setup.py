#!/usr/bin/env python3
"""
Standalone Production Pipeline Setup Verification

Run this script to verify that the production pipeline is properly set up
and all dependencies are available.
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (Requires 3.9+)")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'duckdb', 
        'lancedb', 'openai', 'sentence_transformers', 
        'pydantic', 'nltk'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

def check_environment_variables():
    """Check if required environment variables are set"""
    print("\n🔑 Checking environment variables...")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        if openai_key.startswith('sk-'):
            print("   ✅ OPENAI_API_KEY (properly formatted)")
            return True
        else:
            print("   ⚠️  OPENAI_API_KEY (set but format looks incorrect)")
            return False
    else:
        print("   ❌ OPENAI_API_KEY (not set)")
        print("   Set with: $env:OPENAI_API_KEY = 'sk-your-key' (PowerShell)")
        print("   Or create .env file with: OPENAI_API_KEY=sk-your-key")
        return False

def check_file_structure():
    """Check if required files and directories exist"""
    print("\n📂 Checking file structure...")
    
    required_items = [
        ('file', 'run_complete_ml_pipeline.py'),
        ('file', 'requirements.txt'),
        ('dir', 'agents'),
        ('dir', 'embeddings'),
        ('dir', 'core'),
        ('dir', 'data'),
        ('file', 'data/sentiment_system.duckdb')
    ]
    
    all_present = True
    
    for item_type, item_path in required_items:
        path = Path(item_path)
        
        if item_type == 'file' and path.is_file():
            print(f"   ✅ {item_path}")
        elif item_type == 'dir' and path.is_dir():
            print(f"   ✅ {item_path}/")
        else:
            print(f"   ❌ {item_path} ({item_type} missing)")
            all_present = False
    
    return all_present

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test DuckDB connection
        import duckdb
        conn = duckdb.connect('data/sentiment_system.duckdb')
        tables = conn.execute('SHOW TABLES').fetchall()
        conn.close()
        print(f"   ✅ DuckDB connection ({len(tables)} tables found)")
        
        # Test OpenAI import
        from openai import OpenAI
        print("   ✅ OpenAI client import")
        
        # Test LanceDB import
        import lancedb
        print("   ✅ LanceDB import")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Functionality test failed: {e}")
        return False

def main():
    """Main verification function"""
    print("🚀 Production Pipeline Setup Verification")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_environment_variables(),
        check_file_structure(),
        test_basic_functionality()
    ]
    
    print("\n" + "=" * 50)
    
    if all(checks):
        print("🎉 ✅ ALL CHECKS PASSED!")
        print("\n🚀 Your production pipeline is ready to use!")
        print("\n📚 Next steps:")
        print("   - Review: COMPLETE_PIPELINE_GUIDE.md")
        print("   - Quick start: python complete_workflow.py --demo")
        print("   - Full pipeline: python run_complete_ml_pipeline.py --mode training")
        return True
    else:
        print("❌ SOME CHECKS FAILED!")
        print("\n🔧 Please fix the issues above before proceeding.")
        print("📚 See: SETUP_GUIDE.md for detailed setup instructions")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 