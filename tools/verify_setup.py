#!/usr/bin/env python3
"""
RAG Pipeline v2 Setup Verification Script

This script verifies that all components of the RAG pipeline v2 are properly
configured and can communicate with each other.

Usage:
    python tools/verify_setup.py
"""

import sys
import os
from pathlib import Path
import importlib.util
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_file_exists(file_path: str, description: str) -> bool:
    """Check if a file exists and report result"""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - NOT FOUND")
        return False

def check_import(module_path: str, module_name: str, description: str) -> bool:
    """Check if a module can be imported"""
    try:
        # Add parent directory to path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"✅ {description}: Import successful")
        return True
    except Exception as e:
        print(f"❌ {description}: Import failed - {e}")
        return False

def check_database_connection(db_path: str) -> bool:
    """Check DuckDB connection"""
    try:
        import duckdb
        conn = duckdb.connect(db_path)
        result = conn.execute("SELECT COUNT(*) FROM setups").fetchone()
        conn.close()
        print(f"✅ DuckDB Connection: {result[0]} setups found")
        return True
    except Exception as e:
        print(f"❌ DuckDB Connection failed: {e}")
        return False

def check_lancedb_connection(lancedb_path: str) -> bool:
    """Check LanceDB connection"""
    try:
        import lancedb
        db = lancedb.connect(lancedb_path)
        tables = db.table_names()
        print(f"✅ LanceDB Connection: {len(tables)} tables found - {tables}")
        return True
    except Exception as e:
        print(f"❌ LanceDB Connection failed: {e}")
        return False

def check_openai_api() -> bool:
    """Check OpenAI API configuration"""
    try:
        from openai import OpenAI
        client = OpenAI()
        # Make a minimal test call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        print(f"✅ OpenAI API: Connection successful")
        return True
    except Exception as e:
        print(f"❌ OpenAI API failed: {e}")
        return False

def main():
    """Main verification function"""
    print("🔍 RAG Pipeline v2 Setup Verification")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Check core structure
    print("\n📁 Directory Structure:")
    structure_checks = [
        ("orchestration/run_complete_pipeline_duckdb.py", "Main Pipeline Runner"),
        ("orchestration/orchestrator_langgraph.py", "LangGraph Orchestrator"),
        ("agents/news/enhanced_news_agent_duckdb.py", "News Agent"),
        ("agents/userposts/enhanced_userposts_agent_complete.py", "UserPosts Agent"),
        ("agents/fundamentals/enhanced_fundamentals_agent_duckdb.py", "Fundamentals Agent"),
        ("embeddings/embed_news_duckdb.py", "News Embeddings"),
        ("embeddings/embed_userposts_duckdb.py", "UserPosts Embeddings"),
        ("embeddings/embed_fundamentals_duckdb.py", "Fundamentals Embeddings"),
        ("ml/ml_model.py", "ML Model"),
        ("ml/hybrid_model.py", "Hybrid Model"),
        ("ml/kg_explainer.py", "Knowledge Graph Explainer"),
        ("data/sentiment_system.duckdb", "DuckDB Database"),
        ("storage/lancedb_store", "LanceDB Store"),
        ("requirements.txt", "Dependencies"),
        ("README.md", "Documentation")
    ]
    
    for file_path, description in structure_checks:
        if not check_file_exists(file_path, description):
            all_checks_passed = False
    
    # Check CLI tools
    print("\n🛠️ CLI Tools:")
    cli_tools = [
        ("tools/cli_extract_news_features.py", "News CLI"),
        ("tools/cli_extract_userposts_features.py", "UserPosts CLI"),
        ("tools/cli_extract_fundamentals_features.py", "Fundamentals CLI"),
        ("tools/export_features_duckdb.py", "Feature Export"),
        ("tools/setup_validator_duckdb.py", "Setup Validator")
    ]
    
    for file_path, description in cli_tools:
        if not check_file_exists(file_path, description):
            all_checks_passed = False
    
    # Check Python imports
    print("\n🐍 Python Module Imports:")
    import_checks = [
        ("agents/news/enhanced_news_agent_duckdb.py", "enhanced_news_agent", "News Agent Import"),
        ("agents/fundamentals/enhanced_fundamentals_agent_duckdb.py", "enhanced_fundamentals_agent", "Fundamentals Agent Import"),
        ("ml/ml_model.py", "ml_model", "ML Model Import"),
        ("ml/hybrid_model.py", "hybrid_model", "Hybrid Model Import")
    ]
    
    for module_path, module_name, description in import_checks:
        if not check_import(module_path, module_name, description):
            all_checks_passed = False
    
    # Check database connections
    print("\n💾 Database Connections:")
    if not check_database_connection("data/sentiment_system.duckdb"):
        all_checks_passed = False
    
    if not check_lancedb_connection("storage/lancedb_store"):
        all_checks_passed = False
    
    # Check API connections
    print("\n🌐 API Connections:")
    if not check_openai_api():
        print("⚠️  OpenAI API check skipped - configure .env file first")
    
    # Check dependencies
    print("\n📦 Core Dependencies:")
    core_deps = [
        "duckdb", "lancedb", "pandas", "numpy", "openai", 
        "sentence_transformers", "pydantic", "scikit_learn"
    ]
    
    missing_deps = []
    for dep in core_deps:
        try:
            if dep == "scikit_learn":
                import sklearn
            else:
                __import__(dep.replace("-", "_"))
            print(f"✅ {dep}: Installed")
        except ImportError:
            print(f"❌ {dep}: Missing")
            missing_deps.append(dep)
            all_checks_passed = False
    
    # Final summary
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 ALL CHECKS PASSED! RAG Pipeline v2 is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Configure .env file with your OpenAI API key")
        print("2. Run: python orchestration/run_complete_pipeline_duckdb.py")
        print("3. Or try: python tools/cli_extract_news_features.py --discover")
        return True
    else:
        print("❌ SOME CHECKS FAILED. Please address the issues above.")
        if missing_deps:
            print(f"\n📦 Install missing dependencies:")
            print(f"pip install {' '.join(missing_deps)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 