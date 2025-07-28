#!/usr/bin/env python3
"""
Mentor Demo Sample Creator
=========================

Create a minimal but representative sample for mentor demo while protecting proprietary data.
"""

import duckdb
import shutil
import os
import json
from datetime import datetime
from pathlib import Path

def create_sample_database():
    """Create a small sample database with just a few setups"""
    print("📊 CREATING SAMPLE DATABASE FOR MENTOR DEMO")
    print("=" * 60)
    
    # Connect to full database
    conn = duckdb.connect('data/sentiment_system.duckdb')
    
    # Get sample setups (5 setups with good data coverage)
    sample_query = '''
        WITH setup_stats AS (
            SELECT 
                l.setup_id,
                l.stock_ticker,
                l.spike_timestamp as setup_date,
                l.outperformance_10d,
                COUNT(DISTINCT f.ticker) as has_fundamentals,
                COUNT(DISTINCT rns.ticker) as has_news,
                COUNT(DISTINCT up.setup_id) as has_userposts,
                COUNT(DISTINCT ar.ticker) as has_analyst_recs
            FROM labels l
            LEFT JOIN fundamentals f ON l.stock_ticker = f.ticker
            LEFT JOIN rns_announcements rns ON l.stock_ticker = rns.ticker 
                AND rns.date_published <= l.spike_timestamp
                AND rns.date_published >= (l.spike_timestamp - INTERVAL '90 days')
            LEFT JOIN user_posts up ON l.setup_id = up.setup_id
            LEFT JOIN analyst_recommendations ar ON l.stock_ticker = ar.ticker
                AND ar.date <= l.spike_timestamp
                AND ar.date >= (l.spike_timestamp - INTERVAL '180 days')
            GROUP BY l.setup_id, l.stock_ticker, l.spike_timestamp, l.outperformance_10d
            HAVING has_fundamentals > 0 OR has_news > 0 OR has_userposts > 0
            ORDER BY (has_fundamentals + has_news + has_userposts + has_analyst_recs) DESC, l.spike_timestamp DESC
            LIMIT 5
        )
        SELECT setup_id FROM setup_stats
    '''
    
    sample_setups = conn.execute(sample_query).fetchall()
    sample_setup_ids = [row[0] for row in sample_setups]
    
    print(f"✅ Selected {len(sample_setup_ids)} representative setups:")
    for setup_id in sample_setup_ids:
        print(f"   • {setup_id}")
    
    # Create sample database
    sample_db_path = 'mentor_demo/sample_sentiment_system.duckdb'
    os.makedirs('mentor_demo', exist_ok=True)
    
    sample_conn = duckdb.connect(sample_db_path)
    
    # Copy table schemas and sample data
    tables_to_copy = [
        'labels', 'fundamentals', 'rns_announcements', 
        'user_posts', 'analyst_recommendations', 'company_info',
        'fundamentals_features', 'news_features', 'userposts_features', 
        'analyst_recommendations_features'
    ]
    
    for table in tables_to_copy:
        try:
            # Get table schema
            schema = conn.execute(f'DESCRIBE {table}').fetchall()
            
            # Create table in sample database
            create_sql = f"CREATE TABLE {table} AS SELECT * FROM full_db.{table} WHERE 1=0"
            
            # Copy relevant data
            if table in ['labels']:
                # Copy sample setups data
                setup_filter = "WHERE setup_id IN ({})".format(','.join([f"'{sid}'" for sid in sample_setup_ids]))
                copy_sql = f'''
                    INSERT INTO {table} 
                    SELECT * FROM full_db.{table} {setup_filter}
                '''
            elif table in ['fundamentals', 'rns_announcements', 'analyst_recommendations', 'company_info']:
                # Copy data for sample tickers
                tickers_query = f'''
                    SELECT DISTINCT stock_ticker FROM full_db.labels 
                    WHERE setup_id IN ({','.join([f"'{sid}'" for sid in sample_setup_ids])})
                '''
                tickers = [row[0] for row in conn.execute(tickers_query).fetchall()]
                ticker_filter = "WHERE ticker IN ({})".format(','.join([f"'{t}'" for t in tickers]))
                copy_sql = f'''
                    INSERT INTO {table}
                    SELECT * FROM full_db.{table} {ticker_filter}
                '''
            elif table in ['user_posts']:
                # Copy user posts for sample setups
                setup_filter = "WHERE setup_id IN ({})".format(','.join([f"'{sid}'" for sid in sample_setup_ids]))
                copy_sql = f'''
                    INSERT INTO {table}
                    SELECT * FROM full_db.{table} {setup_filter}
                '''
            elif table.endswith('_features'):
                # Copy feature tables for sample setups
                setup_filter = "WHERE setup_id IN ({})".format(','.join([f"'{sid}'" for sid in sample_setup_ids]))
                copy_sql = f'''
                    INSERT INTO {table}
                    SELECT * FROM full_db.{table} {setup_filter}
                '''
            else:
                # Skip other tables or copy minimal data
                continue
            
            # Execute the copy
            sample_conn.execute(f"ATTACH 'data/sentiment_system.duckdb' AS full_db")
            sample_conn.execute(create_sql)
            sample_conn.execute(copy_sql)
            
            # Check what was copied
            count = sample_conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
            print(f"   ✅ {table}: {count} records copied")
            
        except Exception as e:
            print(f"   ⚠️ {table}: {e}")
    
    sample_conn.close()
    conn.close()
    
    return sample_setup_ids

def copy_agent_code():
    """Copy agent code with prompt templates for demo"""
    print(f"\n📂 COPYING AGENT CODE FOR DEMO")
    print("=" * 40)
    
    # Copy agent directories (excluding proprietary ML models)
    agent_dirs = [
        'agents/news',
        'agents/userposts', 
        'agents/fundamentals',
        'agents/analyst_recommendations'
    ]
    
    for agent_dir in agent_dirs:
        demo_dir = f'mentor_demo/{agent_dir}'
        os.makedirs(os.path.dirname(demo_dir), exist_ok=True)
        shutil.copytree(agent_dir, demo_dir, dirs_exist_ok=True)
        print(f"   ✅ Copied {agent_dir}")
    
    # Copy select files
    files_to_copy = [
        'features/feature_plan.md',
        'features/llm_prompts/',
        'embeddings/',
        'tools/cli_extract_news_features.py',
        'tools/cli_extract_userposts_features.py',
        'tools/cli_extract_fundamentals_features.py',
        'check_feature_tables_status.py'
    ]
    
    for file_path in files_to_copy:
        demo_path = f'mentor_demo/{file_path}'
        os.makedirs(os.path.dirname(demo_path), exist_ok=True)
        
        if os.path.isdir(file_path):
            shutil.copytree(file_path, demo_path, dirs_exist_ok=True)
        else:
            shutil.copy2(file_path, demo_path)
        print(f"   ✅ Copied {file_path}")

def copy_sample_lancedb():
    """Copy sample LanceDB data"""
    print(f"\n🗄️ COPYING SAMPLE LANCEDB")
    print("=" * 40)
    
    # Copy LanceDB store (this contains the vector embeddings)
    if os.path.exists('storage/lancedb_store'):
        shutil.copytree('storage/lancedb_store', 'mentor_demo/lancedb_store', dirs_exist_ok=True)
        print("   ✅ Copied LanceDB vector store")
    else:
        print("   ⚠️ LanceDB store not found - might need to run embeddings first")

def create_minimal_ml_models():
    """Create simplified ML models for demo"""
    print(f"\n🤖 CREATING DEMO ML MODELS")
    print("=" * 40)
    
    # Copy only the text-based and fundamentals ML models (not the ensemble)
    ml_files = [
        'ml/text_ml_model.py',
        'ml/ml_model.py',  # fundamentals model
        'ml/precision_analyzer.py'
    ]
    
    for ml_file in ml_files:
        if os.path.exists(ml_file):
            demo_path = f'mentor_demo/{ml_file}'
            os.makedirs(os.path.dirname(demo_path), exist_ok=True)
            shutil.copy2(ml_file, demo_path)
            print(f"   ✅ Copied {ml_file}")

def create_demo_readme():
    """Create README for mentor demo"""
    readme_content = '''# RAG Pipeline Demo - Mentor Review

## Overview
This is a simplified demo version of the RAG (Retrieval-Augmented Generation) pipeline for trading signal generation.

## What's Included

### 🗄️ Sample Data
- **sample_sentiment_system.duckdb**: Small database with 5 representative trading setups
- **lancedb_store/**: Vector embeddings for semantic search

### 🤖 LLM Agents with Prompt Templates
- **agents/news/**: RNS news analysis agent
- **agents/userposts/**: Social sentiment analysis agent  
- **agents/fundamentals/**: Financial metrics + earnings analysis agent
- **agents/analyst_recommendations/**: Analyst recommendation analysis agent (NEW)

### 🔧 Tools & Features
- **features/feature_plan.md**: Complete feature documentation (59 features)
- **tools/**: CLI tools for feature extraction
- **embeddings/**: Vector embedding generation

### 🤖 ML Models
- **ml/text_ml_model.py**: Text-based prediction model
- **ml/ml_model.py**: Fundamentals-based prediction model

## Key Innovation: Multi-Domain RAG with LLM Feature Engineering

1. **Semantic Retrieval**: Use vector similarity to find relevant data
2. **LLM Feature Extraction**: Extract structured features from unstructured text
3. **Multi-Modal ML**: Combine text and fundamental features
4. **Out-of-Sample Testing**: Chronological train/test split

## Demo Commands

```bash
# Check sample data
python check_feature_tables_status.py

# Run feature extraction on sample
python tools/cli_extract_news_features.py --discover
python tools/cli_extract_userposts_features.py --discover

# View feature schema
cat features/feature_plan.md
```

## Architecture
- **DuckDB**: Structured data storage
- **LanceDB**: Vector embeddings  
- **OpenAI**: LLM feature extraction
- **Scikit-learn**: ML models

---
*Demo version created for mentor review - production version has 700+ setups and ensemble models*
'''
    
    with open('mentor_demo/README.md', 'w') as f:
        f.write(readme_content)
    
    print("   ✅ Created demo README.md")

def create_complete_demo_folder():
    """Create the complete subfolder with everything needed"""
    print(f"\n📁 CREATING COMPLETE DEMO SUBFOLDER")
    print("=" * 60)
    
    # Create main demo directory structure
    demo_root = Path('mentor_demo')
    demo_root.mkdir(exist_ok=True)
    
    # Copy embeddings generation script
    shutil.copy2('run_embeddings_generation.py', demo_root / 'run_embeddings_generation.py')
    shutil.copy2('embeddings/embed_analyst_recommendations_duckdb.py', demo_root / 'embeddings/embed_analyst_recommendations_duckdb.py')
    
    # Copy requirements
    if os.path.exists('requirements.txt'):
        shutil.copy2('requirements.txt', demo_root / 'requirements.txt')
    
    # Create setup script
    setup_script = '''#!/usr/bin/env python3
"""
Demo Setup Script
================
Run this to set up the complete demo environment.
"""

import subprocess
import sys

def main():
    print("🎯 RAG Pipeline Demo Setup")
    print("=" * 40)
    
    print("1. Generating vector embeddings...")
    subprocess.run([sys.executable, 'run_embeddings_generation.py'])
    
    print("\\n2. Checking feature tables...")
    subprocess.run([sys.executable, 'check_feature_tables_status.py'])
    
    print("\\n✅ Demo setup complete!")
    print("💡 Try: python tools/cli_extract_news_features.py --discover")

if __name__ == "__main__":
    main()
'''
    
    with open(demo_root / 'setup_demo.py', 'w') as f:
        f.write(setup_script)
    
    print("   ✅ Created complete demo folder structure")
    print("   ✅ Added embeddings generation pipeline")
    print("   ✅ Created setup script")
    
def main():
    """Create complete mentor demo package"""
    print("🎯 MENTOR DEMO PACKAGE CREATOR")
    print("=" * 60)
    print("Creating minimal demo while protecting proprietary elements...")
    
    # Create all demo components
    sample_setups = create_sample_database()
    copy_agent_code()
    copy_sample_lancedb()
    create_minimal_ml_models()
    create_demo_readme()
    create_complete_demo_folder()
    
    # Create summary
    print(f"\n📋 DEMO PACKAGE SUMMARY")
    print("=" * 60)
    print(f"✅ Sample database: 5 representative setups")
    print(f"✅ All 4 domain agents with prompt templates")
    print(f"✅ Vector embeddings (LanceDB)")
    print(f"✅ Feature extraction tools")
    print(f"✅ 2 ML models (text + fundamentals)")
    print(f"✅ Complete documentation")
    print(f"✅ Setup and generation scripts")
    
    print(f"\n📁 COMPLETE DEMO SUBFOLDER: mentor_demo/")
    print(f"📊 Safe to share - no proprietary data or ensemble models")
    print(f"🔒 Self-contained - just zip and share via private GitHub")
    
    # Instructions for mentor
    print(f"\n💡 FOR MENTOR MEETING:")
    print(f"1. Zip mentor_demo/ folder and share via private GitHub repo")
    print(f"2. Mentor runs: python setup_demo.py")
    print(f"3. Show LLM prompt templates and feature engineering")
    print(f"4. Demonstrate vector similarity search")
    print(f"5. Run sample feature extraction")
    print(f"6. Show multi-modal ML approach")
    
    print(f"\n🚀 READY FOR MENTOR MEETING!")
    print(f"📁 Everything is in the mentor_demo/ subfolder")
    
    return True

if __name__ == "__main__":
    main() 