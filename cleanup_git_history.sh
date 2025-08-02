#!/bin/bash

# 🧹 Git History Cleanup Script for Enhanced RAG Pipeline
# This script removes large files from git history to prepare for GitHub migration

echo "🧹 Starting Git History Cleanup for Enhanced RAG Pipeline"
echo "⚠️  WARNING: This will rewrite git history. Make sure you have a backup!"
echo ""

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a git repository. Please run this script from the repository root."
    exit 1
fi

# Confirm before proceeding
read -p "Are you sure you want to proceed? This will rewrite git history! (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo "🔄 Starting cleanup process..."

# Function to remove files/directories from git history
cleanup_pattern() {
    local pattern="$1"
    local description="$2"
    
    echo "🗑️  Removing $description ($pattern)..."
    
    git filter-branch --force --index-filter \
        "git rm --cached --ignore-unmatch -r '$pattern' 2>/dev/null || true" \
        --prune-empty --tag-name-filter cat -- --all
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully removed $description"
    else
        echo "⚠️  Warning: Issues removing $description (may not exist in history)"
    fi
}

# 1. Remove DuckDB files
echo ""
echo "📊 Cleaning DuckDB files..."
cleanup_pattern "data/*.duckdb*" "DuckDB database files"
cleanup_pattern "*.duckdb*" "any DuckDB files"

# 2. Remove LanceDB stores
echo ""
echo "🗄️  Cleaning LanceDB vector stores..."
cleanup_pattern "data/storage_lancedb_store" "main LanceDB store"
cleanup_pattern "data/lancedb_store" "alternative LanceDB store"
cleanup_pattern "lancedb_store" "any LanceDB directories"
cleanup_pattern "storage" "storage directories"

# 3. Remove large model files
echo ""
echo "🤖 Cleaning model files..."
cleanup_pattern "models/sentence_transformers" "cached transformer models"
cleanup_pattern "models/ensemble" "ensemble models"
cleanup_pattern "models/financial" "financial models"
cleanup_pattern "models/text" "text models"
cleanup_pattern "models/3stage*" "3-stage models"
cleanup_pattern "models_clean" "clean models"

# 4. Remove generated data directories
echo ""
echo "📁 Cleaning generated data..."
cleanup_pattern "data/ml_features" "ML features"
cleanup_pattern "data/ml_features_clean" "clean ML features"
cleanup_pattern "data/ml_pipeline_output" "pipeline outputs"
cleanup_pattern "data/predictions" "predictions"
cleanup_pattern "data/predictions_corrected" "corrected predictions"
cleanup_pattern "data/leakage_analysis" "leakage analysis"
cleanup_pattern "evaluation_results" "evaluation results"
cleanup_pattern "visualizations" "visualizations"
cleanup_pattern "ml/analysis" "ML analysis outputs"
cleanup_pattern "ml/prediction" "ML predictions"
cleanup_pattern "test_output" "test outputs"

# 5. Remove pickle files
echo ""
echo "🥒 Cleaning pickle files..."
cleanup_pattern "*.pkl" "pickle files"
cleanup_pattern "*.pickle" "pickle files"
cleanup_pattern "data/*.pkl" "data pickle files"

# 6. Remove temporary and cache files
echo ""
echo "🗂️  Cleaning temporary files..."
cleanup_pattern "*.tmp" "temporary files"
cleanup_pattern "*.log" "log files"
cleanup_pattern "__pycache__" "Python cache"

# Clean up refs and force garbage collection
echo ""
echo "🧽 Cleaning up git references and running garbage collection..."

# Remove backup refs created by filter-branch
git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin

# Expire reflog
git reflog expire --expire=now --all

# Aggressive garbage collection
git gc --prune=now --aggressive

echo ""
echo "📊 Repository size analysis:"
echo "Current repository size:"
du -sh .git

echo ""
echo "Checking largest remaining files:"
git ls-files | xargs -I {} sh -c 'echo "$(git cat-file -s {} 2>/dev/null || echo 0) {}"' | sort -nr | head -10

echo ""
echo "✅ Git history cleanup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Review the changes: git log --oneline"
echo "2. Check repository size is < 100MB"
echo "3. Verify all essential files are still present"
echo "4. Push to GitHub: git push origin --force --all"
echo ""
echo "⚠️  Important notes:"
echo "- Large data files should be uploaded separately to Hetzner server"
echo "- DuckDB files (~250MB) need manual upload"
echo "- LanceDB vector stores need separate transfer"
echo "- Model files can be re-downloaded with download_models.py"
echo ""
echo "🎯 Your repository is now ready for GitHub migration!"