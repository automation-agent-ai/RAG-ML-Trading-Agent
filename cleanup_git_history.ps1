# 🧹 Git History Cleanup PowerShell Script for Enhanced RAG Pipeline
# This script removes large files from git history to prepare for GitHub migration

Write-Host "🧹 Starting Git History Cleanup for Enhanced RAG Pipeline" -ForegroundColor Cyan
Write-Host "⚠️  WARNING: This will rewrite git history. Make sure you have a backup!" -ForegroundColor Yellow
Write-Host ""

# Check if we're in a git repository
if (-not (Test-Path ".git")) {
    Write-Host "❌ Error: Not in a git repository. Please run this script from the repository root." -ForegroundColor Red
    exit 1
}

# Confirm before proceeding
$confirmation = Read-Host "Are you sure you want to proceed? This will rewrite git history! (y/N)"
if ($confirmation -ne "y" -and $confirmation -ne "Y") {
    Write-Host "Aborted." -ForegroundColor Yellow
    exit 1
}

Write-Host "🔄 Starting cleanup process..." -ForegroundColor Cyan

# Function to remove files/directories from git history
function Cleanup-Pattern {
    param (
        [string]$pattern,
        [string]$description
    )
    
    Write-Host "🗑️  Removing $description ($pattern)..." -ForegroundColor Cyan
    
    try {
        git filter-branch --force --index-filter "git rm --cached --ignore-unmatch -r '$pattern' 2>nul || exit 0" --prune-empty --tag-name-filter cat -- --all
        Write-Host "✅ Successfully removed $description" -ForegroundColor Green
    }
    catch {
        Write-Host "⚠️  Warning: Issues removing $description (may not exist in history)" -ForegroundColor Yellow
    }
}

# 1. Remove DuckDB files
Write-Host ""
Write-Host "📊 Cleaning DuckDB files..." -ForegroundColor Cyan
Cleanup-Pattern "data/*.duckdb*" "DuckDB database files"
Cleanup-Pattern "*.duckdb*" "any DuckDB files"

# 2. Remove LanceDB stores
Write-Host ""
Write-Host "🗄️  Cleaning LanceDB vector stores..." -ForegroundColor Cyan
Cleanup-Pattern "data/storage_lancedb_store" "main LanceDB store"
Cleanup-Pattern "data/lancedb_store" "alternative LanceDB store"
Cleanup-Pattern "lancedb_store" "any LanceDB directories"
Cleanup-Pattern "storage" "storage directories"

# 3. Remove large model files
Write-Host ""
Write-Host "🤖 Cleaning model files..." -ForegroundColor Cyan
Cleanup-Pattern "models/sentence_transformers" "cached transformer models"
Cleanup-Pattern "models/ensemble" "ensemble models"
Cleanup-Pattern "models/financial" "financial models"
Cleanup-Pattern "models/text" "text models"
Cleanup-Pattern "models/3stage*" "3-stage models"
Cleanup-Pattern "models_clean" "clean models"

# 4. Remove generated data directories
Write-Host ""
Write-Host "📁 Cleaning generated data..." -ForegroundColor Cyan
Cleanup-Pattern "data/ml_features" "ML features"
Cleanup-Pattern "data/ml_features_clean" "clean ML features"
Cleanup-Pattern "data/ml_pipeline_output" "pipeline outputs"
Cleanup-Pattern "data/predictions" "predictions"
Cleanup-Pattern "data/predictions_corrected" "corrected predictions"
Cleanup-Pattern "data/leakage_analysis" "leakage analysis"
Cleanup-Pattern "evaluation_results" "evaluation results"
Cleanup-Pattern "visualizations" "visualizations"
Cleanup-Pattern "ml/analysis" "ML analysis outputs"
Cleanup-Pattern "ml/prediction" "ML predictions"
Cleanup-Pattern "test_output" "test outputs"

# 5. Remove pickle files
Write-Host ""
Write-Host "🥒 Cleaning pickle files..." -ForegroundColor Cyan
Cleanup-Pattern "*.pkl" "pickle files"
Cleanup-Pattern "*.pickle" "pickle files"
Cleanup-Pattern "data/*.pkl" "data pickle files"

# 6. Remove temporary and cache files
Write-Host ""
Write-Host "🗂️  Cleaning temporary files..." -ForegroundColor Cyan
Cleanup-Pattern "*.tmp" "temporary files"
Cleanup-Pattern "*.log" "log files"
Cleanup-Pattern "__pycache__" "Python cache"

# Clean up refs and force garbage collection
Write-Host ""
Write-Host "🧽 Cleaning up git references and running garbage collection..." -ForegroundColor Cyan

# Remove backup refs created by filter-branch
git for-each-ref --format="delete %(refname)" refs/original | ForEach-Object { git update-ref $_ }

# Expire reflog
git reflog expire --expire=now --all

# Aggressive garbage collection
git gc --prune=now --aggressive

Write-Host ""
Write-Host "📊 Repository size analysis:" -ForegroundColor Cyan
Write-Host "Current repository size:"
$gitSize = (Get-ChildItem -Path .git -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "$([math]::Round($gitSize, 2)) MB" -ForegroundColor Green

Write-Host ""
Write-Host "Checking largest remaining files:" -ForegroundColor Cyan
git ls-files | ForEach-Object { 
    $size = git cat-file -s $_
    "$size $_"
} | Sort-Object -Descending { [int]($_ -split ' ')[0] } | Select-Object -First 10

Write-Host ""
Write-Host "✅ Git history cleanup completed!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Review the changes: git log --oneline" -ForegroundColor White
Write-Host "2. Check repository size is < 100MB" -ForegroundColor White
Write-Host "3. Verify all essential files are still present" -ForegroundColor White
Write-Host "4. Push to GitHub: git push origin --force --all" -ForegroundColor White
Write-Host ""
Write-Host "⚠️  Important notes:" -ForegroundColor Yellow
Write-Host "- Large data files should be uploaded separately to Hetzner server" -ForegroundColor White
Write-Host "- DuckDB files (~250MB) need manual upload" -ForegroundColor White
Write-Host "- LanceDB vector stores need separate transfer" -ForegroundColor White
Write-Host "- Model files can be re-downloaded with download_models.py" -ForegroundColor White
Write-Host ""
Write-Host "🎯 Your repository is now ready for GitHub migration!" -ForegroundColor Green