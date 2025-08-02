# 🔒 Safe Git History Cleanup using git-filter-repo (Recommended)
# This script safely removes large files from git history for GitHub migration

Write-Host "🔒 Safe Git History Cleanup for Enhanced RAG Pipeline" -ForegroundColor Cyan
Write-Host "Using git-filter-repo (recommended by Git maintainers)" -ForegroundColor Green
Write-Host ""

# Check if we're in a git repository
if (-not (Test-Path ".git")) {
    Write-Host "❌ Error: Not in a git repository. Please run this script from the repository root." -ForegroundColor Red
    exit 1
}

# Check if git-filter-repo is installed
try {
    git filter-repo --help | Out-Null
    Write-Host "✅ git-filter-repo is installed" -ForegroundColor Green
}
catch {
    Write-Host "❌ git-filter-repo is not installed. Installing..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please install git-filter-repo first:" -ForegroundColor Cyan
    Write-Host "1. pip install git-filter-repo" -ForegroundColor White
    Write-Host "2. Or download from: https://github.com/newren/git-filter-repo" -ForegroundColor White
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "⚠️  IMPORTANT NOTES:" -ForegroundColor Yellow
Write-Host "- This ONLY removes files from git history" -ForegroundColor White
Write-Host "- Your local files will remain completely untouched" -ForegroundColor White
Write-Host "- Large data files stay on your drive for separate upload" -ForegroundColor White
Write-Host "- Creates a backup branch before cleanup" -ForegroundColor White
Write-Host ""

# Confirm before proceeding
$confirmation = Read-Host "Proceed with safe git history cleanup? (y/N)"
if ($confirmation -ne "y" -and $confirmation -ne "Y") {
    Write-Host "Aborted." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "🔄 Starting safe cleanup process..." -ForegroundColor Cyan

# Create backup branch
Write-Host "📋 Creating backup branch..." -ForegroundColor Cyan
git branch backup-before-cleanup 2>$null
Write-Host "✅ Backup branch 'backup-before-cleanup' created" -ForegroundColor Green

# Show repository size before cleanup
Write-Host ""
Write-Host "📊 Repository size before cleanup:" -ForegroundColor Cyan
$gitSizeBefore = (Get-ChildItem -Path .git -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "$([math]::Round($gitSizeBefore, 2)) MB" -ForegroundColor Yellow

Write-Host ""
Write-Host "🗑️ Removing large files from git history..." -ForegroundColor Cyan

# Define patterns to remove (these files will stay on your local drive)
$patterns = @(
    # Database files
    "data/*.duckdb*",
    "*.duckdb*",
    
    # LanceDB vector stores
    "data/storage_lancedb_store/",
    "data/lancedb_store/",
    "lancedb_store/",
    "storage/",
    
    # Model files
    "models/sentence_transformers/",
    "models/ensemble/",
    "models/financial/",
    "models/text/",
    "models/3stage*/",
    "models_clean/",
    
    # Generated data
    "data/ml_features/",
    "data/ml_features_clean/",
    "data/ml_pipeline_output/",
    "data/predictions/",
    "data/predictions_corrected/",
    "data/leakage_analysis/",
    "evaluation_results/",
    "visualizations/",
    "ml/analysis/",
    "ml/prediction/",
    "test_output/",
    
    # Pickle and temporary files
    "*.pkl",
    "*.pickle",
    "data/*.pkl",
    "*.tmp",
    "*.log",
    "__pycache__/"
)

# Build the git-filter-repo command
$pathArgs = @()
foreach ($pattern in $patterns) {
    $pathArgs += "--path"
    $pathArgs += $pattern
}

try {
    # Execute git-filter-repo with all patterns
    Write-Host "Executing: git filter-repo --invert-paths" -ForegroundColor Gray
    & git filter-repo --invert-paths @pathArgs --force
    
    Write-Host "✅ Git history cleanup completed successfully!" -ForegroundColor Green
}
catch {
    Write-Host "❌ Error during cleanup: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "You can restore from backup: git checkout backup-before-cleanup" -ForegroundColor Yellow
    exit 1
}

# Show repository size after cleanup
Write-Host ""
Write-Host "📊 Repository size after cleanup:" -ForegroundColor Cyan
$gitSizeAfter = (Get-ChildItem -Path .git -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "$([math]::Round($gitSizeAfter, 2)) MB" -ForegroundColor Green
$sizeSaved = $gitSizeBefore - $gitSizeAfter
Write-Host "💾 Space saved: $([math]::Round($sizeSaved, 2)) MB" -ForegroundColor Green

# Verify local files are still present
Write-Host ""
Write-Host "🔍 Verifying your local files are still present:" -ForegroundColor Cyan
$localFiles = @(
    "data/sentiment_system.duckdb",
    "data/storage_lancedb_store",
    "models/sentence_transformers"
)

foreach ($file in $localFiles) {
    if (Test-Path $file) {
        Write-Host "✅ $file - Still present locally" -ForegroundColor Green
    } else {
        Write-Host "ℹ️  $file - Not found (normal if not created yet)" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Review changes: git log --oneline" -ForegroundColor White
Write-Host "2. Check essential files: ls -la" -ForegroundColor White
Write-Host "3. Create GitHub repository: enhanced-rag-pipeline" -ForegroundColor White
Write-Host "4. Push to GitHub: git remote add github <url> && git push github main" -ForegroundColor White
Write-Host "5. Upload large data files separately to Hetzner server" -ForegroundColor White
Write-Host ""
Write-Host "🔐 Safety features:" -ForegroundColor Cyan
Write-Host "- Backup branch created: 'backup-before-cleanup'" -ForegroundColor White
Write-Host "- Local files preserved on your drive" -ForegroundColor White
Write-Host "- Only git history was cleaned" -ForegroundColor White
Write-Host ""
Write-Host "🎯 Repository ready for GitHub migration!" -ForegroundColor Green