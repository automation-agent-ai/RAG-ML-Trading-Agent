# 🔧 Installing git-filter-repo (Recommended Safe Method)

## Why git-filter-repo?

`git-filter-repo` is the **official replacement** for `git filter-branch` and is **much safer**:
- ✅ **Officially recommended** by Git maintainers
- ✅ **Faster performance** (up to 10x faster)
- ✅ **Better safety checks** and error handling
- ✅ **More reliable** with fewer edge cases
- ✅ **Creates backups** automatically

## Installation Options

### Option 1: Install via pip (Recommended)
```powershell
# Install git-filter-repo via pip
pip install git-filter-repo

# Verify installation
git filter-repo --help
```

### Option 2: Manual Installation (Windows)
```powershell
# Download the script
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/newren/git-filter-repo/main/git-filter-repo" -OutFile "git-filter-repo"

# Make it executable and move to Git's bin directory
# Find your Git installation path first:
git --exec-path

# Then copy the script there (adjust path as needed):
# Copy-Item git-filter-repo "C:\Program Files\Git\libexec\git-core\"
```

### Option 3: Alternative - Use existing script
If you can't install git-filter-repo, you can use the original `cleanup_git_history.ps1` script, but:
- ⚠️ It's less safe than git-filter-repo
- ⚠️ May take longer to execute
- ⚠️ Has more potential for errors

## File Safety Clarification

**IMPORTANT**: Both methods only clean git history, not your local files:

### What gets removed:
- ❌ Files from **git history** and **git tracking**
- ❌ Files in **remote repositories** (after push)

### What stays safe:
- ✅ **All files on your local drive**
- ✅ **Working directory files** remain untouched
- ✅ **Data files** for separate Hetzner upload
- ✅ **Model files** (can be re-downloaded)

## Recommended Workflow

1. **Install git-filter-repo**:
   ```powershell
   pip install git-filter-repo
   ```

2. **Run the safe cleanup script**:
   ```powershell
   .\cleanup_git_history_safe.ps1
   ```

3. **Verify your local files are intact**:
   ```powershell
   # Check your data files are still there
   ls data/
   ls models/
   ```

4. **Push to GitHub**:
   ```powershell
   git remote add github https://github.com/USERNAME/enhanced-rag-pipeline.git
   git push github main
   ```

## Backup Strategy

The safe script automatically creates a backup branch:
```powershell
# If something goes wrong, restore from backup:
git checkout backup-before-cleanup
```

## Size Expectations

After cleanup, your repository should be:
- **Before**: ~2-3 GB (with large files)
- **After**: ~50-100 MB (code only)
- **Local files**: Still present on your drive for separate upload

This ensures your GitHub repository is clean and professional while preserving all your data for the Hetzner deployment.