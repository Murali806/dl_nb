# 🔧 Colab Compatibility Fix Guide

## Quick Summary

This guide shows how to add Colab compatibility to all notebooks with:
1. ✅ **Automatic directory creation** (fixes `../data/` not found error)
2. ✅ **URL-based data download** (no manual uploads needed)
3. ✅ **Environment detection** (works in both Colab and Local)

---

## 🎯 The Fix

### **Add These 2 Cells After the Import Cell in Each Notebook**

#### **Cell 1: Environment Setup** (Add to ALL notebooks 01-09)

```python
# ========================================
# ENVIRONMENT DETECTION & SETUP
# ========================================

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
    print("🌐 Running in Google Colab")
except:
    IN_COLAB = False
    print("💻 Running locally")

# Setup directories
if IN_COLAB:
    # Create necessary directories for Colab
    os.makedirs('data', exist_ok=True)
    os.makedirs('visualizations/tokenization', exist_ok=True)
    os.makedirs('visualizations/embeddings', exist_ok=True)
    os.makedirs('visualizations/data_pipeline', exist_ok=True)
    print("✅ Created directories")
    
    # Set paths for Colab
    DATA_DIR = 'data'
    VIZ_DIR = 'visualizations'
else:
    # Use relative paths for local execution
    DATA_DIR = '../data'
    VIZ_DIR = '../visualizations'
    
    # Create directories if they don't exist (LOCAL FIX)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(f'{VIZ_DIR}/tokenization', exist_ok=True)
    os.makedirs(f'{VIZ_DIR}/embeddings', exist_ok=True)
    os.makedirs(f'{VIZ_DIR}/data_pipeline', exist_ok=True)
    print(f"✅ Created directories: {DATA_DIR}, {VIZ_DIR}")

print(f"\n📁 Data directory: {DATA_DIR}")
print(f"📊 Visualization directory: {VIZ_DIR}")
```

#### **Cell 2: Data Download** (Add ONLY to Notebook 01)

```python
# ========================================
# DATA ACQUISITION
# ========================================

sample_path = f'{DATA_DIR}/sample.txt'

# Check if data already exists
if os.path.exists(sample_path):
    print(f"✅ Data file already exists: {sample_path}")
else:
    if IN_COLAB:
        # Download sample data automatically in Colab
        import urllib.request
        
        print("📥 Downloading sample data (Shakespeare text)...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        
        try:
            urllib.request.urlretrieve(url, sample_path)
            print(f"✅ Downloaded successfully to {sample_path}")
        except Exception as e:
            print(f"❌ Error downloading file: {e}")
            print("\n💡 Alternative: You can manually upload a text file")
    else:
        print(f"❌ Error: {sample_path} not found!")
        print(f"\n💡 Please create the file at: {sample_path}")
        print(f"   You can download it from:")
        print(f"   https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

# Verify and show stats
if os.path.exists(sample_path):
    with open(sample_path, 'r', encoding='utf-8') as f:
        text_preview = f.read()
    
    print(f"\n📊 File Statistics:")
    print(f"   Total characters: {len(text_preview):,}")
    print(f"   Total lines: {len(text_preview.splitlines()):,}")
    print(f"\n📝 First 200 characters:")
    print(f"   {text_preview[:200]}...")
    print(f"\n✅ Data ready for processing!")
```

---

## 📝 Step-by-Step Instructions

### **For Each Notebook (01-09):**

1. **Open the notebook**
2. **Find the first code cell** (usually imports)
3. **Add a new cell after imports**
4. **Copy Cell 1 (Environment Setup)** from above
5. **For Notebook 01 only**: Add Cell 2 (Data Download)
6. **Update file paths** (see below)
7. **Save the notebook**

---

## 🔄 Update File Paths

### **Find and Replace in Each Notebook:**

#### **Data Files:**
```python
# OLD:
with open('../data/sample.txt', 'r') as f:
with open('../data/char_tokenizer.pkl', 'wb') as f:

# NEW:
with open(f'{DATA_DIR}/sample.txt', 'r') as f:
with open(f'{DATA_DIR}/char_tokenizer.pkl', 'wb') as f:
```

#### **Visualization Files:**
```python
# OLD:
plt.savefig('../visualizations/tokenization/plot.png')
plt.savefig('../visualizations/embeddings/plot.png')

# NEW:
plt.savefig(f'{VIZ_DIR}/tokenization/plot.png')
plt.savefig(f'{VIZ_DIR}/embeddings/plot.png')
```

---

## 📊 Notebook-Specific Notes

### **Notebook 01** (Character Tokenization)
- ✅ Add Cell 1 (Environment Setup)
- ✅ Add Cell 2 (Data Download)
- ✅ Update all `'../data/'` → `f'{DATA_DIR}/'`
- ✅ Update all `'../visualizations/'` → `f'{VIZ_DIR}/'`

### **Notebooks 02-09** (All Others)
- ✅ Add Cell 1 (Environment Setup) only
- ✅ Update all file paths to use variables
- ✅ Data files created by previous notebooks will be found automatically

---

## ✅ What This Fixes

### **Problem 1: Directory Not Found (Local)**
**Before:**
```
FileNotFoundError: [Errno 2] No such file or directory: '../data/sample.txt'
```

**After:**
```
✅ Created directories: ../data, ../visualizations
📁 Data directory: ../data
```

### **Problem 2: Manual File Upload (Colab)**
**Before:**
- Had to click upload button
- Button didn't work in VS Code

**After:**
```
📥 Downloading sample data (Shakespeare text)...
✅ Downloaded successfully to data/sample.txt
```

---

## 🚀 Quick Test

After making changes, test by running:

```python
# This should work in both Colab and Local
print(f"Data dir: {DATA_DIR}")
print(f"Exists: {os.path.exists(DATA_DIR)}")

# This should work
with open(f'{DATA_DIR}/sample.txt', 'r') as f:
    text = f.read()
print(f"Loaded {len(text)} characters")
```

---

## 📋 Checklist

- [ ] Added Environment Setup cell to Notebook 01
- [ ] Added Data Download cell to Notebook 01
- [ ] Updated file paths in Notebook 01
- [ ] Added Environment Setup cell to Notebooks 02-09
- [ ] Updated file paths in Notebooks 02-09
- [ ] Tested in local environment
- [ ] Tested in Google Colab

---

## 💡 Tips

1. **Copy-paste carefully** - Make sure to get the entire cell content
2. **Test incrementally** - Fix one notebook, test it, then move to the next
3. **Keep backups** - Git commit before making changes
4. **Check paths** - Make sure all `'../data/'` are replaced with `f'{DATA_DIR}/'`

---

## 🆘 Troubleshooting

### **Issue: Still getting "directory not found"**
**Solution:** Make sure you added the `os.makedirs()` lines in the LOCAL section

### **Issue: Data not downloading in Colab**
**Solution:** Check your internet connection and the URL is accessible

### **Issue: Visualizations not saving**
**Solution:** Make sure you updated `plt.savefig()` to use `f'{VIZ_DIR}/...'`

---

## ✨ Result

After applying these fixes:
- ✅ **Works in Colab** - Automatic data download, no uploads needed
- ✅ **Works Locally** - Directories created automatically
- ✅ **No errors** - All file paths work correctly
- ✅ **Easy to use** - Just run the cells!

Happy coding! 🎉
