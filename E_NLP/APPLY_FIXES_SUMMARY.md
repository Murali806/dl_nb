# 🔧 Apply Fixes to Notebooks 02-09 - Summary

## ✅ What Needs to Be Done

You need to **manually add 2 cells** to each notebook (02-09) after the imports cell.

---

## 📋 Step-by-Step Instructions

### **For Each Notebook (02-09):**

1. Open the notebook in Jupyter/VS Code
2. Find the **first code cell** (imports)
3. **Insert a new cell** below it
4. **Copy-paste Cell 1** (Environment Setup) - see below
5. **Update all file paths** - Replace `'../data/'` with `f'{DATA_DIR}/'` and `'../visualizations/'` with `f'{VIZ_DIR}/'`
6. Save the notebook

---

## 📝 Cell to Add (After Imports)

### **Cell 1: Environment Setup** 

Add this markdown cell first:
```markdown
### 🌐 Environment Setup (Colab/Local)

This cell detects whether you're running in Google Colab or locally and sets up the environment accordingly.
```

Then add this code cell:
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

---

## 🔄 File Path Updates

### **Find and Replace in Each Notebook:**

#### **Notebook 02** (BPE Tokenization):
```python
# OLD:
with open('../data/sample.txt', 'r', encoding='utf-8') as f:
with open('../data/bpe_tokenizer.pkl', 'wb') as f:
plt.savefig('../visualizations/tokenization/bpe_vocab_growth.png'
plt.savefig('../visualizations/tokenization/bpe_comparison.png'
plt.savefig('../visualizations/tokenization/token_length_dist.png'

# NEW:
with open(f'{DATA_DIR}/sample.txt', 'r', encoding='utf-8') as f:
with open(f'{DATA_DIR}/bpe_tokenizer.pkl', 'wb') as f:
plt.savefig(f'{VIZ_DIR}/tokenization/bpe_vocab_growth.png'
plt.savefig(f'{VIZ_DIR}/tokenization/bpe_comparison.png'
plt.savefig(f'{VIZ_DIR}/tokenization/token_length_dist.png'
```

#### **Notebook 03** (Data Pipeline):
```python
# OLD:
with open('../data/sample.txt', 'r') as f:
with open('../data/char_tokenizer.pkl', 'rb') as f:
with open('../data/data_config.pkl', 'wb') as f:
plt.savefig('../visualizations/data_pipeline/...'

# NEW:
with open(f'{DATA_DIR}/sample.txt', 'r') as f:
with open(f'{DATA_DIR}/char_tokenizer.pkl', 'rb') as f:
with open(f'{DATA_DIR}/data_config.pkl', 'wb') as f:
plt.savefig(f'{VIZ_DIR}/data_pipeline/...'
```

#### **Notebooks 04-09** (Similar pattern):
Replace ALL occurrences of:
- `'../data/'` → `f'{DATA_DIR}/'`
- `'../visualizations/'` → `f'{VIZ_DIR}/'`

---

## 📊 Notebooks to Fix

- [ ] **Notebook 02**: BPE Tokenization
- [ ] **Notebook 03**: Data Pipeline
- [ ] **Notebook 04**: Embeddings
- [ ] **Notebook 05**: Self-Attention
- [ ] **Notebook 06**: Multi-Head Attention
- [ ] **Notebook 07**: Feed Forward
- [ ] **Notebook 08**: Transformer Block
- [ ] **Notebook 09**: Training & Generation

---

## 🎯 Quick Test After Each Fix

Run this in a cell to verify:
```python
# Test environment setup
print(f"IN_COLAB: {IN_COLAB}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"VIZ_DIR: {VIZ_DIR}")
print(f"Data dir exists: {os.path.exists(DATA_DIR)}")
```

---

## 💡 Why Manual Editing?

Jupyter notebooks are JSON files with complex structure. Manual editing in Jupyter/VS Code is:
- ✅ **Safer** - Visual interface prevents JSON corruption
- ✅ **Easier** - Can see the changes immediately
- ✅ **Faster** - Copy-paste is quicker than programmatic edits

---

## 🆘 If You Get Stuck

1. **Open** `E_NLP/COLAB_FIX_GUIDE.md` for detailed instructions
2. **Check** the example in Notebook 01 (already has the fix)
3. **Test** each notebook after fixing to ensure it works

---

## ✨ Expected Result

After fixing all notebooks:
- ✅ **No "directory not found" errors** locally
- ✅ **Works in both Colab and Local** environments
- ✅ **All file operations succeed**
- ✅ **Visualizations save correctly**

Good luck! 🚀
