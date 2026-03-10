# 🌐 Google Colab Setup Guide

## Running LLM Notebooks in Google Colab

This guide explains how to run the "Building LLMs from Scratch" notebooks in Google Colab.

---

## 📋 Quick Start

### Step 1: Upload to Google Drive (Recommended)

1. Upload the entire `E_NLP` folder to your Google Drive
2. Open Google Colab: https://colab.research.google.com/
3. Mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Navigate to your notebook:

```python
import os
os.chdir('/content/drive/MyDrive/E_NLP/notebooks')
```

5. Open any notebook and run!

---

## 🔧 Alternative: File Upload Method

If you prefer not to use Google Drive, add these cells at the beginning of each notebook:

### Cell 1: Environment Detection

```python
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
    import os
    os.makedirs('data', exist_ok=True)
    os.makedirs('visualizations/tokenization', exist_ok=True)
    os.makedirs('visualizations/embeddings', exist_ok=True)
    os.makedirs('visualizations/data_pipeline', exist_ok=True)
    
    DATA_DIR = 'data'
    VIZ_DIR = 'visualizations'
else:
    DATA_DIR = '../data'
    VIZ_DIR = '../visualizations'

print(f"📁 Data directory: {DATA_DIR}")
print(f"📊 Visualization directory: {VIZ_DIR}")
```

### Cell 2: Get Sample Data (For Notebook 01)

**Option A: Download from URL (Easiest)**
```python
if IN_COLAB:
    import urllib.request
    
    print("📥 Downloading sample data...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, f'{DATA_DIR}/sample.txt')
    
    # Verify
    with open(f'{DATA_DIR}/sample.txt', 'r') as f:
        text = f.read()
    print(f"✅ Downloaded {len(text):,} characters")
    print(f"📝 First 200 chars: {text[:200]}...")
```

**Option B: Upload from Local (In Colab UI)**
```python
if IN_COLAB:
    from google.colab import files
    
    print("📁 Click 'Choose Files' button below to upload sample.txt:")
    uploaded = files.upload()
    
    for filename in uploaded.keys():
        if filename.endswith('.txt'):
            os.rename(filename, f'{DATA_DIR}/sample.txt')
            print(f"✅ Uploaded: {filename} → {DATA_DIR}/sample.txt")
            break
```

**Option C: Use Google Drive (Best for large files)**
```python
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Copy from your Drive
    import shutil
    shutil.copy('/content/drive/MyDrive/sample.txt', f'{DATA_DIR}/sample.txt')
    print("✅ Copied from Google Drive")
```

### Cell 3: Update File Paths

Replace all file paths in the notebooks:

**Before:**
```python
with open('../data/sample.txt', 'r') as f:
    text = f.read()
```

**After:**
```python
with open(f'{DATA_DIR}/sample.txt', 'r') as f:
    text = f.read()
```

**Before:**
```python
plt.savefig('../visualizations/tokenization/plot.png')
```

**After:**
```python
plt.savefig(f'{VIZ_DIR}/tokenization/plot.png')
```

---

## 📊 Sample Data

If you don't have sample.txt, you can download Shakespeare's text:

```python
import urllib.request

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, f'{DATA_DIR}/sample.txt')

print("✅ Sample data downloaded!")
```

---

## 🎯 Notebook-Specific Instructions

### Notebook 01: Character Tokenization
- **Upload**: `sample.txt` (any text file)
- **Output**: Character mappings, visualizations

### Notebooks 02-09: Sequential Execution
- **Important**: Run notebooks in order (01 → 02 → 03 → ...)
- Each notebook depends on data from previous notebooks
- Files are saved to `data/` directory automatically

---

## 💾 Downloading Results

### Download Generated Files

```python
from google.colab import files

# Download trained model
files.download('data/trained_model.pth')

# Download visualizations
import shutil
shutil.make_archive('visualizations', 'zip', 'visualizations')
files.download('visualizations.zip')
```

### Download All Data

```python
# Create archive of all data
shutil.make_archive('llm_project_data', 'zip', 'data')
files.download('llm_project_data.zip')
```

---

## ⚡ GPU Acceleration

Enable GPU for faster training (Notebook 09):

1. Go to **Runtime** → **Change runtime type**
2. Select **GPU** under Hardware accelerator
3. Click **Save**

Verify GPU is available:

```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

---

## 🔄 Session Persistence

**Important**: Colab sessions timeout after ~12 hours of inactivity.

### Save Your Work Frequently

```python
# Save important data
torch.save(model.state_dict(), f'{DATA_DIR}/checkpoint.pth')

# Save to Google Drive (if mounted)
import shutil
shutil.copytree('data', '/content/drive/MyDrive/E_NLP_backup/data')
```

### Resume from Checkpoint

```python
# Load saved model
checkpoint = torch.load(f'{DATA_DIR}/checkpoint.pth')
model.load_state_dict(checkpoint)
```

---

## 🐛 Troubleshooting

### Issue: "File not found"

**Solution**: Make sure you've run all previous notebooks in order.

```python
# Check if required files exist
import os
required_files = [
    'data/sample.txt',
    'data/char_tokenizer.pkl',
    'data/data_config.pkl'
]

for file in required_files:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} - Missing!")
```

### Issue: "Out of memory"

**Solution**: Reduce batch size or model size.

```python
# In training notebooks, reduce these:
batch_size = 16  # Instead of 32
n_layer = 4      # Instead of 6
```

### Issue: "Session disconnected"

**Solution**: 
1. Reconnect to runtime
2. Re-run setup cells
3. Load checkpoints if available

---

## 📝 Complete Example: Notebook 01 in Colab

Here's a complete modified version of the first cell in Notebook 01:

```python
# ========================================
# COLAB SETUP
# ========================================

# Check environment
try:
    import google.colab
    IN_COLAB = True
    print("🌐 Running in Google Colab")
except:
    IN_COLAB = False
    print("💻 Running locally")

# Setup
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import torch
import pickle

if IN_COLAB:
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('visualizations/tokenization', exist_ok=True)
    
    # Set paths
    DATA_DIR = 'data'
    VIZ_DIR = 'visualizations/tokenization'
    
    # Upload file
    from google.colab import files
    print("\n📁 Upload your sample.txt file:")
    uploaded = files.upload()
    
    for filename in uploaded.keys():
        if filename.endswith('.txt'):
            os.rename(filename, f'{DATA_DIR}/sample.txt')
            print(f"✅ File ready: {DATA_DIR}/sample.txt")
            break
else:
    DATA_DIR = '../data'
    VIZ_DIR = '../visualizations/tokenization'

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print(f"\n✅ Setup complete!")
print(f"   Data: {DATA_DIR}")
print(f"   Visualizations: {VIZ_DIR}")
```

---

## 🎓 Best Practices

1. **Run notebooks sequentially** (01 → 09)
2. **Save checkpoints** frequently during training
3. **Download results** before session expires
4. **Use GPU** for training (Notebook 09)
5. **Monitor memory** usage in long-running cells

---

## 📚 Additional Resources

- **Colab Documentation**: https://colab.research.google.com/
- **PyTorch on Colab**: https://pytorch.org/tutorials/beginner/colab.html
- **Sample Data**: https://github.com/karpathy/char-rnn/tree/master/data

---

## ✅ Checklist

Before starting:
- [ ] Have sample.txt ready (or know how to download it)
- [ ] Understand file paths will be different in Colab
- [ ] Know how to enable GPU if needed
- [ ] Plan to save work frequently

Happy learning! 🚀
