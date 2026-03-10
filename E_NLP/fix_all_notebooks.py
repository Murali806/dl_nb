"""
Script to add Colab compatibility to all LLM notebooks.
Adds:
1. Environment detection (Colab vs Local)
2. Automatic directory creation (os.makedirs)
3. URL-based data download for Notebook 01
4. Environment-aware file paths
"""

import json
import os

def create_environment_setup_cell():
    """Create the environment detection and setup cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 🌐 Environment Setup (Colab/Local)\n",
            "\n",
            "This cell detects whether you're running in Google Colab or locally and sets up the environment accordingly."
        ]
    }, {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# ========================================\n",
            "# ENVIRONMENT DETECTION & SETUP\n",
            "# ========================================\n",
            "\n",
            "# Check if running in Colab\n",
            "try:\n",
            "    import google.colab\n",
            "    IN_COLAB = True\n",
            "    print(\"🌐 Running in Google Colab\")\n",
            "except:\n",
            "    IN_COLAB = False\n",
            "    print(\"💻 Running locally\")\n",
            "\n",
            "# Setup directories\n",
            "if IN_COLAB:\n",
            "    # Create necessary directories for Colab\n",
            "    os.makedirs('data', exist_ok=True)\n",
            "    os.makedirs('visualizations/tokenization', exist_ok=True)\n",
            "    os.makedirs('visualizations/embeddings', exist_ok=True)\n",
            "    os.makedirs('visualizations/data_pipeline', exist_ok=True)\n",
            "    print(\"✅ Created directories\")\n",
            "    \n",
            "    # Set paths for Colab\n",
            "    DATA_DIR = 'data'\n",
            "    VIZ_DIR = 'visualizations'\n",
            "else:\n",
            "    # Use relative paths for local execution\n",
            "    DATA_DIR = '../data'\n",
            "    VIZ_DIR = '../visualizations'\n",
            "    \n",
            "    # Create directories if they don't exist (LOCAL FIX)\n",
            "    os.makedirs(DATA_DIR, exist_ok=True)\n",
            "    os.makedirs(f'{VIZ_DIR}/tokenization', exist_ok=True)\n",
            "    os.makedirs(f'{VIZ_DIR}/embeddings', exist_ok=True)\n",
            "    os.makedirs(f'{VIZ_DIR}/data_pipeline', exist_ok=True)\n",
            "    print(f\"✅ Created directories: {DATA_DIR}, {VIZ_DIR}\")\n",
            "\n",
            "print(f\"\\n📁 Data directory: {DATA_DIR}\")\n",
            "print(f\"📊 Visualization directory: {VIZ_DIR}\")"
        ],
        "outputs": []
    }

def create_data_download_cell():
    """Create the data download cell for Notebook 01."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 📥 Get Sample Data\n",
            "\n",
            "This cell automatically downloads sample text data (Shakespeare) for training."
        ]
    }, {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# ========================================\n",
            "# DATA ACQUISITION\n",
            "# ========================================\n",
            "\n",
            "sample_path = f'{DATA_DIR}/sample.txt'\n",
            "\n",
            "# Check if data already exists\n",
            "if os.path.exists(sample_path):\n",
            "    print(f\"✅ Data file already exists: {sample_path}\")\n",
            "else:\n",
            "    if IN_COLAB:\n",
            "        # Download sample data automatically in Colab\n",
            "        import urllib.request\n",
            "        \n",
            "        print(\"📥 Downloading sample data (Shakespeare text)...\")\n",
            "        url = \"https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt\"\n",
            "        \n",
            "        try:\n",
            "            urllib.request.urlretrieve(url, sample_path)\n",
            "            print(f\"✅ Downloaded successfully to {sample_path}\")\n",
            "        except Exception as e:\n",
            "            print(f\"❌ Error downloading file: {e}\")\n",
            "            print(\"\\n💡 Alternative: You can manually upload a text file\")\n",
            "    else:\n",
            "        print(f\"❌ Error: {sample_path} not found!\")\n",
            "        print(f\"\\n💡 Please create the file at: {sample_path}\")\n",
            "        print(f\"   You can download it from:\")\n",
            "        print(f\"   https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt\")\n",
            "\n",
            "# Verify and show stats\n",
            "if os.path.exists(sample_path):\n",
            "    with open(sample_path, 'r', encoding='utf-8') as f:\n",
            "        text_preview = f.read()\n",
            "    \n",
            "    print(f\"\\n📊 File Statistics:\")\n",
            "    print(f\"   Total characters: {len(text_preview):,}\")\n",
            "    print(f\"   Total lines: {len(text_preview.splitlines()):,}\")\n",
            "    print(f\"\\n📝 First 200 characters:\")\n",
            "    print(f\"   {text_preview[:200]}...\")\n",
            "    print(f\"\\n✅ Data ready for processing!\")"
        ],
        "outputs": []
    }

print("✅ Script created!")
print("\n📝 This script provides helper functions to add Colab compatibility.")
print("   Run this in a Python environment to process notebooks.")
print("\n💡 For manual fixes, add the environment setup cell after imports in each notebook.")
