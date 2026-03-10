#!/usr/bin/env python3
"""
Automatically add Colab compatibility to notebooks 02-09.
This script adds environment detection and updates file paths.
"""

import json
import os
import sys

# Environment setup cell (markdown)
ENV_SETUP_MARKDOWN = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Environment Setup (Colab/Local)\n",
        "\n",
        "This cell detects whether you're running in Google Colab or locally and sets up the environment accordingly."
    ]
}

# Environment setup cell (code)
ENV_SETUP_CODE = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ========================================\n",
        "# ENVIRONMENT DETECTION & SETUP\n",
        "# ========================================\n",
        "\n",
        "# Check if running in Colab\n",
        "try:\n",
        "    import google.colab\n",
        "    IN_COLAB = True\n",
        "    print(\"Running in Google Colab\")\n",
        "except:\n",
        "    IN_COLAB = False\n",
        "    print(\"Running locally\")\n",
        "\n",
        "# Setup directories\n",
        "if IN_COLAB:\n",
        "    # Create necessary directories for Colab\n",
        "    os.makedirs('data', exist_ok=True)\n",
        "    os.makedirs('visualizations/tokenization', exist_ok=True)\n",
        "    os.makedirs('visualizations/embeddings', exist_ok=True)\n",
        "    os.makedirs('visualizations/data_pipeline', exist_ok=True)\n",
        "    print(\"Created directories\")\n",
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
        "    print(f\"Created directories: {DATA_DIR}, {VIZ_DIR}\")\n",
        "\n",
        "print(f\"\\nData directory: {DATA_DIR}\")\n",
        "print(f\"Visualization directory: {VIZ_DIR}\")"
    ]
}

def update_file_paths(source_lines):
    """Update file paths in source code to use environment variables."""
    updated_lines = []
    for line in source_lines:
        # Replace data paths
        if "'../data/" in line:
            line = line.replace("'../data/", "f'{DATA_DIR}/")
        elif '"../data/' in line:
            line = line.replace('"../data/', 'f"{DATA_DIR}/')
        
        # Replace visualization paths
        if "'../visualizations/" in line:
            line = line.replace("'../visualizations/", "f'{VIZ_DIR}/")
        elif '"../visualizations/' in line:
            line = line.replace('"../visualizations/', 'f"{VIZ_DIR}/')
        
        updated_lines.append(line)
    return updated_lines

def fix_notebook(notebook_path):
    """Add environment setup and update paths in a notebook."""
    print(f"\nProcessing: {notebook_path}")
    
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the first code cell (imports)
    first_code_idx = None
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            first_code_idx = i
            break
    
    if first_code_idx is None:
        print("  [ERROR] No code cells found!")
        return False
    
    # Check if environment setup already exists
    if first_code_idx + 1 < len(notebook['cells']):
        next_cell = notebook['cells'][first_code_idx + 1]
        if 'ENVIRONMENT DETECTION' in ''.join(next_cell.get('source', [])):
            print("  [SKIP] Environment setup already exists")
            env_setup_exists = True
        else:
            env_setup_exists = False
    else:
        env_setup_exists = False
    
    # Insert environment setup cells if they don't exist
    if not env_setup_exists:
        notebook['cells'].insert(first_code_idx + 1, ENV_SETUP_MARKDOWN)
        notebook['cells'].insert(first_code_idx + 2, ENV_SETUP_CODE)
        print("  [OK] Added environment setup cells")
    
    # Update file paths in all code cells
    paths_updated = 0
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and cell.get('source'):
            original_source = cell['source']
            updated_source = update_file_paths(original_source)
            if original_source != updated_source:
                cell['source'] = updated_source
                paths_updated += 1
    
    print(f"  [OK] Updated {paths_updated} cells with new file paths")
    
    # Save notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"  [OK] Saved: {notebook_path}")
    return True

def main():
    """Main function to fix all notebooks."""
    notebooks_dir = 'E_NLP/notebooks'
    
    # List of notebooks to fix
    notebooks_to_fix = [
        '02_bpe_tokenization.ipynb',
        '03_data_pipeline.ipynb',
        '04_embeddings.ipynb',
        '05_self_attention.ipynb',
        '06_multi_head_attention.ipynb',
        '07_feed_forward.ipynb',
        '08_transformer_block.ipynb',
        '09_training_generation.ipynb'
    ]
    
    print("=" * 80)
    print("AUTO-FIXING NOTEBOOKS 02-09")
    print("=" * 80)
    
    success_count = 0
    for notebook_name in notebooks_to_fix:
        notebook_path = os.path.join(notebooks_dir, notebook_name)
        
        if not os.path.exists(notebook_path):
            print(f"\n[ERROR] Not found: {notebook_path}")
            continue
        
        try:
            if fix_notebook(notebook_path):
                success_count += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
    
    print("\n" + "=" * 80)
    print(f"Successfully fixed {success_count}/{len(notebooks_to_fix)} notebooks!")
    print("\nSummary:")
    print("  - Added environment detection cells")
    print("  - Updated file paths to use DATA_DIR and VIZ_DIR")
    print("  - Created directory creation code for local execution")
    print("\nAll notebooks are now Colab-compatible!")
    print("=" * 80)

if __name__ == '__main__':
    main()
