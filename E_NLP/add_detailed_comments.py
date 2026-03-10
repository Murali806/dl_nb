#!/usr/bin/env python3
"""
Script to add detailed comments to Python cells in notebooks 02-09,
following the commenting style from 01_char_tokenization.ipynb
"""

import json
from pathlib import Path

def add_comments_to_get_stats(code):
    """Add detailed comments to get_stats function"""
    if 'def get_stats(ids):' in code and '# ┌' not in code:
        commented = '''# ┌─────────────────────────────────────────────────────────────────────┐
# │                    PAIR COUNTING LOGIC                              │
# │                                                                     │
# │  Input:  [1, 2, 3, 1, 2, 4, 1, 2]                                  │
# │                                                                     │
# │  zip(ids, ids[1:]) creates sliding window of adjacent pairs:       │
# │  ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┐         │
# │  │ (1,2) │ (2,3) │ (3,1) │ (1,2) │ (2,4) │ (4,1) │ (1,2) │         │
# │  └───────┴───────┴───────┴───────┴───────┴───────┴───────┘         │
# │                                                                     │
# │  Count each unique pair:                                            │
# │  {(1,2): 3, (2,3): 1, (3,1): 1, (2,4): 1, (4,1): 1}                │
# │                                                                     │
# │  This tells us which pairs appear most frequently in the sequence   │
# └─────────────────────────────────────────────────────────────────────┘

def get_stats(ids):
    """
    Count frequency of adjacent pairs in the sequence.
    
    Args:
        ids: List of token IDs
    
    Returns:
        Dictionary mapping pairs to their counts
    """
    counts = {}
    # zip(ids, ids[1:]) creates pairs: (ids[0],ids[1]), (ids[1],ids[2]), ...
    # This sliding window captures all adjacent token pairs in the sequence
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts'''
        return commented
    return code

def add_comments_to_merge(code):
    """Add detailed comments to merge function"""
    if 'def merge(ids, pair, idx):' in code and '# ┌' not in code:
        commented = '''# ┌─────────────────────────────────────────────────────────────────────┐
# │                    MERGE OPERATION LOGIC                            │
# │                                                                     │
# │  Input:  ids = [1, 2, 3, 1, 2, 4, 1, 2]                            │
# │          pair = (1, 2)                                              │
# │          idx = 99  (new token ID for merged pair)                  │
# │                                                                     │
# │  Process: Scan through ids, whenever we find (1,2) replace with 99 │
# │                                                                     │
# │  Step-by-step:                                                      │
# │  i=0: Found (1,2) → append 99, skip to i=2                         │
# │  i=2: Found 3 → append 3, i=3                                      │
# │  i=3: Found (1,2) → append 99, skip to i=5                         │
# │  i=5: Found 4 → append 4, i=6                                      │
# │  i=6: Found (1,2) → append 99, skip to i=8                         │
# │                                                                     │
# │  Output: [99, 3, 99, 4, 99]                                         │
# │                                                                     │
# │  Length: 8 → 5 (saved 3 tokens by merging 3 occurrences)           │
# └─────────────────────────────────────────────────────────────────────┘

def merge(ids, pair, idx):
    """
    Replace all occurrences of pair with new token idx.
    
    Args:
        ids: List of token IDs
        pair: Tuple of (token1, token2) to merge
        idx: New token ID for the merged pair
    
    Returns:
        New list with pairs replaced
    """
    newids = []
    i = 0
    while i < len(ids):
        # Check if current position and next position form the target pair
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            # Found the pair! Replace with new merged token
            newids.append(idx)
            i += 2  # Skip both tokens since we merged them
        else:
            # Not the pair we're looking for, keep the token as-is
            newids.append(ids[i])
            i += 1
    return newids'''
        return commented
    return code

def add_comments_to_encode_bpe(code):
    """Add detailed comments to encode_bpe function"""
    if 'def encode_bpe(text):' in code and '# ┌' not in code:
        commented = '''# ┌─────────────────────────────────────────────────────────────────────┐
# │                    BPE ENCODING PIPELINE                            │
# │                                                                     │
# │  Step 1: Convert text to bytes                                      │
# │  "Hello" → [72, 101, 108, 108, 111]                                │
# │                                                                     │
# │  Step 2: Apply learned merges iteratively                           │
# │  Start with byte-level tokens, then merge based on training         │
# │                                                                     │
# │  Example with merges: {(72,101): 256, (256,108): 257}              │
# │  [72, 101, 108, 108, 111]                                           │
# │    ↓ merge (72,101)→256                                             │
# │  [256, 108, 108, 111]                                               │
# │    ↓ merge (256,108)→257                                            │
# │  [257, 108, 111]                                                    │
# │    ↓ no more applicable merges                                      │
# │  Final: [257, 108, 111]                                             │
# │                                                                     │
# │  Key: We apply merges in the order they were learned (lowest idx)  │
# └─────────────────────────────────────────────────────────────────────┘

def encode_bpe(text):
    """
    Encode text using trained BPE tokenizer.
    
    Args:
        text: String to encode
    
    Returns:
        List of token IDs
    """
    # Start with bytes - convert text to UTF-8 byte representation
    # This ensures we can handle any Unicode text
    tokens = list(text.encode('utf-8'))
    
    # Apply merges iteratively until no more merges are possible
    while len(tokens) >= 2:
        # Get pair statistics for current token sequence
        stats = get_stats(tokens)
        
        # Find the pair with lowest merge index (earliest learned merge)
        # merges.get(p, float('inf')) returns merge index or infinity if not in merges
        # min() selects the pair that was learned earliest in training
        pair = min(stats, key=lambda p: merges.get(p, float('inf')))
        
        # If pair not in merges, we're done - no more applicable merges
        if pair not in merges:
            break
        
        # Apply the merge: replace all occurrences of pair with merged token
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    
    return tokens'''
        return commented
    return code

def add_comments_to_decode_bpe(code):
    """Add detailed comments to decode_bpe function"""
    if 'def decode_bpe(tokens):' in code and '# ┌' not in code:
        commented = '''# ┌─────────────────────────────────────────────────────────────────────┐
# │                    BPE DECODING PIPELINE                            │
# │                                                                     │
# │  Reverse the encoding process: token IDs → bytes → text            │
# │                                                                     │
# │  Example:                                                           │
# │  tokens = [257, 108, 111]                                           │
# │                                                                     │
# │  vocab[257] = b'Hel'  (learned merge)                              │
# │  vocab[108] = b'l'    (original byte)                              │
# │  vocab[111] = b'o'    (original byte)                              │
# │                                                                     │
# │  Step 1: Look up each token in vocab                               │
# │  [257, 108, 111] → [b'Hel', b'l', b'o']                            │
# │                                                                     │
# │  Step 2: Concatenate bytes                                          │
# │  b'Hel' + b'l' + b'o' = b'Hello'                                   │
# │                                                                     │
# │  Step 3: Decode bytes to UTF-8 string                              │
# │  b'Hello' → "Hello"                                                 │
# └─────────────────────────────────────────────────────────────────────┘

def decode_bpe(tokens):
    """
    Decode token IDs back to text.
    
    Args:
        tokens: List of token IDs
    
    Returns:
        Decoded string
    """
    # Look up each token ID in vocab to get its byte representation
    # vocab maps token IDs to their corresponding byte sequences
    text_bytes = b"".join(vocab[idx] for idx in tokens)
    
    # Decode bytes to UTF-8 string
    # errors='replace' handles any invalid UTF-8 sequences gracefully
    text = text_bytes.decode('utf-8', errors='replace')
    
    return text'''
        return commented
    return code

def add_comments_to_training_loop(code):
    """Add detailed comments to BPE training loop"""
    if 'for i in range(num_merges):' in code and '# ┌' not in code:
        # Add comment before the loop
        parts = code.split('# Training loop\n')
        if len(parts) == 2:
            before, after = parts
            commented = before + '''# ┌─────────────────────────────────────────────────────────────────────┐
# │                    BPE TRAINING ALGORITHM                           │
# │                                                                     │
# │  Iterative compression: merge most frequent pairs repeatedly        │
# │                                                                     │
# │  Iteration 0: tokens = [t, h, e, _, t, h, e, ...]                  │
# │               pairs = {(t,h): 1000, (h,e): 800, ...}               │
# │               most frequent = (t,h)                                 │
# │               create new token 256 for 'th'                         │
# │               merge all (t,h) → 256                                 │
# │                                                                     │
# │  Iteration 1: tokens = [256, e, _, 256, e, ...]                    │
# │               pairs = {(256,e): 800, ...}                           │
# │               most frequent = (256,e)                               │
# │               create new token 257 for 'the'                        │
# │               merge all (256,e) → 257                               │
# │                                                                     │
# │  Continue until we reach target vocabulary size                     │
# │                                                                     │
# │  Result: vocab grows from 256 (bytes) to target size               │
# │          common patterns become single tokens                       │
# └─────────────────────────────────────────────────────────────────────┘

# Training loop
''' + after
            return commented
    return code

def add_comments_to_bytes_conversion(code):
    """Add comments to bytes conversion"""
    if 'tokens = list(text.encode' in code and '# Convert text to bytes' in code and '# ┌' not in code:
        commented = '''# ┌─────────────────────────────────────────────────────────────────────┐
# │                    TEXT → BYTES CONVERSION                          │
# │                                                                     │
# │  Why bytes? BPE operates on bytes, not characters                   │
# │  This ensures it can handle ANY Unicode text                        │
# │                                                                     │
# │  Example:                                                           │
# │  "Hello" → UTF-8 encoding → [72, 101, 108, 108, 111]               │
# │  "世界"   → UTF-8 encoding → [228, 184, 150, 231, 149, 140]         │
# │                                                                     │
# │  Each byte is in range 0-255, giving us 256 base tokens            │
# │  BPE will learn to merge these bytes into larger units             │
# └─────────────────────────────────────────────────────────────────────┘

# Convert text to bytes
# .encode('utf-8') converts string to bytes, list() makes it a list of integers
tokens = list(text.encode('utf-8'))'''
        return commented
    return code

def process_notebook(notebook_path):
    """Process a single notebook and add detailed comments"""
    print(f"Processing {notebook_path.name}...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    modified = False
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Apply comment additions
            new_source = source
            new_source = add_comments_to_get_stats(new_source)
            new_source = add_comments_to_merge(new_source)
            new_source = add_comments_to_encode_bpe(new_source)
            new_source = add_comments_to_decode_bpe(new_source)
            new_source = add_comments_to_training_loop(new_source)
            new_source = add_comments_to_bytes_conversion(new_source)
            
            if new_source != source:
                # Split back into lines for JSON format
                cell['source'] = [line + '\n' for line in new_source.split('\n')[:-1]]
                if new_source.split('\n')[-1]:  # Add last line without \n if it exists
                    cell['source'].append(new_source.split('\n')[-1])
                modified = True
    
    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"  [OK] Updated {notebook_path.name}")
        return True
    else:
        print(f"  [INFO] No changes needed for {notebook_path.name}")
        return False

def main():
    """Process all notebooks from 02 to 09"""
    notebooks_dir = Path('notebooks')
    
    # Process notebooks 02-09
    notebooks = [
        '02_bpe_tokenization.ipynb',
        '03_data_pipeline.ipynb',
        '04_embeddings.ipynb',
        '05_self_attention.ipynb',
        '06_multi_head_attention.ipynb',
        '07_feed_forward.ipynb',
        '08_transformer_block.ipynb',
        '09_training_generation.ipynb'
    ]
    
    total_modified = 0
    for nb_name in notebooks:
        nb_path = notebooks_dir / nb_name
        if nb_path.exists():
            if process_notebook(nb_path):
                total_modified += 1
        else:
            print(f"  [WARNING] Not found: {nb_path}")
    
    print(f"\n{'='*60}")
    print(f"[COMPLETE] Modified {total_modified} notebook(s)")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
