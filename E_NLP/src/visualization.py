"""
Visualization utilities for LLM from Scratch project.

This module provides reusable plotting functions for visualizing
tokenization, embeddings, attention patterns, and data pipelines.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.animation import FuncAnimation
import torch


# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_token_distribution(tokens: List[int], vocab: Dict[int, str], 
                           top_n: int = 20, save_path: Optional[str] = None):
    """
    Plot frequency distribution of tokens.
    
    Args:
        tokens: List of token IDs
        vocab: Dictionary mapping token IDs to strings
        top_n: Number of top tokens to display
        save_path: Path to save the figure
    """
    from collections import Counter
    
    # Count token frequencies
    token_counts = Counter(tokens)
    top_tokens = token_counts.most_common(top_n)
    
    # Create labels
    labels = [vocab.get(tid, f"ID:{tid}") for tid, _ in top_tokens]
    counts = [count for _, count in top_tokens]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(labels)), counts, color=sns.color_palette("viridis", len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title(f'Top {top_n} Token Distribution', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count, i, f' {count}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_tokenization_comparison(text: str, char_tokens: List[int], 
                                 bpe_tokens: List[int], save_path: Optional[str] = None):
    """
    Compare character-level vs BPE tokenization.
    
    Args:
        text: Original text
        char_tokens: Character-level token IDs
        bpe_tokens: BPE token IDs
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Character-level
    axes[0].text(0.05, 0.5, f'Text: "{text[:100]}..."', 
                transform=axes[0].transAxes, fontsize=11, va='center')
    axes[0].text(0.05, 0.3, f'Char Tokens: {len(char_tokens)} tokens', 
                transform=axes[0].transAxes, fontsize=11, va='center', color='red')
    axes[0].text(0.05, 0.1, f'Sample: {char_tokens[:20]}...', 
                transform=axes[0].transAxes, fontsize=9, va='center', family='monospace')
    axes[0].set_title('Character-Level Tokenization', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    # BPE
    axes[1].text(0.05, 0.5, f'Text: "{text[:100]}..."', 
                transform=axes[1].transAxes, fontsize=11, va='center')
    axes[1].text(0.05, 0.3, f'BPE Tokens: {len(bpe_tokens)} tokens', 
                transform=axes[1].transAxes, fontsize=11, va='center', color='green')
    axes[1].text(0.05, 0.1, f'Sample: {bpe_tokens[:20]}...', 
                transform=axes[1].transAxes, fontsize=9, va='center', family='monospace')
    axes[1].set_title('Byte Pair Encoding (BPE)', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    
    # Comparison stats
    compression_ratio = len(char_tokens) / len(bpe_tokens)
    fig.text(0.5, 0.02, f'Compression Ratio: {compression_ratio:.2f}x (BPE is {compression_ratio:.2f}x more efficient)', 
             ha='center', fontsize=12, fontweight='bold', color='blue')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_sliding_window(sequence: List[int], block_size: int = 8, 
                       num_examples: int = 3, save_path: Optional[str] = None):
    """
    Visualize the sliding window mechanism for creating input-target pairs.
    
    Args:
        sequence: Token sequence
        block_size: Context window size
        num_examples: Number of examples to show
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(num_examples, 1, figsize=(14, 3 * num_examples))
    
    if num_examples == 1:
        axes = [axes]
    
    colors = sns.color_palette("Set2", block_size + 1)
    
    for idx, ax in enumerate(axes):
        start_idx = idx * 2  # Offset each example
        
        # Input sequence
        input_seq = sequence[start_idx:start_idx + block_size]
        target_seq = sequence[start_idx + 1:start_idx + block_size + 1]
        
        # Plot input
        for i, token in enumerate(input_seq):
            rect = FancyBboxPatch((i, 0.6), 0.8, 0.3, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=colors[i], 
                                 edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(i + 0.4, 0.75, str(token), ha='center', va='center', 
                   fontsize=12, fontweight='bold')
        
        # Plot target
        for i, token in enumerate(target_seq):
            rect = FancyBboxPatch((i, 0.2), 0.8, 0.3, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=colors[i + 1], 
                                 edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(i + 0.4, 0.35, str(token), ha='center', va='center', 
                   fontsize=12, fontweight='bold')
        
        # Arrows
        for i in range(len(input_seq)):
            ax.annotate('', xy=(i + 0.4, 0.5), xytext=(i + 0.4, 0.6),
                       arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
        
        # Labels
        ax.text(-0.5, 0.75, 'Input (X):', ha='right', va='center', 
               fontsize=11, fontweight='bold')
        ax.text(-0.5, 0.35, 'Target (Y):', ha='right', va='center', 
               fontsize=11, fontweight='bold')
        
        ax.set_xlim(-1, block_size)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f'Example {idx + 1}: Offset by 1 position', 
                    fontsize=12, fontweight='bold', loc='left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_batch_tensor_3d(batch: torch.Tensor, save_path: Optional[str] = None):
    """
    Visualize a 3D batch tensor (Batch × Sequence × Embedding).
    
    Args:
        batch: Tensor of shape (B, T, C)
        save_path: Path to save the figure
    """
    B, T, C = batch.shape
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid
    x = np.arange(T)
    y = np.arange(B)
    X, Y = np.meshgrid(x, y)
    
    # Plot each batch as a surface
    for b in range(B):
        Z = batch[b, :, :min(C, 10)].detach().numpy().T  # Show first 10 channels
        for c in range(Z.shape[0]):
            ax.plot(X[b], Y[b] * np.ones_like(X[b]), Z[c], alpha=0.6)
    
    ax.set_xlabel('Sequence Position (T)', fontsize=11)
    ax.set_ylabel('Batch Index (B)', fontsize=11)
    ax.set_zlabel('Embedding Values', fontsize=11)
    ax.set_title(f'Batch Tensor Visualization\nShape: ({B}, {T}, {C})', 
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_embedding_matrix(embedding_matrix: torch.Tensor, 
                         vocab_size: int = None, 
                         save_path: Optional[str] = None):
    """
    Visualize embedding matrix as a heatmap.
    
    Args:
        embedding_matrix: Embedding weight matrix (vocab_size, n_embd)
        vocab_size: Number of tokens to display (default: all)
        save_path: Path to save the figure
    """
    matrix = embedding_matrix.detach().numpy()
    
    if vocab_size:
        matrix = matrix[:vocab_size, :]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', 
                   vmin=-np.abs(matrix).max(), vmax=np.abs(matrix).max())
    
    ax.set_xlabel('Embedding Dimension', fontsize=12)
    ax.set_ylabel('Token ID', fontsize=12)
    ax.set_title(f'Token Embedding Matrix\nShape: {matrix.shape}', 
                fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Embedding Value')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_positional_encoding(pos_encoding: torch.Tensor, 
                            save_path: Optional[str] = None):
    """
    Visualize positional encoding patterns.
    
    Args:
        pos_encoding: Positional encoding tensor (seq_len, n_embd)
        save_path: Path to save the figure
    """
    encoding = pos_encoding.detach().numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Heatmap
    im = axes[0].imshow(encoding.T, aspect='auto', cmap='RdBu_r',
                       vmin=-1, vmax=1)
    axes[0].set_xlabel('Position', fontsize=12)
    axes[0].set_ylabel('Embedding Dimension', fontsize=12)
    axes[0].set_title('Positional Encoding Heatmap', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=axes[0], label='Encoding Value')
    
    # Line plot for selected dimensions
    dims_to_plot = [0, 1, 2, 3, 10, 20, 50, 100]
    dims_to_plot = [d for d in dims_to_plot if d < encoding.shape[1]]
    
    for dim in dims_to_plot:
        axes[1].plot(encoding[:, dim], label=f'Dim {dim}', alpha=0.7)
    
    axes[1].set_xlabel('Position', fontsize=12)
    axes[1].set_ylabel('Encoding Value', fontsize=12)
    axes[1].set_title('Positional Encoding Patterns (Selected Dimensions)', 
                     fontsize=13, fontweight='bold')
    axes[1].legend(loc='upper right', ncol=2)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_embedding_addition(token_emb: torch.Tensor, 
                           pos_emb: torch.Tensor,
                           final_emb: torch.Tensor,
                           position: int = 0,
                           save_path: Optional[str] = None):
    """
    Visualize token + positional embedding addition.
    
    Args:
        token_emb: Token embedding vector
        pos_emb: Positional embedding vector
        final_emb: Final embedding (token + pos)
        position: Position index to visualize
        save_path: Path to save the figure
    """
    tok = token_emb[position].detach().numpy()
    pos = pos_emb[position].detach().numpy()
    final = final_emb[position].detach().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Token embedding
    axes[0].bar(range(len(tok)), tok, color='steelblue', alpha=0.7)
    axes[0].set_title('Token Embedding\n"What is this word?"', 
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Value')
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Positional embedding
    axes[1].bar(range(len(pos)), pos, color='coral', alpha=0.7)
    axes[1].set_title('Positional Embedding\n"Where is this word?"', 
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Dimension')
    axes[1].set_ylabel('Value')
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Final embedding
    axes[2].bar(range(len(final)), final, color='green', alpha=0.7)
    axes[2].set_title('Final Embedding\n"Content + Location"', 
                     fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Dimension')
    axes[2].set_ylabel('Value')
    axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    fig.suptitle(f'Embedding Addition at Position {position}', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_vocab_growth(vocab_sizes: List[int], iterations: List[int],
                     save_path: Optional[str] = None):
    """
    Plot vocabulary growth during BPE training.
    
    Args:
        vocab_sizes: List of vocabulary sizes
        iterations: List of iteration numbers
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(iterations, vocab_sizes, marker='o', linewidth=2, 
           markersize=6, color='purple')
    ax.fill_between(iterations, vocab_sizes, alpha=0.3, color='purple')
    
    ax.set_xlabel('BPE Merge Iteration', fontsize=12)
    ax.set_ylabel('Vocabulary Size', fontsize=12)
    ax.set_title('Vocabulary Growth During BPE Training', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.annotate(f'Start: {vocab_sizes[0]}', 
               xy=(iterations[0], vocab_sizes[0]),
               xytext=(10, 20), textcoords='offset points',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    ax.annotate(f'End: {vocab_sizes[-1]}', 
               xy=(iterations[-1], vocab_sizes[-1]),
               xytext=(-60, -30), textcoords='offset points',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_pipeline_diagram(save_path: Optional[str] = None):
    """
    Create a visual diagram of the complete preprocessing pipeline.
    
    Args:
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    stages = [
        "Raw Text",
        "Tokenization",
        "Token IDs",
        "Batching",
        "Token Embeddings",
        "Positional Embeddings",
        "Final Embeddings"
    ]
    
    colors = sns.color_palette("husl", len(stages))
    
    for i, (stage, color) in enumerate(zip(stages, colors)):
        y = 1 - (i * 0.15)
        
        # Box
        rect = FancyBboxPatch((0.2, y - 0.05), 0.6, 0.08,
                             boxstyle="round,pad=0.01",
                             facecolor=color, edgecolor='black', 
                             linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        
        # Text
        ax.text(0.5, y, stage, ha='center', va='center',
               fontsize=13, fontweight='bold')
        
        # Arrow
        if i < len(stages) - 1:
            ax.annotate('', xy=(0.5, y - 0.13), xytext=(0.5, y - 0.05),
                       arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.axis('off')
    ax.set_title('LLM Preprocessing Pipeline', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# Interactive Plotly visualizations

def plot_embedding_interactive(embeddings: np.ndarray, labels: List[str] = None):
    """
    Create interactive 2D embedding visualization using Plotly.
    
    Args:
        embeddings: 2D array of embeddings (n_samples, 2)
        labels: List of labels for each point
    """
    fig = go.Figure(data=go.Scatter(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        mode='markers+text',
        text=labels if labels else None,
        textposition="top center",
        marker=dict(
            size=10,
            color=np.arange(len(embeddings)),
            colorscale='Viridis',
            showscale=True
        )
    ))
    
    fig.update_layout(
        title='Token Embeddings (2D Projection)',
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        hovermode='closest',
        width=900,
        height=700
    )
    
    fig.show()


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
    print("Available functions:")
    print("  - plot_token_distribution()")
    print("  - plot_tokenization_comparison()")
    print("  - plot_sliding_window()")
    print("  - plot_batch_tensor_3d()")
    print("  - plot_embedding_matrix()")
    print("  - plot_positional_encoding()")
    print("  - plot_embedding_addition()")
    print("  - plot_vocab_growth()")
    print("  - create_pipeline_diagram()")
    print("  - plot_embedding_interactive()")
