# SGN Architecture: Feature Evolution Visual Guide

## Purpose
This document provides intuitive explanations of **how features evolve** through each stage of the SGN network, **why each transformation helps**, and **what the network learns** for noise suppression.

---

## Table of Contents
1. [Input Stage: Raw Audio to Spectrograms](#1-input-stage)
2. [Rotation Layer: Spatial Feature Enhancement](#2-rotation-layer)
3. [Concatenation: Adding Echo Context](#3-concatenation)
4. [BiLSTM: Temporal Pattern Learning](#4-bilstm)
5. [Dual LSTM Branches: Task Specialization](#5-dual-lstm-branches)
6. [FC Layers: Feature Refinement](#6-fc-layers)
7. [Filter Block: Adaptive Masking](#7-filter-block)
8. [Complete Feature Journey](#8-complete-feature-journey)

---

## 1. Input Stage: Raw Audio to Spectrograms

### What Happens
```
Time Domain          →    Frequency Domain
[Amplitude vs Time]  →    [Frequency vs Time]

Mic 1: ~~~∿∿~~~      →    [Spectrogram 1]
Mic 2: ~~~∿∿~~~      →    [Spectrogram 2]
Ref:   ~~~∿∿~~~      →    [Spectrogram Ref]
```

### Visual Representation

**Time Domain (1 second of audio):**
```
Amplitude
   ↑
   |    Speech + Noise
   |   /\  /\/\  /\
   |  /  \/    \/  \
   |_________________→ Time
   0ms            1000ms
```

**Frequency Domain (Spectrogram):**
```
Frequency (Hz)
   ↑
8000|  ░░░░░░░░░░░░░░  ← High freq noise
4000|  ██░░██░░██░░░░  ← Speech formants
2000|  ████████░░░░░░  ← Fundamental freq
1000|  ████████████░░  ← Low freq energy
   0|__________________→ Time
     0ms  200ms  400ms
     
Legend: █ = High energy, ░ = Low energy
```

### Why This Helps

**Frequency Domain Advantages:**
1. **Separability**: Speech and noise occupy different frequency regions
2. **Stationarity**: Noise is more stationary in frequency domain
3. **Perceptual**: Human hearing is frequency-based
4. **Efficiency**: Can process all frequencies in parallel

**What Features Look Like:**
- **Clean Speech**: Strong harmonics at fundamental frequency + formants
- **Noise**: Broadband energy across all frequencies
- **Echo**: Delayed copy of reference signal

---

## 2. Rotation Layer: Spatial Feature Enhancement

### What Happens

**Input: 2 Microphones × 320 Features**
```
Mic 1: [f₁, f₂, f₃, ..., f₃₂₀]  ← 320 frequency features
Mic 2: [f₁, f₂, f₃, ..., f₃₂₀]  ← 320 frequency features
```

**Rotation Transformation:**
```
         [W_rot]
           ↓
    Linear Mixing
           ↓
8 Channels × 640 Features
```

### Visual Representation

**Before Rotation (2D Spatial View):**
```
Mic 2 Signal
   ↑
   |     Noise
   |      ●●●
   |    ●●●●●
   |   ●●●●●●
   |  ●●●●●●●  Speech
   |   ●●●●●●   ●●●
   |    ●●●●●  ●●●●
   |     ●●●  ●●●●●
   |__________________→ Mic 1 Signal

Problem: Speech and noise overlap!
```

**After Rotation (Learned Feature Space):**
```
Feature Dim 2
   ↑
   |              Noise
   |               ●●●
   |              ●●●●
   |             ●●●●●
   |
   |  Speech
   |   ●●●●●
   |  ●●●●●●
   | ●●●●●●●
   |__________________→ Feature Dim 1

Solution: Better separation!
```

### Why Rotation Helps

**1. Spatial Mixing (Beamforming Analogy)**

Traditional beamforming:
```
        Mic 1 ──┐
                ├──→ w₁×Mic1 + w₂×Mic2 = Output
        Mic 2 ──┘

Weights (w₁, w₂) chosen to:
- Enhance signal from target direction
- Suppress signal from noise direction
```

SGN Rotation:
```
        Mic 1 ──┐
                ├──→ [W_rot] → 8 channels
        Mic 2 ──┘

Weights learned to:
- Create multiple "beams" (8 channels)
- Each channel captures different spatial pattern
- Optimal for noise suppression (learned from data)
```

**2. Feature Space Projection**

Think of it like PCA/ICA but learned:
```
Original Space:        Rotated Space:
  Mic 1 & Mic 2    →   Principal Components
  (correlated)         (decorrelated)
  
  ●●●●●●●●         →   ●●●●
  ●●●●●●●●              ●●●●
  ●●●●●●●●                  ●●●●
                                ●●●●
```

**3. Dimensionality Expansion**

```
640 features → 8 × 640 = 5,120 features

Why expand?
- More capacity to represent complex patterns
- Each of 8 channels can specialize
- Similar to convolutional filters in CNNs
```

### What the Network Learns

**Learned Rotation Matrix Visualization:**
```
W_rot (640×640 per channel):

Channel 1: Focus on direct path
  ████░░░░░░░░░░░░
  ████░░░░░░░░░░░░
  ░░░░░░░░░░░░░░░░

Channel 2: Focus on reflections
  ░░░░████░░░░░░░░
  ░░░░████░░░░░░░░
  ░░░░░░░░░░░░░░░░

Channel 3: Noise suppression
  ░░░░░░░░░░░░████
  ░░░░░░░░░░░░████
  ░░░░░░░░░░░░░░░░

... (8 channels total)
```

**Interpretation:**
- **Dark regions**: Strong weights (important features)
- **Light regions**: Weak weights (ignored features)
- Each channel learns different spatial pattern

---

## 3. Concatenation: Adding Echo Context

### What Happens

**Rotated Features + Delayed Reference:**
```
Current Frame (k):
  Rotated: [8 × 640]
  
Delayed Reference:
  Frame (k-2): [8 × 480]  ← 2 frames ago
  Frame (k-1): [8 × 480]  ← 1 frame ago
  Total:       [8 × 960]

Concatenated: [8 × 1280]
```

### Visual Representation

**Temporal Alignment:**
```
Time →
Reference:  [k-2]  [k-1]  [k]
            ████   ████   ████
                    ↓
Echo Path:         ████   ████  ← Delayed echo
                    ↓      ↓
Microphone:        ████   ████  ← Received signal
                           ↑
                    Current frame
```

**Feature Concatenation:**
```
┌─────────────────────────────────┐
│  Rotated Features (8×640)       │ ← Spatial info
│  - Current mic signals          │
│  - Spatially enhanced           │
├─────────────────────────────────┤
│  Delayed Reference (8×960)      │ ← Temporal info
│  - Past 2 frames                │
│  - Echo path context            │
└─────────────────────────────────┘
        Combined: 8×1280
```

### Why Concatenation Helps

**1. Echo Cancellation Context**

```
Scenario: Video call with speaker echo

Reference (what you played):
  "Hello" → [k-2] [k-1] [k]

Microphone (what you recorded):
  "Hello" (echo) + Your voice + Noise
           ↑
    Delayed by ~20ms

Network learns:
  Echo ≈ f(Reference[k-2], Reference[k-1])
  Clean = Mic[k] - Echo - Noise
```

**2. Temporal Causality**

```
Without delay:
  Ref[k] → Mic[k]  ✗ (echo hasn't arrived yet!)

With delay:
  Ref[k-2], Ref[k-1] → Mic[k]  ✓ (echo has arrived)
```

**3. Multi-Frame Context**

```
Single frame:
  Limited information about echo path

Multiple frames:
  Can estimate time-varying echo path
  Better for non-stationary scenarios
```

---

## 4. BiLSTM: Temporal Pattern Learning

### What Happens

**Bidirectional Processing:**
```
Forward LSTM:  →→→→→→→→→
Input:         [t₁][t₂][t₃][t₄][t₅]
Backward LSTM: ←←←←←←←←←

Output: Concatenate forward + backward
```

### Visual Representation

**Temporal Context Window:**
```
Time →
Frame:    [t-2]  [t-1]  [t]   [t+1]  [t+2]
          ────────────────────────────────
Forward:  ████   ████   ████   ░░░░   ░░░░  ← Past context
Backward: ░░░░   ░░░░   ████   ████   ████  ← Future context
          ────────────────────────────────
BiLSTM:   ████   ████   ████   ████   ████  ← Full context
                         ↑
                  Current frame
```

**Hidden State Evolution:**
```
Forward LSTM (captures past):
t=1: h₁ = f(x₁)
t=2: h₂ = f(x₂, h₁)        ← Remembers t=1
t=3: h₃ = f(x₃, h₂)        ← Remembers t=1,2
t=4: h₄ = f(x₄, h₃)        ← Remembers t=1,2,3

Backward LSTM (captures future):
t=4: h₄ = f(x₄)
t=3: h₃ = f(x₃, h₄)        ← Knows t=4
t=2: h₂ = f(x₂, h₃)        ← Knows t=3,4
t=1: h₁ = f(x₁, h₂)        ← Knows t=2,3,4
```

### Why BiLSTM Helps

**1. Speech Continuity**

```
Speech: "The cat sat on the mat"
         ↓    ↓   ↓   ↓   ↓   ↓
Frames: [f₁] [f₂][f₃][f₄][f₅][f₆]

At frame f₃ ("sat"):
- Forward LSTM knows: "The cat"
- Backward LSTM knows: "on the mat"
- Combined: Full sentence context

Benefit: Better distinguish speech from noise
```

**2. Non-Stationary Noise Tracking**

```
Noise Level:
High  ████████░░░░░░░░████████
Low   ░░░░░░░░████████░░░░░░░░
      ────────────────────────→ Time
              ↑
        Current frame

BiLSTM learns:
- Noise was high in past (forward)
- Noise will be high in future (backward)
- Current frame: Transition period
- Adapt suppression accordingly
```

**3. Echo Path Modeling**

```
Echo Path (time-varying):
Room acoustics change due to:
- Head movement
- Object movement
- Door opening/closing

BiLSTM tracks:
- How echo path evolved (forward)
- How echo path will evolve (backward)
- Better echo prediction
```

### What Features Look Like

**Before BiLSTM:**
```
Frame-by-frame features (no temporal context):
[f₁] [f₂] [f₃] [f₄] [f₅]
 ●    ●    ●    ●    ●   ← Independent

Problem: Can't distinguish:
- Speech pause vs noise
- Transient noise vs speech onset
```

**After BiLSTM:**
```
Temporally-aware features:
[f₁] [f₂] [f₃] [f₄] [f₅]
 ●────●────●────●────●   ← Connected

Solution: Knows:
- Speech pause (silence between words)
- Transient noise (sudden spike)
- Speech onset (beginning of word)
```

---

## 5. Dual LSTM Branches: Task Specialization

### What Happens

**Branch Split:**
```
BiLSTM Output
      ↓
   ┌──┴──┐
   ↓     ↓
LSTM₁  LSTM₂
   ↓     ↓
  EC    NS
```

### Visual Representation

**Task Decomposition:**
```
Noisy Signal = Clean Speech + Echo + Noise
     ↓              ↓          ↓      ↓
  [Input]      [Target]    [EC]    [NS]

Branch 1 (Echo Cancellation):
  Learns: Echo ≈ f(Reference)
  ████████░░░░░░░░  ← High correlation with ref
  
Branch 2 (Noise Suppression):
  Learns: Noise ≈ g(Statistics)
  ░░░░░░░░████████  ← Uncorrelated with ref
```

**Feature Specialization:**
```
Shared BiLSTM:
  ████████████████  ← Common temporal patterns
  
Branch 1 (EC):
  ████░░░░░░░░░░░░  ← Echo-specific features
  - Correlation with reference
  - Linear/non-linear echo paths
  - Delay patterns
  
Branch 2 (NS):
  ░░░░░░░░░░░░████  ← Noise-specific features
  - Statistical noise properties
  - Spectral characteristics
  - Temporal stationarity
```

### Why Dual Branches Help

**1. Multi-Task Learning**

```
Single Task (NS only):
  Network learns: Clean = Noisy - Noise
  Problem: Echo treated as noise → distortion

Dual Task (EC + NS):
  Branch 1: Clean = Noisy - Echo
  Branch 2: Clean = Noisy - Noise
  Combined: Clean = Noisy - Echo - Noise
  Solution: Separate treatment → better quality
```

**2. Feature Specialization**

```
Echo Characteristics:
- Correlated with reference signal
- Predictable delay pattern
- Linear/non-linear relationship

Noise Characteristics:
- Uncorrelated with reference
- Random/stationary
- Statistical properties

Separate branches → Better modeling of each
```

**3. Preventing Interference**

```
Without Separation:
  EC objective: Minimize |Clean - (Noisy - Echo)|
  NS objective: Minimize |Clean - (Noisy - Noise)|
  Problem: Conflicting gradients!

With Separation:
  Branch 1: Focus on EC
  Branch 2: Focus on NS
  Solution: Independent optimization
```

### What Each Branch Learns

**Branch 1 (Echo Cancellation):**
```
Learned Patterns:
1. Direct echo:
   Ref[k-2] → Echo[k]
   ████░░░░

2. Multi-path echo:
   Ref[k-2] + Ref[k-1] → Echo[k]
   ████████

3. Non-linear echo:
   f(Ref[k-2], Ref[k-1]) → Echo[k]
   ████████ (complex function)
```

**Branch 2 (Noise Suppression):**
```
Learned Patterns:
1. Stationary noise:
   Constant spectrum
   ████████████████

2. Non-stationary noise:
   Time-varying spectrum
   ████░░░░████░░░░

3. Transient noise:
   Sudden spikes
   ░░░░████░░░░░░░░
```

---

## 6. FC Layers: Feature Refinement

### What Happens

**Non-Linear Mapping:**
```
LSTM Features → FC Layer → Mask Features

Branch 1: [8×320] → FC₁ → [8×640]
Branch 2: [8×320] → FC₂ → [8×320]
```

### Visual Representation

**Feature Transformation:**
```
LSTM Output (temporal features):
  ████████████████  ← Rich temporal patterns
  ████░░░░████░░░░
  ░░░░████░░░░████

FC Layer (non-linear mapping):
  ↓ ReLU(W·x + b)
  
Mask Features (suppression patterns):
  ████░░░░░░░░░░░░  ← Suppress these frequencies
  ░░░░████████████  ← Keep these frequencies
  ░░░░░░░░░░░░████  ← Partially suppress
```

**Dimension Matching:**
```
Branch 1:
  320 → 640 (expansion)
  More capacity for complex patterns
  
Branch 2:
  320 → 320 (refinement)
  Polish existing features
```

### Why FC Layers Help

**1. Non-Linear Decision Boundaries**

```
Linear Separation (without FC):
  Noise │ Speech
  ●●●●● │ ●●●●●
  ●●●●● │ ●●●●●
  ──────┼────────
        │
  Simple boundary

Non-Linear Separation (with FC):
  Noise    Speech
  ●●●●●   ●●●●●
  ●●●●●  ╱●●●●●
  ●●●●●╱ ●●●●●
  ─────╲────────
        ╲
  Complex boundary
```

**2. Feature Abstraction**

```
LSTM Features:
  "Frame t has high energy at 2kHz"
  "Frame t-1 had high energy at 2kHz"
  "Frame t+1 will have high energy at 2kHz"

FC Layer Learns:
  "Sustained 2kHz energy = Speech formant"
  → Keep this frequency
  
  "Transient 2kHz spike = Noise"
  → Suppress this frequency
```

**3. Mask Space Projection**

```
LSTM Space:
  Temporal patterns
  [h₁, h₂, h₃, ..., h₃₂₀]

Mask Space:
  Suppression gains
  [m₁, m₂, m₃, ..., m₁₆₁]
  where mᵢ ∈ [0, 1]

FC Layer: Maps temporal → spectral
```

---

## 7. Filter Block: Adaptive Masking

### What Happens

**Mask Generation:**
```
FC Outputs → Combine → Sigmoid → Mask
[8×640]              [8×320]    [161]
[8×320]
```

**Apply Mask:**
```
Noisy Spectrum × Mask = Clean Spectrum
```

### Visual Representation

**Mask Visualization:**
```
Frequency (Hz)
   ↑
8000|  ░░░░░░░░░░░░░░  ← Suppress (mask ≈ 0)
4000|  ████░░██░░██░░  ← Keep speech (mask ≈ 1)
2000|  ████████░░░░░░  ← Keep speech (mask ≈ 1)
1000|  ████████████░░  ← Partial (mask ≈ 0.5)
   0|__________________→ Time
     
Legend: █ = Keep (mask=1), ░ = Suppress (mask=0)
```

**Before and After:**
```
Noisy Spectrum:
Freq
  ↑
  |  ████████████████  ← Noise everywhere
  |  ████████████████
  |  ████████████████
  |__________________→ Time

Mask:
  |  ░░░░░░░░░░░░░░░░  ← Suppress high freq
  |  ████░░██░░██░░░░  ← Keep speech freq
  |  ████████████████  ← Keep low freq

Clean Spectrum:
  |  ░░░░░░░░░░░░░░░░  ← Noise removed
  |  ████░░██░░██░░░░  ← Speech preserved
  |  ████████████████  ← Low freq kept
```

### Why Adaptive Filtering Works

**1. Frequency-Selective Suppression**

```
Traditional (Fixed Filter):
  All frequencies treated equally
  Mask = constant
  
  Problem: Can't adapt to:
  - Time-varying noise
  - Different noise types
  - Speech characteristics

SGN (Adaptive Filter):
  Each frequency independently controlled
  Mask[f, t] = learned function
  
  Solution: Adapts to:
  - Noise spectrum changes
  - Speech formant locations
  - SNR variations
```

**2. Comparison to Wiener Filter**

```
Wiener Filter (Statistical):
  M[f] = S²[f] / (S²[f] + N²[f])
  
  Requires:
  - Noise statistics (N²)
  - Speech statistics (S²)
  - Stationarity assumption
  
  Limitations:
  - Assumes Gaussian noise
  - Requires noise estimation
  - Linear processing

SGN Mask (Learned):
  M[f, t] = σ(NN(X_noisy, R_ref))
  
  Learns:
  - Non-linear relationships
  - Complex noise patterns
  - Speech characteristics
  
  Advantages:
  - No assumptions needed
  - Handles non-stationary noise
  - Uses multi-mic + temporal info
```

**3. Soft Masking vs Hard Masking**

```
Hard Mask (Binary):
  M[f] ∈ {0, 1}
  ████░░░░████░░░░
  
  Problem:
  - Musical noise artifacts
  - Harsh transitions
  - Speech distortion

Soft Mask (Continuous):
  M[f] ∈ [0, 1]
  ████▓▓▓▓████▓▓▓▓
  
  Solution:
  - Smooth suppression
  - Natural transitions
  - Better quality
```

### What the Mask Learns

**Speech Regions:**
```
Formants (vowels):
  F1: 500-1000 Hz   → Mask ≈ 1.0
  F2: 1000-2000 Hz  → Mask ≈ 1.0
  F3: 2000-3000 Hz  → Mask ≈ 0.8

Consonants:
  Fricatives: 4-8kHz → Mask ≈ 0.7
  Stops: Broadband   → Mask ≈ 0.6
```

**Noise Regions:**
```
High-frequency noise:
  > 6kHz → Mask ≈ 0.1 (strong suppression)

Low-frequency noise:
  < 500Hz → Mask ≈ 0.3 (moderate suppression)

Speech-overlapping noise:
  1-3kHz → Mask ≈ 0.5-0.8 (careful balance)
```

---

## 8. Complete Feature Journey

### End-to-End Visualization

```
┌─────────────────────────────────────────────────────────┐
│ INPUT: Noisy Audio (2 mics + reference)                │
│ Features: Raw waveforms                                 │
│ Characteristics: Time-domain, mixed signal              │
└────────────────────┬────────────────────────────────────┘
                     ↓ STFT
┌─────────────────────────────────────────────────────────┐
│ SPECTROGRAMS                                            │
│ Features: Frequency-domain representation               │
│ Characteristics: Time-frequency, still mixed            │
│ Dimensions: [B, K, 2, 320]                             │
└────────────────────┬────────────────────────────────────┘
                     ↓ Rotation Layer
┌─────────────────────────────────────────────────────────┐
│ SPATIALLY ENHANCED FEATURES                             │
│ Features: Multi-channel spatial projections             │
│ Characteristics: Better source separation               │
│ Dimensions: [B, K, 8, 640]                             │
│                                                         │
│ What changed:                                           │
│ - 2 mics → 8 channels (learned beamforming)           │
│ - Spatial mixing improves separability                 │
│ - Each channel captures different spatial pattern      │
└────────────────────┬────────────────────────────────────┘
                     ↓ Concatenate with Delayed Ref
┌─────────────────────────────────────────────────────────┐
│ SPATIAL + TEMPORAL FEATURES                             │
│ Features: Current + past reference frames               │
│ Characteristics: Echo cancellation context              │
│ Dimensions: [B, K, 8, 1280]                            │
│                                                         │
│ What changed:                                           │
│ - Added 2 past reference frames (960 features)         │
│ - Enables echo path modeling                           │
│ - Temporal alignment for EC                            │
└────────────────────┬────────────────────────────────────┘
                     ↓ BiLSTM
┌─────────────────────────────────────────────────────────┐
│ TEMPORALLY AWARE FEATURES                               │
│ Features: Past + future context                         │
│ Characteristics: Long-term dependencies                 │
│ Dimensions: [B, K, 8, 320]                             │
│                                                         │
│ What changed:                                           │
│ - Compressed 1280 → 320 (information bottleneck)      │
│ - Added bidirectional temporal context                 │
│ - Learned speech continuity patterns                   │
│ - Tracked non-stationary noise                         │
└────────────────────┬────────────────────────────────────┘
                     ↓ Split into Dual Branches
┌──────────────────────────┬──────────────────────────────┐
│ BRANCH 1: EC FEATURES    │ BRANCH 2: NS FEATURES        │
│ Specialization: Echo     │ Specialization: Noise        │
│ Dimensions: [B,K,8,320]  │ Dimensions: [B,K,8,320]      │
│                          │                              │
│ Learns:                  │ Learns:                      │
│ - Echo correlation       │ - Noise statistics           │
│ - Delay patterns         │ - Spectral characteristics   │
│ - Non-linear paths       │ - Temporal stationarity      │
└──────────────────────────┴──────────────────────────────┘
                     ↓ FC Layers
┌──────────────────────────┬──────────────────────────────┐
│ FC1 OUTPUT: [B,K,8,640]  │ FC2 OUTPUT: [B,K,8,320]      │
│ Non-linear mapping       │ Feature refinement           │
│ Expanded capacity        │ Polished features            │
└──────────────────────────┴──────────────────────────────┘
                     ↓ Combine + Filter
┌─────────────────────────────────────────────────────────┐
│ SUPPRESSION MASK                                        │
│ Features: Frequency-wise gains                          │
│ Characteristics: Adaptive, learned                      │
│ Dimensions: [B, K, 161]                                │
│                                                         │
│ Values: M[f, t] ∈ [0, 1]                               │
│ - 0: Complete suppression (noise)                      │
│ - 1: Full preservation (speech)                        │
│ - 0.5: Partial suppression (uncertain)                 │
└────────────────────┬────────────────────────────────────┘
                     ↓ Apply Mask
┌─────────────────────────────────────────────────────────┐
│ CLEAN SPECTRUM                                          │
│ Features: Noise-suppressed frequency representation     │
│ Characteristics: Enhanced speech, reduced noise         │
│ Dimensions: [B, K, 161]                                │
└────────────────────┬────────────────────────────────────┘
                     ↓ ISTFT
┌─────────────────────────────────────────────────────────┐
│ OUTPUT: Clean Audio                                     │
│ Features: Time-domain waveform                          │
│ Characteristics: Speech preserved, noise removed        │
└─────────────────────────────────────────────────────────┘
```

### Feature Complexity Evolution

```
Stage                 Complexity    Information Content
─────────────────────────────────────────────────────────
Input (Raw Audio)     Low           Mixed signal
  ↓
STFT                  Medium        Frequency separation
  ↓
Rotation              High          Spatial separation
  ↓
Concat + BiLSTM       Very High     Spatio-temporal
  ↓
Dual Branches         Specialized   Task-specific
  ↓
FC Layers             Refined       Decision-ready
  ↓
Mask                  Simple        Binary-like decisions
  ↓
Output (Clean)        Low           Separated signal
```

---

## 9. Key Insights

### Why Each Component is Essential

**1. Rotation Layer**
- **Without**: Limited to 2-mic information, poor spatial separation
- **With**: 8 learned spatial patterns, better source separation
- **Analogy**: Like having 8 different microphone array configurations

**2. BiLSTM**
- **Without**: Frame-by-frame processing, no temporal context
- **With**: Full temporal awareness, speech continuity
- **Analogy**: Like reading a sentence vs reading individual letters

**3. Dual Branches**
- **Without**: Single model tries to do EC+NS, conflicting objectives
- **With**: Specialized models, better at each task
- **Analogy**: Like having specialist doctors vs general practitioner

**4. Adaptive Filtering**
- **Without**: Fixed suppression, poor adaptation
- **With**: Learned frequency-selective suppression
- **Analogy**: Like smart noise cancellation vs simple volume reduction

### What Makes SGN Effective

**Combines Multiple Domains:**
1. **Spatial**: Multi-microphone (Rotation)
2. **Temporal**: LSTM (BiLSTM, Branches)
3. **Spectral**: Frequency-domain (STFT, Mask)

**Learns from Data:**
- No assumptions about noise type
- Adapts to speaker characteristics
- Handles non-stationary scenarios

**Efficient Architecture:**
- Only 5.5M parameters
- Real-time capable (0.5 GMAC/s)
- Low latency (20ms)

---

*End of Visual Guide*
