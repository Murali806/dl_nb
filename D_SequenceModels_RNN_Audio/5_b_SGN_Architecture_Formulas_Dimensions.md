# SGN Architecture: Mathematical Formulation and Dimension Analysis

## Overview
This document provides complete mathematical formulation of the SGN (Speech Enhancement Network) model for multi-microphone Echo Cancellation (EC) and Noise Suppression (NS).

**Model Configuration (15.x - 2 mic SGN+EC+NS):**
- Window size: 20ms (sine window)
- Hop size: 10ms (50% overlap)
- FFT size: 320
- Delay: 20ms (2 frames lookback)
- Memory: 5.5M parameters
- Complexity: ~0.5 GMAC/s

---

## 1. STFT Preprocessing

### 1.1 Short-Time Fourier Transform

**Input:** Time-domain signals from 2 microphones and reference
- $x_1(n)$: Microphone 1 signal, $n \in [0, N-1]$
- $x_2(n)$: Microphone 2 signal
- $r(n)$: Reference signal (for echo cancellation)

**Windowing:**
$$w(n) = \sin\left(\frac{\pi(n + 0.5)}{L}\right), \quad n \in [0, L-1]$$

where $L = 320$ (20ms at 16kHz sampling rate)

**STFT Computation:**
$$X_m(k, f) = \sum_{n=0}^{L-1} x_m(n + kH) \cdot w(n) \cdot e^{-j2\pi fn/L}$$

where:
- $m \in \{1, 2\}$: microphone index
- $k$: frame index
- $f \in [0, 159]$: frequency bin (using only positive frequencies)
- $H = 160$: hop size (10ms)

**Magnitude and Phase:**
$$|X_m(k, f)| = \sqrt{\text{Re}(X_m(k,f))^2 + \text{Im}(X_m(k,f))^2}$$
$$\angle X_m(k, f) = \arctan\left(\frac{\text{Im}(X_m(k,f))}{\text{Re}(X_m(k,f))}\right)$$

**Dimensions:**
- Input: $[B, T]$ where $B$ = batch size, $T$ = time samples
- After STFT: $[B, K, F]$ where $K$ = number of frames, $F = 161$ frequency bins
- Network uses: $[B, K, 320]$ (concatenating real and imaginary parts)

---

## 2. Pre-processing Layer

### 2.1 FFT Layer
Converts time-domain to frequency-domain representation.

**Input dimensions:** $[B, 1, 160]$ per microphone
**Output dimensions:** $[B, 1, 161]$ (magnitude spectrum)

For 2 microphones:
$$\mathbf{X}_{\text{pre}} = [\mathbf{X}_1; \mathbf{X}_2] \in \mathbb{R}^{B \times 2 \times 161}$$

### 2.2 Pre-processing Transformation
$$\mathbf{X}_{\text{proc}} = \text{PreProc}(\mathbf{X}_{\text{pre}}) \in \mathbb{R}^{B \times 2 \times 320}$$

This doubles the feature dimension to 320 (likely concatenating magnitude and phase or real/imaginary components).

---

## 3. Rotation Layer

### 3.1 Purpose
The Rotation layer performs a **learnable linear transformation** that:
1. Mixes spatial information from multiple microphones
2. Projects to higher-dimensional feature space
3. Acts as data-driven beamforming
4. Enhances feature separability

### 3.2 Mathematical Formulation

**Weight Matrix:**
$$\mathbf{W}_{\text{rot}} \in \mathbb{R}^{640 \times 640}$$

**Bias Vector:**
$$\mathbf{b}_{\text{rot}} \in \mathbb{R}^{640}$$

**Forward Pass:**
$$\mathbf{Y}_{\text{rot}} = \mathbf{W}_{\text{rot}} \cdot \mathbf{X}_{\text{concat}} + \mathbf{b}_{\text{rot}}$$

where $\mathbf{X}_{\text{concat}} \in \mathbb{R}^{B \times 640}$ is the concatenated input from 2 mics (each 320-dim).

**Dimension Transformation:**
$$[B, 2, 320] \xrightarrow{\text{reshape}} [B, 640] \xrightarrow{\text{Linear}} [B, 640] \xrightarrow{\text{reshape}} [B, 8, 80]$$

Wait, let me recalculate based on the architecture diagram...

**Corrected Dimensions (from diagram):**
- Input: 2 microphones × 320 features = $[B, 2 \times 320]$
- Output: 8 channels × 640 features = $[B, 8, 640]$

Actually, the rotation layer output is $8 \times 640$, so:

$$\mathbf{Y}_{\text{rot}} \in \mathbb{R}^{B \times 8 \times 640}$$

The transformation can be viewed as:
$$\mathbf{Y}_{\text{rot}}[b, c, :] = \mathbf{W}_{\text{rot}}^{(c)} \cdot \mathbf{X}_{\text{flat}}[b, :] + \mathbf{b}_{\text{rot}}^{(c)}$$

where:
- $c \in [0, 7]$: output channel index
- $\mathbf{W}_{\text{rot}}^{(c)} \in \mathbb{R}^{640 \times 640}$
- $\mathbf{X}_{\text{flat}}[b, :] \in \mathbb{R}^{640}$ (flattened 2×320 input)

### 3.3 Why Rotation Helps

**Spatial Feature Mixing:**
$$\mathbf{Y}_{\text{rot}}[i] = \sum_{j=1}^{640} w_{ij} \cdot \mathbf{X}[j]$$

This creates **linear combinations** of features from both microphones, similar to:
- **Beamforming:** $y = \sum_m w_m \cdot x_m$ (weighted sum of mic signals)
- **ICA/PCA:** Finding optimal projections for source separation

**Learned vs Traditional Beamforming:**
- Traditional: $\mathbf{w}$ computed from geometry/statistics
- SGN Rotation: $\mathbf{W}_{\text{rot}}$ learned from data to maximize noise suppression

### 3.4 Gradient Flow (Backpropagation)

**Loss gradient w.r.t. output:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{Y}_{\text{rot}}} \in \mathbb{R}^{B \times 8 \times 640}$$

**Gradient w.r.t. weights:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{\text{rot}}} = \frac{\partial \mathcal{L}}{\partial \mathbf{Y}_{\text{rot}}} \cdot \mathbf{X}_{\text{flat}}^T$$

**Gradient w.r.t. input:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{X}_{\text{flat}}} = \mathbf{W}_{\text{rot}}^T \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{Y}_{\text{rot}}}$$

---

## 4. Concatenation with Delayed Reference

### 4.1 Delay Operation

**Reference Signal STFT:**
$$\mathbf{R}(k, f) = \text{STFT}(r(n))$$

**Delayed Reference (2 frames = 20ms lookback):**
$$\mathbf{R}_{\text{delayed}}(k) = [\mathbf{R}(k-2), \mathbf{R}(k-1)]$$

**Dimensions:**
- Current rotated features: $[B, 8, 640]$
- Delayed reference: $[B, 8, 960]$ (includes 2 past frames)

### 4.2 Concatenation

$$\mathbf{Z} = [\mathbf{Y}_{\text{rot}}; \mathbf{R}_{\text{delayed}}] \in \mathbb{R}^{B \times 8 \times 1280}$$

**Why Concatenation Helps:**
- Provides **echo path information** for cancellation
- Temporal alignment accounts for acoustic delay
- Enables prediction of echo component: $\hat{e}(k) = f(\mathbf{R}(k-2), \mathbf{R}(k-1))$

---

## 5. Bidirectional LSTM (BiLSTM)

### 5.1 LSTM Cell Equations

**Forward LSTM at time $t$:**

**Forget Gate:**
$$\mathbf{f}_t^{(\rightarrow)} = \sigma(\mathbf{W}_f^{(\rightarrow)} \cdot [\mathbf{h}_{t-1}^{(\rightarrow)}; \mathbf{z}_t] + \mathbf{b}_f^{(\rightarrow)})$$

**Input Gate:**
$$\mathbf{i}_t^{(\rightarrow)} = \sigma(\mathbf{W}_i^{(\rightarrow)} \cdot [\mathbf{h}_{t-1}^{(\rightarrow)}; \mathbf{z}_t] + \mathbf{b}_i^{(\rightarrow)})$$

**Cell Candidate:**
$$\tilde{\mathbf{c}}_t^{(\rightarrow)} = \tanh(\mathbf{W}_c^{(\rightarrow)} \cdot [\mathbf{h}_{t-1}^{(\rightarrow)}; \mathbf{z}_t] + \mathbf{b}_c^{(\rightarrow)})$$

**Cell State Update:**
$$\mathbf{c}_t^{(\rightarrow)} = \mathbf{f}_t^{(\rightarrow)} \odot \mathbf{c}_{t-1}^{(\rightarrow)} + \mathbf{i}_t^{(\rightarrow)} \odot \tilde{\mathbf{c}}_t^{(\rightarrow)}$$

**Output Gate:**
$$\mathbf{o}_t^{(\rightarrow)} = \sigma(\mathbf{W}_o^{(\rightarrow)} \cdot [\mathbf{h}_{t-1}^{(\rightarrow)}; \mathbf{z}_t] + \mathbf{b}_o^{(\rightarrow)})$$

**Hidden State:**
$$\mathbf{h}_t^{(\rightarrow)} = \mathbf{o}_t^{(\rightarrow)} \odot \tanh(\mathbf{c}_t^{(\rightarrow)})$$

**Backward LSTM** (processes sequence in reverse):
$$\mathbf{h}_t^{(\leftarrow)} = \text{LSTM}^{(\leftarrow)}(\mathbf{z}_t, \mathbf{h}_{t+1}^{(\leftarrow)})$$

### 5.2 BiLSTM Output

**Concatenation of forward and backward:**
$$\mathbf{h}_t^{\text{bi}} = [\mathbf{h}_t^{(\rightarrow)}; \mathbf{h}_t^{(\leftarrow)}]$$

**Dimension Transformation:**
- Input: $\mathbf{z}_t \in \mathbb{R}^{8 \times 1280}$
- Forward hidden: $\mathbf{h}_t^{(\rightarrow)} \in \mathbb{R}^{8 \times 160}$
- Backward hidden: $\mathbf{h}_t^{(\leftarrow)} \in \mathbb{R}^{8 \times 160}$
- BiLSTM output: $\mathbf{h}_t^{\text{bi}} \in \mathbb{R}^{8 \times 320}$

### 5.3 Why BiLSTM Helps

**Temporal Context:**
- Forward LSTM: Captures past context $\mathbf{h}_t^{(\rightarrow)} = f(\mathbf{z}_1, ..., \mathbf{z}_t)$
- Backward LSTM: Captures future context $\mathbf{h}_t^{(\leftarrow)} = f(\mathbf{z}_T, ..., \mathbf{z}_t)$
- Combined: Full temporal context $\mathbf{h}_t^{\text{bi}} = f(\mathbf{z}_1, ..., \mathbf{z}_T)$

**Benefits for Noise Suppression:**
1. **Non-stationary noise tracking:** Adapts to time-varying noise
2. **Speech continuity:** Uses future frames to better estimate current frame
3. **Echo path modeling:** Tracks time-varying acoustic paths

---

## 6. LSTM Branches (×2)

### 6.1 Dual Branch Architecture

**Branch 1 (likely for Echo Cancellation):**
$$\mathbf{h}_t^{(1)} = \text{LSTM}_1(\mathbf{h}_t^{\text{bi}}, \mathbf{h}_{t-1}^{(1)})$$

**Branch 2 (likely for Noise Suppression):**
$$\mathbf{h}_t^{(2)} = \text{LSTM}_2(\mathbf{h}_t^{\text{bi}}, \mathbf{h}_{t-1}^{(2)})$$

**Dimensions:**
- Input to each branch: $\mathbf{h}_t^{\text{bi}} \in \mathbb{R}^{8 \times 320}$
- Output from each branch: $\mathbf{h}_t^{(i)} \in \mathbb{R}^{8 \times 320}$

### 6.2 LSTM Cell (Standard)

Each branch uses standard LSTM equations (same as BiLSTM but unidirectional):

$$\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_f)$$
$$\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_i)$$
$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \cdot [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_c)$$
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_o)$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

### 6.3 Why Dual Branches Help

**Task Decomposition:**
- **Branch 1 specialization:** Learns echo-specific patterns
  - Echo is correlated with reference signal
  - Requires modeling linear/non-linear echo paths
  
- **Branch 2 specialization:** Learns noise-specific patterns
  - Noise is uncorrelated with reference
  - Requires statistical noise modeling

**Mathematical Intuition:**
$$\mathbf{y}_{\text{noisy}} = \mathbf{s}_{\text{clean}} + \mathbf{e}_{\text{echo}} + \mathbf{n}_{\text{noise}}$$

- Branch 1 estimates: $\hat{\mathbf{e}}_{\text{echo}}$
- Branch 2 estimates: $\hat{\mathbf{n}}_{\text{noise}}$
- Combined: $\hat{\mathbf{s}}_{\text{clean}} = \mathbf{y}_{\text{noisy}} - \hat{\mathbf{e}}_{\text{echo}} - \hat{\mathbf{n}}_{\text{noise}}$

**Multi-task Learning Benefit:**
- Shared BiLSTM learns common representations
- Separate branches learn task-specific features
- Prevents interference between EC and NS objectives

---

## 7. Fully Connected (FC) Layers

### 7.1 FC Layer 1 (after Branch 1)

**Linear Transformation:**
$$\mathbf{y}_1 = \mathbf{W}_1 \cdot \mathbf{h}^{(1)} + \mathbf{b}_1$$

**Dimensions:**
- Input: $\mathbf{h}^{(1)} \in \mathbb{R}^{8 \times 320}$
- Weights: $\mathbf{W}_1 \in \mathbb{R}^{640 \times 320}$
- Output: $\mathbf{y}_1 \in \mathbb{R}^{8 \times 640}$

**Activation (typically ReLU or none):**
$$\mathbf{y}_1 = \max(0, \mathbf{W}_1 \cdot \mathbf{h}^{(1)} + \mathbf{b}_1)$$

### 7.2 FC Layer 2 (after Branch 2)

**Linear Transformation:**
$$\mathbf{y}_2 = \mathbf{W}_2 \cdot \mathbf{h}^{(2)} + \mathbf{b}_2$$

**Dimensions:**
- Input: $\mathbf{h}^{(2)} \in \mathbb{R}^{8 \times 320}$
- Weights: $\mathbf{W}_2 \in \mathbb{R}^{320 \times 320}$
- Output: $\mathbf{y}_2 \in \mathbb{R}^{8 \times 320}$

### 7.3 Why FC Layers Help

**Non-linear Mapping:**
- Maps LSTM features to mask/gain space
- Learns complex relationships: $\text{Mask} = f_{\text{FC}}(\text{LSTM features})$

**Dimension Matching:**
- Ensures outputs match for combination
- FC1 expands: $320 \rightarrow 640$ (more capacity)
- FC2 maintains: $320 \rightarrow 320$ (refinement)

---

## 8. Filter Block (Mask Generation)

### 8.1 Feature Combination

**Concatenation or Addition:**
$$\mathbf{y}_{\text{combined}} = [\mathbf{y}_1; \mathbf{y}_2] \quad \text{or} \quad \mathbf{y}_{\text{combined}} = \mathbf{y}_1 + \mathbf{y}_2$$

Based on diagram (⊕ symbol), likely addition after dimension matching:
$$\mathbf{y}_{\text{combined}} = \mathbf{y}_1[:, :320] + \mathbf{y}_2 \in \mathbb{R}^{8 \times 320}$$

### 8.2 Mask Computation

**Final FC Layer:**
$$\mathbf{m}_{\text{logits}} = \mathbf{W}_{\text{mask}} \cdot \mathbf{y}_{\text{combined}} + \mathbf{b}_{\text{mask}}$$

**Sigmoid Activation (for gain mask):**
$$\mathbf{M}(k, f) = \sigma(\mathbf{m}_{\text{logits}}(k, f)) = \frac{1}{1 + e^{-\mathbf{m}_{\text{logits}}(k, f)}}$$

where $\mathbf{M}(k, f) \in [0, 1]$ is the gain mask for frame $k$, frequency $f$.

**Alternative: Softmax (for ratio mask):**
$$\mathbf{M}_{\text{speech}}(k, f) = \frac{e^{\mathbf{m}_{\text{speech}}(k, f)}}{e^{\mathbf{m}_{\text{speech}}(k, f)} + e^{\mathbf{m}_{\text{noise}}(k, f)}}$$

### 8.3 Adaptive Filtering

**Apply Mask to Noisy Spectrum:**
$$\hat{\mathbf{S}}_{\text{clean}}(k, f) = \mathbf{M}(k, f) \cdot \mathbf{X}_{\text{noisy}}(k, f)$$

**Element-wise multiplication:**
$$\hat{S}_{\text{clean}}(k, f) = M(k, f) \times X_{\text{noisy}}(k, f)$$

**In complex domain:**
$$\hat{S}_{\text{clean}}(k, f) = M(k, f) \times |X_{\text{noisy}}(k, f)| \times e^{j\angle X_{\text{noisy}}(k, f)}$$

### 8.4 Why Adaptive Filtering Works

**Frequency-Selective Suppression:**
- Each frequency bin has independent gain $M(k, f)$
- Suppresses noise-dominated bins: $M(k, f) \approx 0$
- Preserves speech-dominated bins: $M(k, f) \approx 1$

**Comparison to Wiener Filter:**

Traditional Wiener filter:
$$M_{\text{Wiener}}(k, f) = \frac{|S(k, f)|^2}{|S(k, f)|^2 + |N(k, f)|^2}$$

SGN learned mask:
$$M_{\text{SGN}}(k, f) = \sigma(\text{NN}(\mathbf{X}_{\text{noisy}}, \mathbf{R}_{\text{ref}}))$$

**Advantages of learned mask:**
1. Non-linear: Can model complex noise patterns
2. Context-aware: Uses temporal context from LSTM
3. Multi-microphone: Leverages spatial information
4. Adaptive: Learns from data, not assumptions

---

## 9. Inverse STFT (Post-processing)

### 9.1 Inverse FFT

**Reconstruct Complex Spectrum:**
$$\hat{S}_{\text{clean}}(k, f) = |\hat{S}_{\text{clean}}(k, f)| \times e^{j\angle X_{\text{noisy}}(k, f)}$$

(Phase is typically preserved from noisy signal)

**Inverse FFT:**
$$\hat{s}_{\text{clean}}(n + kH) = \text{IFFT}(\hat{S}_{\text{clean}}(k, :))$$

### 9.2 Overlap-Add

**Windowing:**
$$\hat{s}_{\text{windowed}}(n + kH) = \hat{s}_{\text{clean}}(n + kH) \times w(n)$$

**Overlap-Add Reconstruction:**
$$\hat{s}_{\text{output}}(n) = \sum_{k} \hat{s}_{\text{windowed}}(n - kH)$$

**Normalization (for 50% overlap):**
$$\hat{s}_{\text{output}}(n) = \frac{\sum_{k} \hat{s}_{\text{windowed}}(n - kH)}{\sum_{k} w^2(n - kH)}$$

---

## 10. Loss Function

### 10.1 Time-Domain Loss (MSE)

$$\mathcal{L}_{\text{time}} = \frac{1}{N} \sum_{n=1}^{N} (\hat{s}_{\text{clean}}(n) - s_{\text{target}}(n))^2$$

### 10.2 Frequency-Domain Loss (MSE on Magnitude)

$$\mathcal{L}_{\text{freq}} = \frac{1}{KF} \sum_{k=1}^{K} \sum_{f=1}^{F} (|\hat{S}_{\text{clean}}(k, f)| - |S_{\text{target}}(k, f)|)^2$$

### 10.3 Perceptual Loss (SI-SNR)

**Scale-Invariant SNR:**
$$\text{SI-SNR} = 10 \log_{10} \frac{\|\alpha s_{\text{target}}\|^2}{\|\alpha s_{\text{target}} - \hat{s}_{\text{clean}}\|^2}$$

where $\alpha = \frac{\langle \hat{s}_{\text{clean}}, s_{\text{target}} \rangle}{\|s_{\text{target}}\|^2}$

**Loss:**
$$\mathcal{L}_{\text{SI-SNR}} = -\text{SI-SNR}$$

### 10.4 Combined Loss

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{time}} + \lambda_2 \mathcal{L}_{\text{freq}} + \lambda_3 \mathcal{L}_{\text{SI-SNR}}$$

---

## 11. Backpropagation Through Time (BPTT)

### 11.1 Gradient Flow

**Loss gradient w.r.t. mask:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{M}(k, f)} = \frac{\partial \mathcal{L}}{\partial \hat{S}(k, f)} \cdot X_{\text{noisy}}(k, f)$$

**Gradient through sigmoid:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{m}_{\text{logits}}} = \frac{\partial \mathcal{L}}{\partial \mathbf{M}} \cdot \mathbf{M} \cdot (1 - \mathbf{M})$$

**Gradient through FC layers:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_i} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}_i} \cdot \mathbf{h}^{(i)T}$$

### 11.2 LSTM Gradient

**Gradient through LSTM cell:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}_t} + \frac{\partial \mathcal{L}}{\partial \mathbf{h}_{t+1}} \cdot \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}$$

**Cell state gradient:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} \cdot \mathbf{o}_t \cdot (1 - \tanh^2(\mathbf{c}_t)) + \frac{\partial \mathcal{L}}{\partial \mathbf{c}_{t+1}} \cdot \mathbf{f}_{t+1}$$

**Gate gradients:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{f}_t} = \frac{\partial \mathcal{L}}{\partial \mathbf{c}_t} \cdot \mathbf{c}_{t-1} \cdot \mathbf{f}_t \cdot (1 - \mathbf{f}_t)$$

---

## 12. Parameter Count Analysis

### 12.1 Layer-wise Parameters

**Rotation Layer:**
- Weights: $640 \times 640 = 409,600$
- Bias: $640$
- Total: **409,640**

**BiLSTM:**
- Forward LSTM: $4 \times (320 \times (1280 + 160) + 160) = 4 \times 461,600 = 1,846,400$
- Backward LSTM: $1,846,400$
- Total: **3,692,800**

**LSTM Branch 1:**
- Parameters: $4 \times (320 \times (320 + 320) + 320) = 4 \times 205,120 = 820,480$
- Total: **820,480**

**LSTM Branch 2:**
- Parameters: **820,480**

**FC Layers:**
- FC1: $640 \times 320 + 640 = 205,440$
- FC2: $320 \times 320 + 320 = 102,720$
- Total: **308,160**

**Total Parameters:**
$$409,640 + 3,692,800 + 820,480 + 820,480 + 308,160 = 6,051,560 \approx 6M$$

(Close to stated 5.5M, difference may be in exact architecture details)

---

## 13. Computational Complexity

### 13.1 FLOPs per Frame

**STFT/ISTFT:**
- FFT: $O(F \log F) = O(320 \log 320) \approx 2,560$ FLOPs

**Rotation Layer:**
- Matrix multiply: $640 \times 640 = 409,600$ FLOPs

**BiLSTM:**
- Forward: $4 \times 320 \times 1440 = 1,843,200$ FLOPs
- Backward: $1,843,200$ FLOPs
- Total: $3,686,400$ FLOPs

**LSTM Branches:**
- Branch 1: $4 \times 320 \times 640 = 819,200$ FLOPs
- Branch 2: $819,200$ FLOPs
- Total: $1,638,400$ FLOPs

**FC Layers:**
- FC1: $640 \times 320 = 204,800$ FLOPs
- FC2: $320 \times 320 = 102,400$ FLOPs
- Total: $307,200$ FLOPs

**Total per frame:**
$$2,560 + 409,600 + 3,686,400 + 1,638,400 + 307,200 = 6,044,160 \approx 6M \text{ FLOPs}$$

**At 100 fps (10ms hop):**
$$6M \times 100 = 600M \text{ FLOPs/s} = 0.6 \text{ GFLOPS} \approx 0.5 \text{ GMAC/s}$$

(Matches stated complexity)

---

## 14. Summary of Dimension Flow

```
Input Audio (2 mics + ref):
  [B, T] × 3

↓ STFT

Spectrograms:
  [B, K, 161] × 3

↓ Pre-processing

Features:
  [B, K, 2, 320]

↓ Rotation Layer

Rotated Features:
  [B, K, 8, 640]

↓ Concat with Delayed Ref

Combined:
  [B, K, 8, 1280]

↓ BiLSTM

Temporal Features:
  [B, K, 8, 320]

↓ Dual LSTM Branches

Branch 1: [B, K, 8, 320]
Branch 2: [B, K, 8, 320]

↓ FC Layers

FC1 out: [B, K, 8, 640]
FC2 out: [B, K, 8, 320]

↓ Combine + Filter

Mask:
  [B, K, 161]

↓ Apply Mask

Clean Spectrum:
  [B, K, 161]

↓ ISTFT

Output Audio:
  [B, T]
```

---

## 15. Key Takeaways

1. **Rotation Layer** = Learned beamforming (spatial mixing)
2. **BiLSTM** = Temporal context (past + future)
3. **Dual Branches** = Task decomposition (EC + NS)
4. **FC Layers** = Non-linear mapping to mask space
5. **Adaptive Filtering** = Frequency-selective suppression

**Why SGN Works:**
- Combines spatial (multi-mic) and temporal (LSTM) processing
- Learns optimal features for noise suppression
- Adapts to non-stationary noise and echo
- Efficient: Only 5.5M params, 0.5 GMAC/s

---

*End of Mathematical Formulation*
