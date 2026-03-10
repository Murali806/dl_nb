# Essential Papers for Diffusion Models

This curated reading list covers the foundational and state-of-the-art papers in diffusion models. Papers are organized by topic and difficulty level.

---

## 📚 Reading Strategy

1. **Start with surveys** to get the big picture
2. **Read foundational papers** in chronological order
3. **Implement as you read** - code helps understanding
4. **Focus on intuition first**, then dive into math
5. **Take notes** and create your own summaries

---

## 🌟 Must-Read Papers (Core Curriculum)

### Week 1-2: Foundations

#### 1. Denoising Diffusion Probabilistic Models (DDPM)
**Authors**: Jonathan Ho, Ajay Jain, Pieter Abbeel  
**Year**: 2020  
**Link**: https://arxiv.org/abs/2006.11239  
**Difficulty**: ⭐⭐⭐

**Why Read**: This is THE foundational paper for modern diffusion models.

**Key Contributions**:
- Simplified training objective
- Connection to denoising score matching
- Practical implementation details

**Reading Notes**:
- Focus on Section 2 (Background) and Section 3 (Diffusion Models)
- The simplified loss (Equation 14) is crucial
- Algorithm 1 and 2 show training and sampling

---

#### 2. Estimation of Non-Normalized Statistical Models by Score Matching
**Authors**: Aapo Hyvärinen  
**Year**: 2005  
**Link**: https://jmlr.org/papers/v6/hyvarinen05a.html  
**Difficulty**: ⭐⭐⭐⭐

**Why Read**: Introduces score matching, foundational for score-based models.

**Key Contributions**:
- Score matching objective
- Avoids computing partition function
- Tractable training for energy-based models

**Reading Notes**:
- Theorem 1 is the key result
- Focus on intuition: matching gradients instead of densities
- Skip proofs on first read

---

#### 3. A Connection Between Score Matching and Denoising Autoencoders
**Authors**: Pascal Vincent  
**Year**: 2011  
**Link**: https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf  
**Difficulty**: ⭐⭐⭐

**Why Read**: Bridges denoising and score matching.

**Key Contributions**:
- Denoising score matching
- Computational efficiency
- Connection to autoencoders

**Reading Notes**:
- Section 3 shows the key equivalence
- Explains why adding noise helps
- Practical implications for training

---

### Week 3: Score-Based Models

#### 4. Generative Modeling by Estimating Gradients of the Data Distribution
**Authors**: Yang Song, Stefano Ermon  
**Year**: 2019  
**Link**: https://arxiv.org/abs/1907.05600  
**Difficulty**: ⭐⭐⭐⭐

**Why Read**: Introduces Noise Conditional Score Networks (NCSN).

**Key Contributions**:
- Multiple noise scales
- Annealed Langevin dynamics
- Addresses manifold hypothesis

**Reading Notes**:
- Section 3.2 explains the manifold problem
- Algorithm 1 shows annealed sampling
- Figure 2 visualizes the process

---

#### 5. Score-Based Generative Modeling through Stochastic Differential Equations
**Authors**: Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, et al.  
**Year**: 2021  
**Link**: https://arxiv.org/abs/2011.13456  
**Difficulty**: ⭐⭐⭐⭐⭐

**Why Read**: Unifies discrete and continuous-time diffusion.

**Key Contributions**:
- SDE formulation of diffusion
- Reverse-time SDE
- Probability flow ODE
- Unified framework

**Reading Notes**:
- Section 3 introduces the SDE framework
- Theorem 1 (reverse-time SDE) is crucial
- Section 5 shows the ODE connection
- This is dense - read multiple times

---

### Week 3-4: Flow Models

#### 6. Flow Matching for Generative Modeling
**Authors**: Yaron Lipman, Ricky T. Q. Chen, et al.  
**Year**: 2023  
**Link**: https://arxiv.org/abs/2210.02747  
**Difficulty**: ⭐⭐⭐⭐

**Why Read**: Modern alternative to diffusion training.

**Key Contributions**:
- Conditional flow matching
- Simpler training objective
- Optimal transport paths

**Reading Notes**:
- Section 3 introduces flow matching
- Algorithm 1 is remarkably simple
- Compare with DDPM training

---

#### 7. Neural Ordinary Differential Equations
**Authors**: Ricky T. Q. Chen, et al.  
**Year**: 2018  
**Link**: https://arxiv.org/abs/1806.07366  
**Difficulty**: ⭐⭐⭐⭐

**Why Read**: Foundation for continuous normalizing flows.

**Key Contributions**:
- Continuous-depth networks
- Adjoint method for backprop
- Memory-efficient training

**Reading Notes**:
- Section 2 introduces Neural ODEs
- Section 3 shows continuous normalizing flows
- Implementation details in appendix

---

### Week 4: Applications

#### 8. High-Resolution Image Synthesis with Latent Diffusion Models
**Authors**: Robin Rombach, et al.  
**Year**: 2022  
**Link**: https://arxiv.org/abs/2112.10752  
**Difficulty**: ⭐⭐⭐

**Why Read**: This is Stable Diffusion!

**Key Contributions**:
- Latent space diffusion
- Cross-attention for conditioning
- Practical high-resolution generation

**Reading Notes**:
- Section 3 explains the architecture
- Figure 3 shows the full pipeline
- Ablation studies in Section 4

---

#### 9. Classifier-Free Diffusion Guidance
**Authors**: Jonathan Ho, Tim Salimans  
**Year**: 2022  
**Link**: https://arxiv.org/abs/2207.12598  
**Difficulty**: ⭐⭐

**Why Read**: Essential technique for conditional generation.

**Key Contributions**:
- Guidance without separate classifier
- Improved sample quality
- Simple implementation

**Reading Notes**:
- Very short paper (4 pages)
- Equation 4 is the key formula
- Easy to implement

---

## 📖 Supplementary Papers

### Variational Autoencoders (Background)

#### Auto-Encoding Variational Bayes
**Authors**: Diederik P. Kingma, Max Welling  
**Year**: 2013  
**Link**: https://arxiv.org/abs/1312.6114  
**Difficulty**: ⭐⭐⭐

**Why Read**: VAEs are closely related to diffusion models.

---

### Improved Sampling

#### Denoising Diffusion Implicit Models (DDIM)
**Authors**: Jiaming Song, Chenlin Meng, Stefano Ermon  
**Year**: 2020  
**Link**: https://arxiv.org/abs/2010.02502  
**Difficulty**: ⭐⭐⭐

**Why Read**: Faster sampling with deterministic process.

**Key Contributions**:
- Non-Markovian diffusion
- Deterministic sampling
- 10-50x speedup

---

#### DPM-Solver: Fast ODE Solver for Diffusion Models
**Authors**: Cheng Lu, et al.  
**Year**: 2022  
**Link**: https://arxiv.org/abs/2206.00927  
**Difficulty**: ⭐⭐⭐⭐

**Why Read**: State-of-the-art fast sampling.

---

### Advanced Techniques

#### Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)
**Authors**: Lvmin Zhang, Maneesh Agrawala  
**Year**: 2023  
**Link**: https://arxiv.org/abs/2302.05543  
**Difficulty**: ⭐⭐⭐

**Why Read**: Spatial control for generation.

---

#### LoRA: Low-Rank Adaptation of Large Language Models
**Authors**: Edward J. Hu, et al.  
**Year**: 2021  
**Link**: https://arxiv.org/abs/2106.09685  
**Difficulty**: ⭐⭐

**Why Read**: Efficient fine-tuning technique.

---

## 🔬 Advanced Research (Optional)

### Consistency Models
**Authors**: Yang Song, et al.  
**Year**: 2023  
**Link**: https://arxiv.org/abs/2303.01469  
**Difficulty**: ⭐⭐⭐⭐⭐

**Why Read**: One-step generation.

---

### Diffusion Models for Video Generation
**Authors**: Various  
**Year**: 2022-2024  
**Difficulty**: ⭐⭐⭐⭐

Papers to explore:
- Video Diffusion Models (Ho et al., 2022)
- Imagen Video (Ho et al., 2022)
- Make-A-Video (Singer et al., 2022)

---

### Diffusion for Audio
**Authors**: Various  
**Year**: 2021-2024  
**Difficulty**: ⭐⭐⭐

Papers to explore:
- DiffWave (Kong et al., 2021)
- WaveGrad (Chen et al., 2021)
- Noise2Music (Huang et al., 2023)

---

## 📊 Survey Papers

### Understanding Diffusion Models: A Unified Perspective
**Authors**: Calvin Luo  
**Year**: 2022  
**Link**: https://arxiv.org/abs/2208.11970  
**Difficulty**: ⭐⭐⭐

**Why Read**: Excellent overview connecting all perspectives.

**Reading Notes**:
- Read this FIRST for big picture
- Connects ELBO, score matching, and SDEs
- Clear mathematical exposition

---

### Diffusion Models: A Comprehensive Survey
**Authors**: Ling Yang, et al.  
**Year**: 2023  
**Link**: https://arxiv.org/abs/2209.00796  
**Difficulty**: ⭐⭐

**Why Read**: Comprehensive overview of applications.

---

## 📅 Reading Schedule

### Week 1
- [ ] Understanding Diffusion Models (Survey)
- [ ] DDPM (Ho et al., 2020)
- [ ] VAE paper (background)

### Week 2
- [ ] Score Matching (Hyvärinen, 2005)
- [ ] Denoising Score Matching (Vincent, 2011)
- [ ] DDPM (re-read with new perspective)

### Week 3
- [ ] NCSN (Song & Ermon, 2019)
- [ ] Score-Based SDEs (Song et al., 2021)
- [ ] Neural ODEs (Chen et al., 2018)
- [ ] Flow Matching (Lipman et al., 2023)

### Week 4
- [ ] Latent Diffusion (Rombach et al., 2022)
- [ ] DDIM (Song et al., 2020)
- [ ] Classifier-Free Guidance (Ho & Salimans, 2022)
- [ ] ControlNet (Zhang & Agrawala, 2023)

---

## 💡 Reading Tips

### First Pass (30 min)
1. Read abstract and introduction
2. Look at figures and captions
3. Read conclusion
4. Skim section headings

### Second Pass (1-2 hours)
1. Read carefully, skip proofs
2. Understand main contributions
3. Note key equations
4. Try to explain to yourself

### Third Pass (2-4 hours)
1. Work through derivations
2. Implement key algorithms
3. Reproduce experiments
4. Write summary

### Taking Notes
- **Main Idea**: One sentence summary
- **Key Contributions**: Bullet points
- **Important Equations**: Write them out
- **Questions**: What's unclear?
- **Connections**: How does this relate to other papers?

---

## 🔗 Additional Resources

### Blogs
- **Yang Song's Blog**: https://yang-song.net/blog/
- **Lilian Weng's Blog**: https://lilianweng.github.io/
- **Hugging Face Blog**: https://huggingface.co/blog

### Video Lectures
- **Vizuara's Course**: Principles of Diffusion Models (YouTube)
- **Pieter Abbeel's Lectures**: Deep Unsupervised Learning
- **Stanford CS236**: Deep Generative Models

### Code Repositories
- **Hugging Face Diffusers**: https://github.com/huggingface/diffusers
- **OpenAI Guided Diffusion**: https://github.com/openai/guided-diffusion
- **Yang Song's Score-Based Models**: https://github.com/yang-song/score_sde_pytorch

---

## 📝 Paper Summary Template

Use this template for your notes:

```markdown
# Paper Title

**Authors**: 
**Year**: 
**Link**: 
**Read Date**: 

## One-Sentence Summary


## Key Contributions
1. 
2. 
3. 

## Main Idea


## Important Equations


## Strengths


## Limitations


## Questions


## Connections to Other Work


## Implementation Notes


## Rating: ⭐⭐⭐⭐⭐
```

---

Happy Reading! 📚
