# Diffusion Models - Complete Curriculum Guide

## 📖 How to Use This Curriculum

This is a **self-paced, comprehensive learning path** for mastering diffusion models. The curriculum is designed to be completed in **1 month** (approximately 4-6 hours per day), but you can adjust the pace based on your schedule and background.

---

## 🎯 Learning Philosophy

This curriculum follows these principles:

1. **Build from First Principles**: Start with mathematical foundations
2. **Multiple Perspectives**: Learn three unified views (Variational, Score-Based, Flow-Based)
3. **Theory + Practice**: Every concept includes both mathematical derivations and code
4. **No Black Boxes**: Implement everything from scratch to truly understand
5. **Visual Learning**: Text-based diagrams and visualizations throughout
6. **Progressive Complexity**: Start simple, gradually increase difficulty

---

## 📅 Recommended Study Schedule

### Week 1: Foundations (Days 1-7)

**Day 1-2: Module 1 - Prerequisites**
- Morning: Probability distributions, Gaussian theory
- Afternoon: Interactive visualizations, exercises
- Evening: Bayes' theorem, conditional distributions

**Day 3-4: Module 2 - Generative Models Landscape**
- Morning: Overview of generative modeling
- Afternoon: Taxonomy (VAEs, GANs, Flows, etc.)
- Evening: Comparative analysis, when to use what

**Day 5-7: Module 3 - Diffusion Intuition**
- Morning: Physics analogy, forward process
- Afternoon: Reverse process intuition
- Evening: Complete Batman example walkthrough

**Weekend Review**: Consolidate Week 1 concepts

---

### Week 2: Core Theory (Days 8-14)

**Day 8-10: Module 4 - DDPM**
- Day 8: Forward process mathematics
- Day 9: Reverse process, reparameterization
- Day 10: Training objective, MNIST implementation

**Day 11-12: Module 5 - Score Matching**
- Day 11: Energy-based models, score functions
- Day 12: Denoising score matching, Langevin dynamics

**Day 13-14: Module 6 - SDE Framework**
- Day 13: Forward/reverse SDEs, Fokker-Planck
- Day 14: Probability flow ODE, implementation

**Weekend Review**: Implement a simple diffusion model

---

### Week 3: Advanced Theory (Days 15-21)

**Day 15-17: Module 7 - Normalizing Flows**
- Day 15: Flow basics, change of variables
- Day 16: Neural ODEs, flow matching
- Day 17: Rectified flows, implementation

**Day 18-19: Module 8 - NCSN**
- Day 18: Multiple noise levels, annealed Langevin
- Day 19: CIFAR-10 implementation

**Day 20-21: Module 9 - VAE Connection**
- Day 20: VAE theory, ELBO derivation
- Day 21: VAE implementation, connection to diffusion

**Weekend Review**: Compare all three perspectives

---

### Week 4: Applications (Days 22-30)

**Day 22-23: Module 10 - U-Net Architecture**
- Day 22: U-Net theory, time embeddings
- Day 23: Complete implementation

**Day 24-25: Module 11 - Sampling Algorithms**
- Day 24: DDPM, DDIM, ODE solvers
- Day 25: Classifier-free guidance

**Day 26-28: Module 12 - Applications**
- Day 26: Image generation project
- Day 27: Audio generation project
- Day 28: Video generation project

**Day 29-30: Module 13 - State-of-the-Art**
- Day 29: Stable Diffusion, ControlNet, LoRA
- Day 30: Recent research, future directions

**Final Review**: Complete all projects, review notes

---

## 🎓 Study Strategies

### For Each Module

1. **Read Theory First** (30-40 min)
   - Read the .md files carefully
   - Draw diagrams on paper
   - Write down questions

2. **Run Code Examples** (20-30 min)
   - Open Jupyter notebooks
   - Run cells step-by-step
   - Modify parameters to see effects

3. **Complete Exercises** (30-40 min)
   - Attempt all exercises
   - Check solutions
   - Understand mistakes

4. **Review and Reflect** (15-20 min)
   - Summarize key concepts
   - Connect to previous modules
   - Note areas for further study

### Daily Routine

```
Morning Session (2-3 hours)
├── Theory reading (60-90 min)
├── Break (15 min)
└── Code implementation (60-90 min)

Afternoon Session (2-3 hours)
├── Exercises (60-90 min)
├── Break (15 min)
└── Project work (60-90 min)

Evening (Optional)
└── Review and consolidation (30-60 min)
```

---

## 📊 Progress Tracking

Use this checklist to track your progress:

### Week 1: Foundations
- [ ] Module 1: Prerequisites (Days 1-2)
  - [ ] 1.1 Probability distributions
  - [ ] 1.2 Visualizations
  - [ ] 1.3 Bayes' theorem
  - [ ] 1.4 Markov chains
  - [ ] 1.5 Stochastic processes
  - [ ] 1.6 Differential equations

- [ ] Module 2: Generative Models (Days 3-4)
  - [ ] 2.1 Overview
  - [ ] 2.2 Taxonomy
  - [ ] 2.3 Comparative analysis
  - [ ] 2.4 Why diffusion models

- [ ] Module 3: Diffusion Intuition (Days 5-7)
  - [ ] 3.1 Physics analogy
  - [ ] 3.2 Brownian motion
  - [ ] 3.3 Forward process
  - [ ] 3.4 Reverse process
  - [ ] 3.5 Batman example

### Week 2: Core Theory
- [ ] Module 4: DDPM (Days 8-10)
- [ ] Module 5: Score Matching (Days 11-12)
- [ ] Module 6: SDE Framework (Days 13-14)

### Week 3: Advanced Theory
- [ ] Module 7: Normalizing Flows (Days 15-17)
- [ ] Module 8: NCSN (Days 18-19)
- [ ] Module 9: VAE Connection (Days 20-21)

### Week 4: Applications
- [ ] Module 10: U-Net (Days 22-23)
- [ ] Module 11: Sampling (Days 24-25)
- [ ] Module 12: Applications (Days 26-28)
- [ ] Module 13: State-of-the-Art (Days 29-30)

---

## 🎯 Learning Milestones

### After Week 1
You should be able to:
- Explain what diffusion models are
- Understand the forward and reverse processes
- Recognize the three theoretical perspectives

### After Week 2
You should be able to:
- Derive the DDPM training objective
- Implement score matching
- Understand the SDE formulation

### After Week 3
You should be able to:
- Compare diffusion, score-based, and flow-based models
- Implement NCSN with multiple noise levels
- Understand the VAE connection

### After Week 4
You should be able to:
- Build a complete diffusion model from scratch
- Generate images, audio, and video
- Understand state-of-the-art techniques

---

## 💡 Tips for Success

### 1. Active Learning
- **Don't just read**: Implement concepts yourself
- **Modify code**: Change parameters, see what breaks
- **Derive formulas**: Work through math on paper

### 2. Build Intuition
- **Visualize**: Draw diagrams for every concept
- **Analogies**: Connect to physical processes
- **Simplify**: Start with 1D examples before high-dimensional

### 3. Connect Concepts
- **Map relationships**: How does DDPM relate to score matching?
- **Unified view**: See the three perspectives as one framework
- **Applications**: Think about real-world use cases

### 4. Manage Difficulty
- **Stuck on math?** Skip to code, return later
- **Code not working?** Review theory first
- **Overwhelmed?** Take a break, review fundamentals

### 5. Community Learning
- **Discuss**: Explain concepts to others
- **Share**: Post your implementations
- **Ask**: Don't hesitate to seek help

---

## 📚 Supplementary Resources

### Essential Papers (Read in Order)

1. **Week 1-2**:
   - DDPM (Ho et al., 2020)
   - Score Matching (Hyvärinen, 2005)
   - Denoising Score Matching (Vincent, 2010)

2. **Week 3**:
   - NCSN (Song & Ermon, 2019)
   - Score-Based SDEs (Song et al., 2021)
   - Flow Matching (Lipman et al., 2023)

3. **Week 4**:
   - Latent Diffusion (Rombach et al., 2022)
   - Classifier-Free Guidance (Ho & Salimans, 2022)

### Video Lectures
- Vizuara's "Principles of Diffusion Models" (YouTube)
- Yang Song's talks on score-based models
- Hugging Face diffusion course

### Books
- "Deep Learning" by Goodfellow et al. (Chapters 16-20)
- "Pattern Recognition and Machine Learning" by Bishop

---

## 🔧 Technical Setup

### Required Software
```bash
# Core packages
pip install torch torchvision numpy matplotlib scipy tqdm jupyter

# For visualizations
pip install seaborn plotly ipywidgets

# For audio/video
pip install librosa soundfile opencv-python

# For experiments
pip install wandb tensorboard
```

### Hardware Recommendations
- **Minimum**: CPU, 8GB RAM
- **Recommended**: GPU (NVIDIA), 16GB RAM
- **Optimal**: GPU with 24GB+ VRAM for large models

### Development Environment
- **IDE**: VS Code with Python extension
- **Notebooks**: Jupyter Lab or VS Code notebooks
- **Version Control**: Git for tracking progress

---

## 🎓 Assessment

### Self-Assessment Questions

**After Week 1:**
1. What is the forward diffusion process?
2. How does the reverse process work?
3. What are the three perspectives on diffusion models?

**After Week 2:**
4. Derive the DDPM training objective
5. Explain the score function
6. What is an SDE?

**After Week 3:**
7. How do flow models differ from diffusion models?
8. Why use multiple noise levels in NCSN?
9. How are VAEs related to diffusion models?

**After Week 4:**
10. Implement a diffusion model from scratch
11. Generate samples using different samplers
12. Explain Stable Diffusion architecture

---

## 🚀 Next Steps After Completion

1. **Research**: Read recent papers (2024-2026)
2. **Projects**: Build your own applications
3. **Contribute**: Open-source implementations
4. **Specialize**: Focus on specific domains (medical imaging, audio, etc.)
5. **Teach**: Write blog posts, create tutorials

---

## 📞 Getting Help

If you're stuck:
1. Review the theory files in the module
2. Check the implementation notebooks
3. Refer to the original papers
4. Search for related discussions online
5. Take a break and return with fresh eyes

---

## 🎉 Congratulations!

By completing this curriculum, you'll have:
- ✅ Deep understanding of diffusion models
- ✅ Ability to implement from scratch
- ✅ Knowledge of three theoretical perspectives
- ✅ Practical experience with real applications
- ✅ Foundation for cutting-edge research

**Ready to begin?** Start with `Week1_Foundations/Module1_Prerequisites/`

Good luck on your learning journey! 🚀
