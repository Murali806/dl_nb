# Diffusion Models - Complete Learning Curriculum

A comprehensive, month-long journey into Diffusion Models covering theory, mathematics, and practical implementations.

## 📚 Course Overview

This curriculum provides a deep understanding of Diffusion Models from three unified perspectives:
1. **Variational View** (DDPM) - Discrete-time, ELBO optimization
2. **Score-Based View** (NCSN, SDE) - Score matching, Langevin dynamics
3. **Flow-Based View** (CNF, Flow Matching) - Continuous normalizing flows, ODEs

## 🎯 Learning Objectives

By the end of this course, you will:
- Understand the mathematical foundations of diffusion models
- Implement diffusion models from scratch (no black boxes)
- Master three theoretical perspectives on generative modeling
- Build practical applications for images, audio, and video
- Understand state-of-the-art techniques (Stable Diffusion, ControlNet, etc.)

## 📋 Prerequisites

- **Mathematics**: Calculus (gradients, chain rule), Linear Algebra (matrices, eigenvalues)
- **Probability**: Basic probability theory (will be refreshed in Module 1)
- **Deep Learning**: Neural networks, backpropagation, PyTorch/TensorFlow
- **Python**: Comfortable with NumPy, Matplotlib

## 🗓️ Course Structure (4 Weeks)

### Week 1: Mathematical Foundations & Core Intuition
- Module 1: Prerequisites Refresher (Days 1-2)
- Module 2: Generative Models Landscape (Days 3-4)
- Module 3: Diffusion Process Intuition (Days 5-7)

### Week 2: DDPM, Score-Based Models & SDE Framework
- Module 4: Denoising Diffusion Probabilistic Models (Days 8-10)
- Module 5: Energy-Based Models & Score Matching (Days 11-12)
- Module 6: SDE Framework - Physical Intuition (Days 13-14)

### Week 3: Flow Models, NCSN & Advanced Theory
- Module 7: Continuous Normalizing Flows (Days 15-17)
- Module 8: Noise Conditioned Score Networks (Days 18-19)
- Module 9: Variational Autoencoders Connection (Days 20-21)

### Week 4: Implementation, Applications & State-of-the-Art
- Module 10: U-Net Architecture & Implementation (Days 22-23)
- Module 11: Sampling Algorithms (Days 24-25)
- Module 12: Multi-Modal Applications (Days 26-28)
- Module 13: State-of-the-Art & Research (Days 29-30)

## 📁 Directory Structure

```
F_DiffusionModels/
├── Week1_Foundations/
│   ├── Module1_Prerequisites/
│   ├── Module2_GenerativeModels/
│   └── Module3_DiffusionIntuition/
├── Week2_DDPM_ScoreBased_SDE/
│   ├── Module4_DDPM/
│   ├── Module5_ScoreMatching/
│   └── Module6_SDE_Framework/
├── Week3_Flows_NCSN_VAE/
│   ├── Module7_NormalizingFlows/
│   ├── Module8_NCSN/
│   └── Module9_VAE/
├── Week4_Implementation_Applications/
│   ├── Module10_UNet/
│   ├── Module11_Sampling/
│   ├── Module12_Applications/
│   └── Module13_StateOfTheArt/
├── Projects/
│   ├── Image_Generation/
│   ├── Audio_Generation/
│   └── Video_Generation/
└── Resources/
    ├── Papers/
    ├── Notebooks/
    └── References/
```

## 🎓 Learning Approach

Each module follows this structure:
1. **Theory Files (.md)**: Mathematical derivations with ASCII/text diagrams
2. **Implementation Files (.ipynb)**: Code with extensive comments and visualizations
3. **Exercises**: Practice problems to test understanding
4. **Projects**: Real-world applications

## 🔑 Key Features

✅ **60+ files** covering theory, math, and implementation  
✅ **Text-based diagrams** throughout (ASCII art, Mermaid)  
✅ **Complete mathematical derivations** with step-by-step explanations  
✅ **Three major projects** (image, audio, video)  
✅ **From-scratch implementations** (no black boxes)  
✅ **All three theoretical perspectives** covered  
✅ **Modern techniques** (CFG, DDIM, Flow Matching, ControlNet)  
✅ **Physical intuition** (SDEs, Brownian motion, thermodynamics)  

## 📖 Recommended Reading Order

1. Start with `Week1_Foundations/Module1_Prerequisites/` to refresh fundamentals
2. Follow the weekly structure sequentially
3. Complete exercises before moving to the next module
4. Work on projects after completing relevant theory modules

## 🛠️ Setup Instructions

```bash
# Install required packages
pip install torch torchvision numpy matplotlib scipy tqdm jupyter

# Optional: For audio/video generation
pip install librosa soundfile opencv-python

# Clone or navigate to this directory
cd F_DiffusionModels
```

## 📚 Key Papers Covered

- **DDPM**: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- **Score Matching**: Estimation of Non-Normalized Statistical Models (Hyvärinen, 2005)
- **Denoising Score Matching**: A Connection with Denoising Autoencoders (Vincent, 2010)
- **NCSN**: Generative Modeling by Estimating Gradients (Song & Ermon, 2019)
- **SDE Framework**: Score-Based Generative Modeling through SDEs (Song et al., 2021)
- **Flow Matching**: Flow Matching for Generative Modeling (Lipman et al., 2023)
- **Stable Diffusion**: High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)

## 🎯 Projects

### Project 1: Image Generation (CIFAR-10)
Build a complete diffusion model for image generation from scratch.

### Project 2: Audio Generation
Generate music and speech using diffusion models on spectrograms.

### Project 3: Video Generation
Create short video clips with temporal consistency.

## 🤝 Contributing

This is a learning curriculum. Feel free to:
- Add your own notes and insights
- Create additional exercises
- Share your project implementations
- Suggest improvements

## 📞 Support

For questions or discussions:
- Review the theory files carefully
- Check the implementation notebooks for practical examples
- Refer to the original papers in the Resources section

## 🙏 Acknowledgments

This curriculum is inspired by:
- Vizuara's "Principles of Diffusion Models" course
- Yang Song's blog and papers
- The broader diffusion models research community

## 📝 License

Educational use only. Please cite original papers when using concepts or code.

---

**Ready to start?** Begin with `Week1_Foundations/Module1_Prerequisites/1_1_probability_distributions_refresher.md`

Happy Learning! 🚀
