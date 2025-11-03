<div align="center">

# Curious Coder — Mathematical & Scientific Algorithms, Explained

[![Mathematics](https://img.shields.io/badge/Mathematics-Core%20Reasoning-FFE4E1?style=for-the-badge)]()
[![Algorithms](https://img.shields.io/badge/Algorithms-Scientific%20Exploration-E6E6FA?style=for-the-badge)]()
[![Concepts](https://img.shields.io/badge/Concepts-Deep%20Analysis-F0F8FF?style=for-the-badge)]()
[![Biological ML](https://img.shields.io/badge/Biological%20ML-Models%20%26%20Systems-FFF0F5?style=for-the-badge)]()

</div>

<div align="center">

![separator](https://img.shields.io/badge/-FFE4E1?style=flat-square&color=FFE4E1)
![separator](https://img.shields.io/badge/-E6E6FA?style=flat-square&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat-square&color=F0F8FF)
![separator](https://img.shields.io/badge/-FFF0F5?style=flat-square&color=FFF0F5)
![separator](https://img.shields.io/badge/-FAFAD2?style=flat-square&color=FAFAD2)

</div>

<div align="center">

[Advanced PyTorch Systems Architecture](https://htmlpreview.github.io/?https://github.com/Cazzy-Aporbo/Curious-Coder/blob/main/Explore-PyTorch/pytorch_systems_documentation.html)

<div align="center">
<sub> Deep Learning Implementations Bridging Healthcare Intelligence and Industrial Automation - Adaptive Gradient Optimization Through Statistical Flow Monitoring</sub>
</div>
</picture>

## What this repository is

A focused, continually-improving collection of **clear, rigorous explanations** of algorithms I use to analyze complex systems—especially in **biological and medical** contexts where data are noisy, high-dimensional, and full of edge cases.  
I care about mathematics. You will see it: definitions first, assumptions stated, and trade-offs made explicit. But everything is written to be **approachable**—you should not need to fight the notation to understand the idea. This progress is a continual work in progress, so check back for more updates. 

This is **not** a tutorial farm, a code dump, or a catalog of buzzwords. It’s a place to understand *how* an algorithm works, *why* it’s appropriate, and *when* to move on to a neighboring method.

<div align="center">

<div align="center">
  
[AI Ethics: Historical Context, Contemporary Challenges, and Future Implications](https://htmlpreview.github.io/?https://github.com/Cazzy-Aporbo/Curious-Coder/blob/main/explore_stuff/ai_ethics_comprehensive.html)
</div>

<div align="center">

![separator](https://img.shields.io/badge/-E6E6FA?style=flat&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat&color=F0F8FF)
![separator](https://img.shields.io/badge/-FFF0F5?style=flat&color=FFF0F5)

</div>

## How to use these notes

1. **Start with context.** Each entry begins with a short statement of what the method is good at and the assumptions it quietly relies on.  
2. **Scan the formulation.** A compact, precise formulation follows (objective, loss, constraints). No heavy derivations unless they matter for usage.  
3. **Check “When to prefer / avoid.”** Practical decision criteria so you can move quickly.  
4. **Look sideways.** Every method lists a few close neighbors (e.g., PCA ↔ ICA; K-means ↔ GMM/EM; Lasso ↔ Ridge/Elastic-Net).  
5. **Apply with discipline.** Metrics and diagnostics are included so results don’t become anecdotes.

<div align="center">

![separator](https://img.shields.io/badge/-FFE4E1?style=flat-square&color=FFE4E1)
![separator](https://img.shields.io/badge/-E6E6FA?style=flat-square&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat-square&color=F0F8FF)

</div>

## Reading guide for each algorithm entry

| Section | Purpose | What you’ll see |
|---|---|---|
| **Intent** | One-sentence “what problem does this solve?” | Clear problem statement |
| **Formulation** | The core mathematics without ceremony | Objective/loss, constraints, variables |
| **Assumptions** | The part that breaks silently if ignored | IID, linearity, separability, smoothness, stationarity, etc. |
| **When to prefer** | Practical conditions where it excels | Data size/shape, noise regime, feature types |
| **When to avoid** | Failure modes and edge cases | Multicollinearity, non-convexity traps, class imbalance, etc. |
| **Neighbor methods** | Closely related alternatives | Swap-ins worth testing |
| **Diagnostics** | How to know it worked | Residual checks, calibration, stability, uncertainty |
| **Biological applications** | Where this has bite | Imaging, genomics, EHR, epidemiology, physiology |

<div align="center">

![separator](https://img.shields.io/badge/-FFF0F5?style=flat&color=FFF0F5)
![separator](https://img.shields.io/badge/-FAFAD2?style=flat&color=FAFAD2)
![separator](https://img.shields.io/badge/-FFE4B5?style=flat&color=FFE4B5)

</div>

## A compact decision orientation

<table>
<tr style="background-color:#FFE4E1">
<td><strong>Data shape</strong></td>
<td><strong>Often a good starting point</strong></td>
<td><strong>If that stalls, try</strong></td>
</tr>
<tr>
<td>Tabular, small→medium</td>
<td>Regularized GLMs; GBDT (XGB/LGBM/CatBoost)</td>
<td>Nonlinear SVM; simple MLP</td>
</tr>
<tr style="background-color:#F0F8FF">
<td>Sequences / longitudinal</td>
<td>Transformers (baseline); temporal CNNs</td>
<td>RNN/LSTM/GRU; HMM/Kalman for structure</td>
</tr>
<tr>
<td>Spatial grids / images</td>
<td>CNNs; U-Net for segmentation</td>
<td>Vision Transformers; diffusion for generation</td>
</tr>
<tr style="background-color:#FFF0F5">
<td>Graphs / molecular</td>
<td>Message-passing GNNs</td>
<td>Graph Transformers; spectral methods</td>
</tr>
<tr>
<td>Very high-dimensional, low labels</td>
<td>Contrastive pretraining; masked modeling</td>
<td>Autoencoders/VAEs; self-distillation</td>
</tr>
<tr style="background-color:#FAFAD2">
<td>Uncertainty is critical</td>
<td>Bayesian GLMs; calibrated ensembles</td>
<td>Bayesian deep learning; conformal prediction</td>
</tr>
</table>

<div align="center">

![separator](https://img.shields.io/badge/-E6E6FA?style=flat-square&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat-square&color=F0F8FF)
![separator](https://img.shields.io/badge/-FFF0F5?style=flat-square&color=FFF0F5)

</div>

## Why this curation exists

I enjoy the structure and honesty of mathematics. In applied work—especially in biology and medicine—results improve when the **assumptions are explicit**, the **algorithms are chosen for the data** (not for fashion), and the **limits are respected**. These notes are written to make that process fast, transparent, and repeatable.

You will see a mix of:
- **Core methods** (GLMs, trees/ensembles, SVMs, PCA, clustering)  
- **Advanced learning** (Transformers, diffusion, contrastive/self-supervised learning, GNNs)  
- **Frontier topics** (flows, neural ODEs, causal estimation, federated/multitask learning)  
- **Biological ML** where signal is subtle and mechanisms matter (U-Net families, AlphaFold-style structure models, EHR sequence models)

<div align="center">

![separator](https://img.shields.io/badge/-FFE4E1?style=flat&color=FFE4E1)
![separator](https://img.shields.io/badge/-E6E6FA?style=flat&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat&color=F0F8FF)

</div>

## Template for an algorithm entry

Use this structure to keep entries consistent and quick to scan.

<div align="center">

# Curious Coder — Algorithms Index

[![Mathematics](https://img.shields.io/badge/Mathematics-Core%20Foundation-FFE4E1?style=for-the-badge)]()
[![Algorithms](https://img.shields.io/badge/Algorithms-Scientific%20Exploration-E6E6FA?style=for-the-badge)]()
[![Concepts](https://img.shields.io/badge/Concepts-Deep%20Analysis-F0F8FF?style=for-the-badge)]()

</div>

<div align="center">

![separator](https://img.shields.io/badge/-FFE4E1?style=flat-square&color=FFE4E1)
![separator](https://img.shields.io/badge/-E6E6FA?style=flat-square&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat-square&color=F0F8FF)
![separator](https://img.shields.io/badge/-FFF0F5?style=flat-square&color=FFF0F5)
![separator](https://img.shields.io/badge/-FAFAD2?style=flat-square&color=FAFAD2)

</div>

This section collects **core, advanced, and frontier** methods I study and use. Entries focus on:  
- **Used for** (primary purpose)  
- **When** (practical decision criteria)  
- **Similar** (adjacent methods to consider)

The intent is clarity: rigorous enough for research, direct enough for practice.

---

## Core Algorithms for DS/AI/ML Engineers

<table>
<tr style="background-color:#FFE4E1">
<td><strong>Method</strong></td>
<td><strong>Used for</strong></td>
<td><strong>When</strong></td>
<td><strong>Similar</strong></td>
</tr>
<tr>
<td><strong>Linear Regression</strong></td>
<td>Linear relationship to continuous target</td>
<td>Interpretable baseline; trend testing</td>
<td>Ridge, Lasso</td>
</tr>
<tr style="background-color:#F0F8FF">
<td><strong>Logistic Regression</strong></td>
<td>Binary classification with calibrated probs</td>
<td>Probabilities + interpretability; baseline</td>
<td>Probit; Softmax Regression (multiclass)</td>
</tr>
<tr>
<td><strong>Decision Trees</strong></td>
<td>Rule-based classification/regression</td>
<td>Nonlinear patterns; mixed feature types</td>
<td>Random Forest; Gradient Boosted Trees</td>
</tr>
<tr style="background-color:#FFF0F5">
<td><strong>Random Forest</strong></td>
<td>Ensemble of trees (bagging)</td>
<td>Robust tabular performance; low overfit</td>
<td>ExtraTrees; GBM</td>
</tr>
<tr>
<td><strong>Gradient Boosting (XGB/LGBM/CatBoost)</strong></td>
<td>Boosted trees for strong tabular accuracy</td>
<td>State-of-the-art on many tabular tasks</td>
<td>AdaBoost; Random Forest</td>
</tr>
<tr style="background-color:#FAFAD2">
<td><strong>K-Nearest Neighbors</strong></td>
<td>Instance-based classification/regression</td>
<td>Simple nonparametric baseline; low-dim data</td>
<td>KDE; RBF-kernel SVM</td>
</tr>
<tr>
<td><strong>Support Vector Machines</strong></td>
<td>Max-margin classification/regression</td>
<td>Medium-sized data; robustness to outliers</td>
<td>Logistic (linear); NNs (nonlinear)</td>
</tr>
<tr style="background-color:#F0F8FF">
<td><strong>Naïve Bayes</strong></td>
<td>Generative classification with independence</td>
<td>Text; very high-dimensional sparse features</td>
<td>Logistic Regression; LDA</td>
</tr>
<tr>
<td><strong>PCA</strong></td>
<td>Orthogonal dimensionality reduction</td>
<td>Compression; de-correlation; visualization</td>
<td>SVD; ICA</td>
</tr>
<tr style="background-color:#FFF0F5">
<td><strong>K-Means</strong></td>
<td>Hard-partition clustering</td>
<td>Fast baseline clustering</td>
<td>GMM (soft clusters); DBSCAN</td>
</tr>
<tr>
<td><strong>Expectation–Maximization</strong></td>
<td>Latent-variable MLE (e.g., GMM)</td>
<td>Overlapping distributions; soft assignments</td>
<td>K-Means; Variational Inference</td>
</tr>
<tr style="background-color:#FFE4E1">
<td><strong>Apriori / FP-Growth</strong></td>
<td>Association rule mining</td>
<td>Frequent itemsets; basket analysis</td>
<td>Eclat</td>
</tr>
<tr>
<td><strong>Dynamic Programming</strong></td>
<td>Optimal substructure optimization</td>
<td>Overlapping subproblems</td>
<td>Greedy (approximate)</td>
</tr>
<tr style="background-color:#F0F8FF">
<td><strong>Gradient Descent</strong></td>
<td>Continuous optimization</td>
<td>Differentiable models; large-scale training</td>
<td>SGD; Adam; RMSProp</td>
</tr>
<tr>
<td><strong>Neural Networks (MLP)</strong></td>
<td>Flexible nonlinear mapping</td>
<td>Complex patterns; large data</td>
<td>CNN; RNN</td>
</tr>
</table>

<div align="center">

![separator](https://img.shields.io/badge/-E6E6FA?style=flat&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat&color=F0F8FF)
![separator](https://img.shields.io/badge/-FFF0F5?style=flat&color=FFF0F5)

</div>

## Advanced Algorithms

<table>
<tr style="background-color:#E6E6FA">
<td><strong>Method</strong></td>
<td><strong>Used for</strong></td>
<td><strong>When</strong></td>
<td><strong>Similar</strong></td>
</tr>
<tr>
<td><strong>CNNs</strong></td>
<td>Spatial representation learning</td>
<td>Vision; local structure</td>
<td>ViTs; Graph Convolutions</td>
</tr>
<tr style="background-color:#F0F8FF">
<td><strong>RNN / LSTM / GRU</strong></td>
<td>Sequence modeling with memory</td>
<td>Time series; language; speech</td>
<td>Transformers; Temporal CNNs</td>
</tr>
<tr>
<td><strong>Transformers</strong></td>
<td>Attention-based sequence modeling</td>
<td>Language; multimodal; long context</td>
<td>RNNs; Attentional CNNs</td>
</tr>
<tr style="background-color:#FFF0F5">
<td><strong>Autoencoders</strong></td>
<td>Compression; anomaly detection</td>
<td>Representation learning</td>
<td>PCA; VAE</td>
</tr>
<tr>
<td><strong>Variational Autoencoders</strong></td>
<td>Probabilistic generative modeling</td>
<td>Latent structure + generation</td>
<td>GANs; Normalizing Flows</td>
</tr>
<tr style="background-color:#FAFAD2">
<td><strong>GANs</strong></td>
<td>Adversarial generative modeling</td>
<td>Realistic synthesis; augmentation</td>
<td>VAEs; Diffusion</td>
</tr>
<tr>
<td><strong>Diffusion Models</strong></td>
<td>Score-based generation</td>
<td>Diversity + stability</td>
<td>GANs; Score Matching</td>
</tr>
<tr style="background-color:#F0F8FF">
<td><strong>Reinforcement Learning (Q-Learning)</strong></td>
<td>Value-based decision policies</td>
<td>Discrete actions; tabular/compact states</td>
<td>Policy Gradient; DQN</td>
</tr>
<tr>
<td><strong>Policy Gradient / Actor–Critic</strong></td>
<td>Direct policy optimization</td>
<td>Continuous/high-dim actions</td>
<td>REINFORCE; PPO</td>
</tr>
<tr style="background-color:#FFF0F5">
<td><strong>K-Means++ / Advanced Clustering</strong></td>
<td>Improved initialization</td>
<td>Reduce bad local minima</td>
<td>Spectral; GMM; DBSCAN</td>
</tr>
<tr>
<td><strong>DBSCAN</strong></td>
<td>Density-based clustering with noise</td>
<td>Arbitrary shapes; outliers</td>
<td>OPTICS; HDBSCAN</td>
</tr>
<tr style="background-color:#FAFAD2">
<td><strong>Spectral Clustering</strong></td>
<td>Graph-Laplacian embeddings</td>
<td>Manifold/complex geometry</td>
<td>GNNs; Laplacian Eigenmaps</td>
</tr>
<tr>
<td><strong>HMMs</strong></td>
<td>Probabilistic sequence models</td>
<td>Hidden state dynamics</td>
<td>Kalman Filters; CRF</td>
</tr>
<tr style="background-color:#F0F8FF">
<td><strong>Kalman Filters</strong></td>
<td>State estimation with noise</td>
<td>Real-time tracking</td>
<td>Particle Filters; HMM</td>
</tr>
<tr>
<td><strong>Graph Neural Networks</strong></td>
<td>Learning on graphs</td>
<td>Relational structure > features</td>
<td>CNN (grids); Graph Transformers</td>
</tr>
<tr style="background-color:#FFF0F5">
<td><strong>MCMC</strong></td>
<td>Sampling complex posteriors</td>
<td>Bayesian inference</td>
<td>Variational Inference; HMC</td>
</tr>
<tr>
<td><strong>GBDT (XGB/LGBM/CatBoost)</strong></td>
<td>Top performance on tabular data</td>
<td>Accuracy with moderate compute</td>
<td>Random Forest; AdaBoost</td>
</tr>
<tr style="background-color:#FAFAD2">
<td><strong>Recommenders (MF: SVD/ALS)</strong></td>
<td>Collaborative filtering</td>
<td>Sparse user–item matrices</td>
<td>NCF; Graph-based Recsys</td>
</tr>
</table>

<div align="center">

![separator](https://img.shields.io/badge/-FFE4E1?style=flat&color=FFE4E1)
![separator](https://img.shields.io/badge/-E6E6FA?style=flat&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat&color=F0F8FF)

</div>

## Frontier / Expert Topics

<table>
<tr style="background-color:#FFF0F5">
<td><strong>Method</strong></td>
<td><strong>Used for</strong></td>
<td><strong>When</strong></td>
<td><strong>Similar</strong></td>
</tr>
<tr>
<td><strong>Normalizing Flows</strong></td>
<td>Exact-likelihood generative modeling</td>
<td>Need density + sampling</td>
<td>VAE; Diffusion</td>
</tr>
<tr style="background-color:#F0F8FF">
<td><strong>Diffusion Transformers</strong></td>
<td>Diffusion + Transformer backbones</td>
<td>Scaled multimodal generation</td>
<td>DDPM; GANs</td>
</tr>
<tr>
<td><strong>Neural ODEs</strong></td>
<td>Continuous-time dynamics</td>
<td>Physics/biology/finance signals</td>
<td>RNNs; SDEs</td>
</tr>
<tr style="background-color:#E6E6FA">
<td><strong>Graph Transformers / Message Passing</strong></td>
<td>Expressive graph learning</td>
<td>Complex relational structure</td>
<td>Spectral GNNs</td>
</tr>
<tr>
<td><strong>Neural Tangent Kernel</strong></td>
<td>Infinite-width NN theory</td>
<td>Generalization & convergence study</td>
<td>Kernels; GPs</td>
</tr>
<tr style="background-color:#FAFAD2">
<td><strong>Meta-Learning (MAML, ProtoNets)</strong></td>
<td>Rapid adaptation</td>
<td>Few-shot; transfer</td>
<td>Bayesian Opt; Fine-tuning</td>
</tr>
<tr>
<td><strong>Bayesian Deep Learning</strong></td>
<td>Uncertainty-aware deep models</td>
<td>High-stakes decisions</td>
<td>MCMC; VI</td>
</tr>
<tr style="background-color:#FFF0F5">
<td><strong>Causal Inference (DoWhy, EconML)</strong></td>
<td>Estimating causal effects</td>
<td>Policy/health interventions</td>
<td>IV; Propensity Scores</td>
</tr>
<tr>
<td><strong>Federated Learning (FedAvg, FedProx)</strong></td>
<td>Privacy-preserving distributed training</td>
<td>Decentralized sensitive data</td>
<td>Distributed SGD; DP</td>
</tr>
<tr style="background-color:#F0F8FF">
<td><strong>Contrastive Learning (SimCLR, CLIP)</strong></td>
<td>Self-supervised representations</td>
<td>Limited labels; large raw data</td>
<td>Autoencoders; Distillation</td>
</tr>
<tr>
<td><strong>Energy-Based Models</strong></td>
<td>Unnormalized density modeling</td>
<td>Intractable partition functions</td>
<td>Boltzmann Machines</td>
</tr>
<tr style="background-color:#E6E6FA">
<td><strong>RL — PPO / SAC / DDPG</strong></td>
<td>Scalable policy optimization</td>
<td>Continuous/high-dim control</td>
<td>REINFORCE; Q-Learning</td>
</tr>
<tr>
<td><strong>Multi-Agent RL</strong></td>
<td>Interacting agents</td>
<td>Markets; autonomy; swarms</td>
<td>Game Theory; Single-agent RL</td>
</tr>
<tr style="background-color:#FAFAD2">
<td><strong>Mixture-of-Experts / Sparse Transformers</strong></td>
<td>Efficient scaling</td>
<td>Conditional computation</td>
<td>Standard Transformers; LoRA</td>
</tr>
<tr>
<td><strong>Quantum ML (VQE, QAOA)</strong></td>
<td>Quantum optimization/chemistry</td>
<td>NISQ-era research</td>
<td>Classical Variational Methods</td>
</tr>
<tr style="background-color:#FFF0F5">
<td><strong>Neurosymbolic AI</strong></td>
<td>Neural perception + symbolic reasoning</td>
<td>Tasks needing both pattern and logic</td>
<td>Knowledge Graphs</td>
</tr>
<tr>
<td><strong>Masked Self-Supervision (BERT, MAE)</strong></td>
<td>Representation pretraining</td>
<td>Large unlabeled corpora</td>
<td>Contrastive; Autoencoders</td>
</tr>
<tr style="background-color:#F0F8FF">
<td><strong>Prompting / Few-Shot Adaptation</strong></td>
<td>LLM task transfer without updates</td>
<td>Generalization to unseen tasks</td>
<td>Meta-Learning; Instruction Tuning</td>
</tr>
<tr>
<td><strong>Curriculum Learning</strong></td>
<td>Staged difficulty schedules</td>
<td>Unstable/complex training</td>
<td>RL Shaping; Augmentation</td>
</tr>
<tr style="background-color:#E6E6FA">
<td><strong>Neural Architecture Search</strong></td>
<td>Automated model design</td>
<td>Edge constraints; task specificity</td>
<td>Bayesian/Hyperparameter Opt</td>
</tr>
</table>

<div align="center">

![separator](https://img.shields.io/badge/-FFE4E1?style=flat&color=FFE4E1)
![separator](https://img.shields.io/badge/-E6E6FA?style=flat&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat&color=F0F8FF)

</div>

## Medical & Biological AI (Selected)

<table>
<tr style="background-color:#F0F8FF">
<td><strong>Area</strong></td>
<td><strong>Model</strong></td>
<td><strong>Used for</strong></td>
<td><strong>Similar</strong></td>
</tr>
<tr>
<td>Neuro / Brain Imaging</td>
<td>U-Net, V-Net, nnU-Net; BrainAGE; GLM (SPM/FSL)</td>
<td>Segmentation; age prediction; activation modeling</td>
<td>SegNet; DeepLab</td>
</tr>
<tr style="background-color:#FFF0F5">
<td>Radiology</td>
<td>Radiomics+ML; DeepMedic; CheXNet</td>
<td>Quantitative features; lesion segmentation; X-ray Dx</td>
<td>ResNet/EfficientNet variants</td>
</tr>
<tr>
<td>Genomics</td>
<td>DeepSEA; AlphaFold; SpliceAI; DeepCpG/EpiDeep</td>
<td>Variant effect; protein structure; splicing; epigenetics</td>
<td>Basset; Basenji; RoseTTAFold</td>
</tr>
<tr style="background-color:#FAFAD2">
<td>Cardiology</td>
<td>ECGNet/DeepECG; EchoNet</td>
<td>Arrhythmia classification; EF estimation</td>
<td>1D CNNs; video CNNs</td>
</tr>
<tr>
<td>Pathology</td>
<td>HoVer-Net; CLAM (MIL); tile-based classifiers</td>
<td>Nucleus segmentation; WSI classification</td>
<td>Mask R-CNN; MIL variants</td>
</tr>
<tr style="background-color:#E6E6FA">
<td>Population & EHR</td>
<td>RETAIN; DeepPatient; BEHRT</td>
<td>Longitudinal risk; multi-outcome prediction</td>
<td>RNNs; Transformers for EHR</td>
</tr>
<tr>
<td>Epidemiology</td>
<td>Compartmental (SIR/SEIR/SEIRD); ABM</td>
<td>Spread modeling; intervention simulation</td>
<td>System dynamics</td>
</tr>
<tr style="background-color:#F0F8FF">
<td>Multimodal Medical AI</td>
<td>MedCLIP; BioViL; Bio/ClinicalBERT</td>
<td>Image–text alignment; biomedical NLP</td>
<td>CLIP; BERT</td>
</tr>
</table>

---

### Why these families
- **Clinical Core**: U-Net, RETAIN, SIR — established workhorses.  
- **Research-Grade**: AlphaFold, DeepSEA, SpliceAI — molecular scale.  
- **Practice-Changing**: CheXNet, EchoNet, CLAM — real clinical impact.  
- **Emerging Frontier**: MedCLIP, BEHRT, BioBERT — multimodal and longitudinal.

<div align="center">

![separator](https://img.shields.io/badge/-FFE4E1?style=flat-square&color=FFE4E1)
![separator](https://img.shields.io/badge/-E6E6FA?style=flat-square&color=E6E6FA)
![separator](https://img.shields.io/badge/-F0F8FF?style=flat-square&color=F0F8FF)

</div>
<div align="center">

# Library Reference & Imports (Curated)

[![Core Python](https://img.shields.io/badge/Core-Python-FFE4E1?style=for-the-badge)](#core-python)
[![Data](https://img.shields.io/badge/Data-Handling-E6E6FA?style=for-the-badge)](#data-handling)
[![Viz](https://img.shields.io/badge/Visualization-Stacks-F0F8FF?style=for-the-badge)](#visualization)
[![ML/AI](https://img.shields.io/badge/Machine%20Learning-Tooling-FFF0F5?style=for-the-badge)](#machine-learning--ai)
[![LLMs](https://img.shields.io/badge/Transformers-LLMs-FAFAD2?style=for-the-badge)](#advanced-ai--transformers--llms)

</div>

<div align="center">

[![sep](https://img.shields.io/badge/-FFE4E1?style=flat-square&color=FFE4E1)](#quick-index)
[![sep](https://img.shields.io/badge/-E6E6FA?style=flat-square&color=E6E6FA)](#quick-index)
[![sep](https://img.shields.io/badge/-F0F8FF?style=flat-square&color=F0F8FF)](#quick-index)
[![sep](https://img.shields.io/badge/-FFF0F5?style=flat-square&color=FFF0F5)](#quick-index)
[![sep](https://img.shields.io/badge/-FAFAD2?style=flat-square&color=FAFAD2)](#quick-index)
[![sep](https://img.shields.io/badge/-E0FFFF?style=flat-square&color=E0FFFF)](#quick-index)

</div>

## Quick Index

- [Core Python](#core-python)
- [Data Handling](#data-handling)
- [Visualization](#visualization)
- [Machine Learning / AI](#machine-learning--ai)
- [Math, Statistics, SciPy](#math-statistics-scipy)
- [NLP / Text](#nlp--text)
- [Utilities & Workflow](#utilities--workflow)
- [Data I/O](#data-io)
- [Visualization Add-ons](#visualization-add-ons)
- [Advanced Data & Big Data](#advanced-data--big-data)
- [Deep Learning & GPU](#deep-learning--gpu)
- [Advanced AI / Transformers / LLMs](#advanced-ai--transformers--llms)
- [Advanced Visualization & Dashboards](#advanced-visualization--dashboards)
- [Statistics, Bayesian, Probabilistic](#statistics-bayesian-probabilistic)
- [Optimization & Math](#optimization--math)
- [Graphs, Knowledge, Advanced Data](#graphs-knowledge-advanced-data)
- [Advanced NLP / Text](#advanced-nlp--text)
- [Advanced Utilities & Parallelism](#advanced-utilities--parallelism)
- [Computer Vision & Image/Video](#computer-vision--imagevideo)
- [Geospatial & Maps](#geospatial--maps)
- [Ultra / Rare Imports (HPC, Research, Frontier)](#ultra--rare-imports-hpc-research-frontier)

<div align="center">

[![sep](https://img.shields.io/badge/-D4E4FC?style=flat-square&color=D4E4FC)](#quick-index)
[![sep](https://img.shields.io/badge/-E7F3E7?style=flat-square&color=E7F3E7)](#quick-index)
[![sep](https://img.shields.io/badge/-FFE5CC?style=flat-square&color=FFE5CC)](#quick-index)
[![sep](https://img.shields.io/badge/-FADADD?style=flat-square&color=FADADD)](#quick-index)
[![sep](https://img.shields.io/badge/-FFF4E6?style=flat-square&color=FFF4E6)](#quick-index)

</div>

## Core Python

<table>
<tr style="background-color:#FFE4E1"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>os, sys, Path</td><td>Filesystem, environment, paths</td><td>Portable path handling via <code>pathlib</code></td></tr>
<tr style="background-color:#F0F8FF"><td>re, json, csv</td><td>Regex, serialization, CSV I/O</td><td>Use <code>jsonlines</code> for large JSONL</td></tr>
<tr><td>math, random, time, datetime</td><td>Math, RNG, timing</td><td><code>dt</code> alias for concise timestamps</td></tr>
<tr style="background-color:#FFF0F5"><td>Counter, defaultdict</td><td>Counting, default dicts</td><td>Efficient tallies and grouping</td></tr>
</table>

<div align="center">[![Back to index](https://img.shields.io/badge/Back_to_Index-Click-FFE4E1?style=flat-square)](#quick-index)</div>

## Data Handling

<table>
<tr style="background-color:#E6E6FA"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>numpy</td><td>Arrays, vectorized math</td><td>Foundation for most stacks</td></tr>
<tr style="background-color:#FAFAD2"><td>pandas</td><td>Tabular data</td><td>Wide ecosystem; groupby, time series</td></tr>
<tr><td>pyarrow</td><td>Columnar memory, parquet</td><td>High-perf interchange with pandas</td></tr>
<tr style="background-color:#F0F8FF"><td>polars</td><td>Fast DataFrame (Rust engine)</td><td>Laziness, speed on medium/large data</td></tr>
</table>

<div align="center">

## Visualization

<table>
<tr style="background-color:#FFE4E1"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>matplotlib, seaborn</td><td>Static plotting</td><td>Seaborn for statistical charts</td></tr>
<tr style="background-color:#F0F8FF"><td>plotly.express, graph_objects</td><td>Interactive plots</td><td>Browser-ready, tooltips, zoom</td></tr>
<tr><td>altair</td><td>Declarative grammar</td><td>Readable specs; Vega-Lite backend</td></tr>
</table>

<div align="center">

## Machine Learning / AI

<table>
<tr style="background-color:#FFF0F5"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>scikit-learn</td><td>Classical ML, metrics, preprocessing</td><td>Baselines, pipelines, grid search</td></tr>
<tr style="background-color:#F0F8FF"><td>XGBoost, LightGBM, CatBoost</td><td>Gradient boosting</td><td>SOTA tabular; categorical support (CatBoost)</td></tr>
<tr><td>PyTorch</td><td>Deep learning</td><td>Define-by-run, custom training loops</td></tr>
<tr style="background-color:#FAFAD2"><td>TensorFlow / Keras</td><td>Deep learning</td><td>High-level layers, production tooling</td></tr>
<tr><td>transformers</td><td>LLMs, transfer learning</td><td>Tokenizers, pipelines, model zoo</td></tr>
</table>

<div align="center">

## Math, Statistics, SciPy

<table>
<tr style="background-color:#E6E6FA"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>scipy (stats, signal, optimize, integrate)</td><td>Scientific routines</td><td>Tests, filters, solvers</td></tr>
<tr style="background-color:#F0F8FF"><td>statistics</td><td>Built-in descriptive stats</td><td>Lightweight helpers</td></tr>
<tr><td>sympy</td><td>Symbolic math</td><td>Derivations, simplifications</td></tr>
</table>

<div align="center">

## NLP / Text

<table>
<tr style="background-color:#FFE4E1"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>nltk, spacy</td><td>Tokenization, parsing</td><td>spaCy for pipelines; NLTK utilities</td></tr>
<tr style="background-color:#F0F8FF"><td>gensim</td><td>Word2Vec, LDA</td><td>Topic modeling and embeddings</td></tr>
<tr><td>wordcloud</td><td>Visual summaries</td><td>Exploratory visuals</td></tr>
</table>

<div align="center">

## Utilities & Workflow

<table>
<tr style="background-color:#FFF0F5"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>tqdm</td><td>Progress bars</td><td>Notebook-friendly via <code>tqdm.notebook</code></td></tr>
<tr style="background-color:#F0F8FF"><td>logging, warnings</td><td>Diagnostics</td><td>Set handlers, suppress noise selectively</td></tr>
<tr><td>joblib, pickle</td><td>Model I/O</td><td>Persist artifacts; mind security</td></tr>
</table>

<div align="center">

## Data I/O

<table>
<tr style="background-color:#FAFAD2"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>csv, sqlite3</td><td>Flat files, local DB</td><td>Good for lightweight pipelines</td></tr>
<tr style="background-color:#F0F8FF"><td>h5py</td><td>HDF5 storage</td><td>Large arrays, hierarchical datasets</td></tr>
<tr><td>requests</td><td>HTTP APIs</td><td>Timeouts, retries, backoff</td></tr>
</table>

<div align="center">

## Visualization Add-ons

<table>
<tr style="background-color:#E6E6FA"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>networkx</td><td>Graphs/networks</td><td>Topology, centrality measures</td></tr>
<tr style="background-color:#F0F8FF"><td>geopandas, folium</td><td>Geospatial viz</td><td>Interactive maps and overlays</td></tr>
</table>

<div align="center">

## Advanced Data & Big Data

<table>
<tr style="background-color:#FFE4E1"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>dask.dataframe</td><td>Out-of-core pandas</td><td>Parallelize wide workflows</td></tr>
<tr style="background-color:#F0F8FF"><td>vaex, modin</td><td>Lazy or distributed DataFrame</td><td>Scale on single machine or cluster</td></tr>
<tr><td>pyspark</td><td>Spark API</td><td>Cluster compute for very large data</td></tr>
</table>

<div align="center">

## Deep Learning & GPU

<table>
<tr style="background-color:#FFF0F5"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>torch, nn, optim, F</td><td>Core training</td><td>Custom loops, modules</td></tr>
<tr style="background-color:#F0F8FF"><td>torch.distributed, TensorBoard</td><td>Multi-GPU, logging</td><td>DDP for scale-out</td></tr>
<tr><td>tensorflow, keras</td><td>DL stacks</td><td>High-level layers and fit loops</td></tr>
<tr style="background-color:#FAFAD2"><td>jax, jnp, flax, optax</td><td>JIT DL, functional NN</td><td>Fast grad, pure functions</td></tr>
</table>

<div align="center">

## Advanced AI / Transformers / LLMs

<table>
<tr style="background-color:#E6E6FA"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>transformers</td><td>LLMs, pipelines</td><td>Text, vision, audio models</td></tr>
<tr style="background-color:#F0F8FF"><td>peft, bitsandbytes</td><td>Efficient finetuning, quantization</td><td>LoRA, 8-bit/4-bit training</td></tr>
<tr><td>accelerate, sentence_transformers</td><td>Distributed, embeddings</td><td>Multi-GPU orchestration, retrieval</td></tr>
</table>

<div align="center">

## Advanced Visualization & Dashboards

<table>
<tr style="background-color:#FFE4E1"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>bokeh, holoviews, hvplot</td><td>Interactive viz stacks</td><td>Linked brushing, high-level APIs</td></tr>
<tr style="background-color:#F0F8FF"><td>panel, dash, streamlit</td><td>Dashboards/apps</td><td>From notebook to app quickly</td></tr>
<tr><td>pyvis, pyvista</td><td>Networks, 3D</td><td>Explorable graphs and volumes</td></tr>
</table>

<div align="center">

## Statistics, Bayesian, Probabilistic

<table>
<tr style="background-color:#FFF0F5"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>pymc, arviz</td><td>Bayesian inference, diagnostics</td><td>Priors, posteriors, PPC</td></tr>
<tr style="background-color:#F0F8FF"><td>statsmodels</td><td>Regression, time series</td><td>GLM, ARIMA families</td></tr>
<tr><td>lifelines, prophet</td><td>Survival, forecasting</td><td>Kaplan–Meier; components/trends</td></tr>
</table>

<div align="center">

## Optimization & Math

<table>
<tr style="background-color:#FAFAD2"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>cvxpy, pulp, ortools</td><td>Convex, LP/MIP, routing</td><td>Solvers and modeling</td></tr>
<tr style="background-color:#F0F8FF"><td>numba</td><td>JIT acceleration</td><td>Speed up Python loops</td></tr>
<tr><td>sympy</td><td>Symbolic math</td><td>Closed forms, derivations</td></tr>
</table>

<div align="center">

## Graphs, Knowledge, Advanced Data

<table>
<tr style="background-color:#E6E6FA"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>networkx, neo4j</td><td>Graph analysis, DB</td><td>Topology + graph stores</td></tr>
<tr style="background-color:#F0F8FF"><td>dgl, torch_geometric, stellargraph</td><td>Graph ML</td><td>Message passing, link prediction</td></tr>
</table>

<div align="center">

## Advanced NLP / Text

<table>
<tr style="background-color:#FFE4E1"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>stanza, flair</td><td>NLP pipelines, embeddings</td><td>Strong pretrained components</td></tr>
<tr style="background-color:#F0F8FF"><td>yake, textblob</td><td>Keywords, sentiment</td><td>Lightweight tasks</td></tr>
<tr><td>gensim LdaModel</td><td>Topic modeling</td><td>Classical LDA workflow</td></tr>
</table>

<div align="center">

## Advanced Utilities & Parallelism

<table>
<tr style="background-color:#FFF0F5"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>ray, joblib</td><td>Distributed, parallel pipelines</td><td>Scale compute across cores/nodes</td></tr>
<tr style="background-color:#F0F8FF"><td>ThreadPoolExecutor, ProcessPoolExecutor</td><td>Concurrency APIs</td><td>IO vs CPU bound tasks</td></tr>
</table>

<div align="center">

## Computer Vision & Image/Video

<table>
<tr style="background-color:#E6E6FA"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>opencv</td><td>Image/video processing</td><td>Transforms, codecs, tracking</td></tr>
<tr style="background-color:#F0F8FF"><td>mediapipe</td><td>Pose/gesture</td><td>Prebuilt inference graphs</td></tr>
<tr><td>albumentations, skimage</td><td>Augmentation, analysis</td><td>Training-ready pipelines</td></tr>
<tr style="background-color:#FAFAD2"><td>imageio, tifffile</td><td>I/O, large images</td><td>Microscopy, GeoTIFFs</td></tr>
</table>

<div align="center">

## Geospatial & Maps

<table>
<tr style="background-color:#FFE4E1"><td><strong>Library</strong></td><td><strong>Role</strong></td><td><strong>Notes</strong></td></tr>
<tr><td>geopandas, shapely</td><td>Geo tables, geometry ops</td><td>Buffers, intersections</td></tr>
<tr style="background-color:#F0F8FF"><td>rasterio, cartopy</td><td>Rasters, cartography</td><td>CRS management</td></tr>
<tr><td>folium, contextily</td><td>Interactive maps, basemaps</td><td>Tiles and layers</td></tr>
</table>

<div align="center">

## Ultra / Rare Imports (HPC, Research, Frontier)

<table>
<tr style="background-color:#FFF0F5"><td><strong>Area</strong></td><td><strong>Examples</strong></td><td><strong>Use</strong></td></tr>
<tr><td>HPC & GPU Kernels</td><td>triton, mpi4py, pycuda, pyopencl, numexpr</td><td>Custom kernels, multi-node, speed</td></tr>
<tr style="background-color:#F0F8FF"><td>Large-Scale Training</td><td>deepspeed, fairscale, megatron</td><td>Sharded models, parallelism</td></tr>
<tr><td>Probabilistic Programming</td><td>pyro, edward2, gpytorch</td><td>Bayesian deep learning, GPs</td></tr>
<tr style="background-color:#E6E6FA"><td>Causal ML</td><td>dowhy, econml, causalinference</td><td>Effects, policy evaluation</td></tr>
<tr><td>Science & Bio</td><td>biopython, deepchem, mdtraj, openmm</td><td>Genomics, chemistry, MD</td></tr>
<tr style="background-color:#FAFAD2"><td>Quantum</td><td>qiskit, cirq, pennylane, qutip</td><td>VQA, simulation</td></tr>
<tr><td>Advanced Viz</td><td>datashader, mayavi, k3d, fastplotlib</td><td>Huge data, 3D interactive</td></tr>
<tr style="background-color:#F0F8FF"><td>Privacy & Federated</td><td>opacus, tensorflow_privacy, syft</td><td>Differential privacy, FL</td></tr>
<tr><td>Infra & MLOps</td><td>prefect, dagster, kedro, mlflow, hydra, feast</td><td>Pipelines, tracking, configs</td></tr>
</table>

<div align="center">

[![sep](https://img.shields.io/badge/-FFE4E1?style=flat-square&color=FFE4E1)](#quick-index)
[![sep](https://img.shields.io/badge/-E6E6FA?style=flat-square&color=E6E6FA)](#quick-index)
[![sep](https://img.shields.io/badge/-F0F8FF?style=flat-square&color=F0F8FF)](#quick-index)
[![sep](https://img.shields.io/badge/-FFF0F5?style=flat-square&color=FFF0F5)](#quick-index)
[![sep](https://img.shields.io/badge/-FAFAD2?style=flat-square&color=FAFAD2)](#quick-index)
[![Back to index](https://img.shields.io/badge/Back_to_Index-Click-D4E4FC?style=flat-square)](#quick-index)

</div>

