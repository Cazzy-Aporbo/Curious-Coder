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

## What this repository is

A focused, continually-improving collection of **clear, rigorous explanations** of algorithms I use to analyze complex systems—especially in **biological and medical** contexts where data are noisy, high-dimensional, and full of edge cases.  
I care about mathematics. You will see it: definitions first, assumptions stated, and trade-offs made explicit. But everything is written to be **approachable**—you should not need to fight the notation to understand the idea. This progress is a continual work in progress, so check back for more updates. 

This is **not** a tutorial farm, a code dump, or a catalog of buzzwords. It’s a place to understand *how* an algorithm works, *why* it’s appropriate, and *when* to move on to a neighboring method.

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

