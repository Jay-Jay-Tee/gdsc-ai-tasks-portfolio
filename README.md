# 🧠 Machine Learning Portfolio  
**GDSC AI/ML Induction Tasks**  
Joshua Jacob Thomas — B.Tech CSE, NIT Calicut  

---

## Overview

This website is an interactive Machine Learning portfolio built to demonstrate:

- Practical ML model implementation  
- Clear understanding of mathematical foundations  
- From-scratch neural network components  
- Thoughtful feature engineering  
- Clean, intuitive UI with real-time visualizations  

Every section includes:
- Model description  
- Performance metrics  
- Core mathematical idea  
- Interactive demo  

---

# 01 — Placement Prediction

**Model:** Logistic Regression  
**Dataset Size:** 10,000 students  
**Test Accuracy:** 80.8%  

This model predicts whether a student will be placed based on features such as:

- CGPA  
- Aptitude score  
- Soft skills  
- Internships  
- Projects  
- SSC marks  
- Placement training  
- Extracurriculars  

### Core Idea

Logistic Regression models binary outcomes using:

\[
p = \frac{1}{1 + e^{-z}}
\]

Where:

\[
z = w_1x_1 + w_2x_2 + ... + b
\]

The sigmoid function converts a weighted sum into a probability.

### Interactive Demo

- Modify student profile
- See probability update instantly
- View feature contribution bars
- Observe model confidence

---

# 02 — Sleep & Academic Performance

**Model:** Linear Regression  
**R² Score:** 0.334  
**MAE:** 0.529  

This regression model predicts academic performance based on:

- Sleep hours  
- Stress  
- Fatigue  
- Missed classes  
- Caffeine  
- Screen time  

### Key Insight

Daytime consequences of poor sleep (fatigue, missed classes, stress) are stronger predictors than sleep duration alone.

### Interactive Demo

- Adjust sliders
- View predicted score (1–10)
- See dominant drivers
- Read contextual interpretation

---

# 03 — Hostel Wi-Fi Performance

**Model:** Random Forest Regressor  
**R² Score:** 0.759  
**MAE:** 508 KB/s  

Since the dataset lacked a direct speed column, a proxy metric was engineered:

```
transfer_rate = total_bytes / session_seconds
```

### Feature Engineering Includes

- Cyclic time encoding (23:00 ≈ 00:00)
- Log-scaled transfer volumes
- Usage behavior ratios
- Chronological train/test split

### Interactive Demo

- Select hour of day
- Select session type
- View predicted throughput
- Detect congestion vs idle behavior
- Get smart recommendations

---

# 04 — MNIST Digit Classifier

**Architecture:** 784 → 128 → ReLU → 10  
**Optimizer:** Adam  
**Epochs:** 5  
**Test Accuracy:** 97.16%  

Trained in PyTorch on 60,000 handwritten digit images.

### Preprocessing Pipeline

- Crop bounding box
- Resize to 20×20
- Center in 28×28 frame
- Normalize using:

```
Normalize((0.5,), (0.5,))
```

### Interactive Demo

- Draw a digit (0–9)
- See 28×28 processed input
- Watch animated forward pass
- View probability distribution across classes

This demo illustrates distribution shift — browser handwriting differs from MNIST training data.

---

# 05 — MicroGrad (From Scratch Autograd Engine)

Implemented automatic differentiation entirely in Python — no PyTorch, no NumPy.

### Core Component

`Value` class:
- Wraps scalar values
- Records operations
- Stores parents in computation graph
- Applies chain rule during `.backward()`

### Chain Rule

If:

```
e = f(d)
d = g(a)
```

Then:

```
∂e/∂a = (∂e/∂d) × (∂d/∂a)
```

### Includes

- Neuron class  
- Layer class  
- MLP class  
- XOR training example  

### Interactive Visualizations

- Computation graph with live gradients  
- Gradient descent stepper  
- tanh vs ReLU comparison  
- XOR decision boundary evolution  

---

# 06 — GPT-2 Transformer (Theory + Modules)

Decoder-only Transformer implemented as standalone Python modules.

### Architecture Parameters

- 12 Attention Heads  
- 768 d_model  
- Causal Masking  
- Pre-LayerNorm  

### Key Components

1. Token + Positional Embedding  
2. Multi-Head Self-Attention  
3. Residual Connections  
4. Feed-Forward Network (768 → 3072 → 768)  
5. LayerNorm  
6. Final projection to vocabulary  

### Attention Formula

\[
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d}}\right)V
\]

### Interactive Visualizations

- Clickable architecture blocks  
- Positional encoding heatmap  
- Self-attention flow between words  
- Next-word prediction probabilities  

---

# Technical Highlights

- Pure HTML + CSS + JavaScript frontend  
- Canvas-based neural network visualizations  
- DevicePixelRatio-aware rendering  
- Custom animation loops  
- Seeded randomness for reproducibility  
- Clean modular architecture  

---

# What This Portfolio Demonstrates

- End-to-end ML workflow  
- Mathematical understanding of models  
- Implementation depth (from scratch components)  
- Feature engineering intuition  
- Ability to explain complex systems clearly  
- Strong UI + systems thinking  

---

# Final Note

This portfolio is designed not just to show that models work —  
but to show that I understand *why* they work.

Every section connects:
- Data  
- Math  
- Code  
- Visualization  
- Interpretation  

---

🧠 Built with curiosity.  
📊 Backed by math.  
⚙️ Implemented from scratch where it matters.  
🚀 Designed to be explored.
