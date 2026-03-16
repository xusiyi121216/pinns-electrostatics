# Physics-Informed Neural Networks for 2D Electrostatic Field Reconstruction and Permittivity Identification

Final project for [Course Name] — [University Name] — [Your Name]

## Table of Contents
- [Overview](#overview)
- [Mathematical Development](#mathematical-development)
- [PINN Methodology](#pinn-methodology)
- [Project Structure](#project-structure)
- [Reproducibility](#reproducibility)
- [Results](#results)
- [References](#references)

---

## Overview

This project applies **Physics-Informed Neural Networks (PINNs)** to a two-dimensional electrostatic problem. Unlike traditional numerical solvers (FEM, FDM), PINNs embed the governing PDE directly into the neural network loss function, requiring no mesh and naturally supporting both forward and inverse problems in a single framework.

**Two problems are addressed:**
- **Forward Problem**: Given boundary conditions, reconstruct the electric potential field φ(x,y) and validate against the analytical solution
- **Inverse Problem**: Given sparse potential observations at 20 sensor locations, simultaneously recover the field and identify an unknown permittivity ε

**Additional analyses:**
- Ablation study: effect of observation point count (5 to 50) on inverse problem accuracy
- Noise robustness: effect of Gaussian measurement noise (0% to 10%) on permittivity identification

---

## Mathematical Development

### From Maxwell's Equations to the Laplace Equation

In electrostatics, the electric field **E** is related to the scalar potential φ by:

$$\mathbf{E} = -\nabla\varphi$$

Gauss's law states:

$$\nabla \cdot \mathbf{D} = \rho_f$$

where **D** = ε**E** is the electric displacement field, ε is the permittivity, and ρ_f is the free charge density. In a source-free (ρ_f = 0), homogeneous (uniform ε) medium, substituting gives:

$$\nabla \cdot (\varepsilon \nabla\varphi) = 0 \quad \Rightarrow \quad \nabla^2\varphi = 0$$

This is the **Laplace equation**, the governing equation for this project.

### Problem Domain and Boundary Conditions

We solve on the unit square Ω = [0,1]²:

$$\frac{\partial^2\varphi}{\partial x^2} + \frac{\partial^2\varphi}{\partial y^2} = 0, \quad (x,y)\in[0,1]^2$$

| Boundary | Condition | Reason |
|----------|-----------|--------|
| Top (y=1) | φ(x,1) = sin(πx) | Smooth, zero at corners |
| Bottom (y=0) | φ = 0 | Grounded |
| Left (x=0) | φ = 0 | Grounded |
| Right (x=1) | φ = 0 | Grounded |

The top boundary uses sin(πx) rather than a constant (e.g., φ=1) to ensure continuity at the four corners. A constant top boundary would create jump discontinuities at (0,1) and (1,1), which are known to significantly degrade PINN accuracy.

### Analytical Solution

The exact closed-form solution is obtained by separation of variables:

$$\varphi(x,y) = \sin(\pi x) \cdot \frac{\sinh(\pi y)}{\sinh(\pi)}$$

This is used to validate the forward problem and to generate synthetic observation data for the inverse problem.

**Verification:** At (x,y) = (0.5, 0.5):

$$\varphi(0.5, 0.5) = \sin(0.5\pi) \cdot \frac{\sinh(0.5\pi)}{\sinh(\pi)} = 1.0 \times \frac{2.301}{11.549} \approx 0.199$$

### Inverse Problem Formulation

For the inverse problem, the permittivity ε is treated as unknown. The observed potential at sensor location (x_i, y_i) is modeled as:

$$\varphi_\text{obs}(x_i, y_i) = \varepsilon \cdot \varphi(x_i, y_i) + \eta_i$$

where η_i ~ N(0, σ²) is optional Gaussian measurement noise. The goal is to recover ε from {(x_i, y_i, φ_obs,i)} without any prior knowledge of ε. The true value is set to ε = 2.5; the initial guess is ε = 1.0.

---

## PINN Methodology

### Network Architecture

A fully-connected neural network approximates the potential field:
```
Input (x,y) → [Linear 2→50] → tanh → [Linear 50→50] → tanh
             → [Linear 50→50] → tanh → [Linear 50→50] → tanh
             → [Linear 50→1] → Output φ(x,y)
```

- **Activation function**: tanh — smooth and infinitely differentiable, required for computing second-order derivatives via autograd
- **Weight initialization**: Xavier normal — prevents vanishing/exploding gradients with tanh
- **Inverse problem**: ε is an additional `nn.Parameter` scalar optimized jointly with network weights

### Automatic Differentiation for PDE Residuals

PyTorch's autograd computes the Laplacian without finite differences:
```python
phi = model(x, y)
phi_x  = autograd.grad(phi, x, create_graph=True)[0]
phi_xx = autograd.grad(phi_x, x, create_graph=True)[0]
phi_y  = autograd.grad(phi, y, create_graph=True)[0]
phi_yy = autograd.grad(phi_y, y, create_graph=True)[0]
residual = phi_xx + phi_yy  # should equal 0 everywhere
```

### Loss Function

$$\mathcal{L} = \mathcal{L}_\text{pde} + 10\,\mathcal{L}_\text{bc} + 100\,\mathcal{L}_\text{data}$$

| Term | Formula | Points | Weight | Physical Meaning |
|------|---------|--------|--------|-----------------|
| L_pde | mean(residual²) | 3,000 random collocation points | 1 | Enforce Laplace equation in domain |
| L_bc | mean((φ_pred - φ_bc)²) | 200 boundary points | 10 | Enforce boundary conditions |
| L_data | mean((ε·φ_pred - φ_obs)²) | 20 observation points | 100 | Fit sparse sensor data (inverse only) |

The higher weight on L_bc (×10) ensures boundary conditions are satisfied accurately. The highest weight on L_data (×100) drives the inverse problem, as the data loss is the only signal for identifying ε.

### Two-Stage Training

| Stage | Optimizer | Duration | Purpose |
|-------|-----------|----------|---------|
| Stage 1 | Adam | 10,000 epochs, lr=1e-3 (halved every 2,000 steps) | Global exploration |
| Stage 2 | L-BFGS | 500 iterations, strong Wolfe line search | Local refinement using second-order information |

Adam handles the global landscape efficiently; L-BFGS exploits curvature information to converge to a sharper minimum.

---

## Project Structure
```
pinns-electrostatics/
├── model.py              # PINN network architecture (PINN class)
├── utils.py              # Analytical solution, data generation, plotting
├── forward_problem.py    # Forward problem: Adam + L-BFGS training and evaluation
├── inverse_problem.py    # Inverse problem: joint field + permittivity identification
├── analysis.py           # Ablation study + noise robustness analysis
├── figures/
│   ├── Forward_Problem_Adam.png      # Forward problem results (Adam)
│   ├── Forward_Problem_LBFGS.png     # Forward problem results (L-BFGS)
│   ├── loss_curve_adam.png           # Training loss curve
│   ├── inverse_problem.png           # Inverse problem convergence
│   ├── ablation_obs_points.png       # Ablation study results
│   └── noise_robustness.png          # Noise robustness results
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Reproducibility

### Requirements
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch>=2.0.0 numpy>=1.24.0 matplotlib>=3.7.0 scipy>=1.10.0
```

### Run
```bash
# Step 1: Forward problem (~3 min on GPU)
python forward_problem.py

# Step 2: Inverse problem (~3 min on GPU)
python inverse_problem.py

# Step 3: Ablation + noise analysis (~20 min on GPU)
python analysis.py
```

All random seeds are fixed (`torch.manual_seed(42)`, `np.random.seed(42)`) for full reproducibility. GPU (CUDA) is used automatically if available, otherwise falls back to CPU.

### Expected Output

| Script | Output Files |
|--------|-------------|
| forward_problem.py | Forward_Problem_Adam.png, Forward_Problem_LBFGS.png, loss_curve_adam.png, forward_model.pth, forward_model_final.pth |
| inverse_problem.py | inverse_problem.png, inverse_model.pth |
| analysis.py | ablation_obs_points.png, noise_robustness.png |

---

## Results

### Forward Problem

| Stage | L2 Relative Error |
|-------|------------------|
| Adam (10,000 epochs) | 0.53% |
| Adam + L-BFGS | **0.26%** |

### Inverse Problem

| Metric | Value |
|--------|-------|
| True ε | 2.5000 |
| Predicted ε | 2.5003 |
| Relative Error | **0.012%** |

### Ablation Study

| Observation Points | Predicted ε | Relative Error |
|-------------------|-------------|----------------|
| 5  | 2.8474 | 13.895% |
| 10 | 2.5050 | 0.199% |
| 20 | 2.4975 | 0.100% |
| 30 | 2.5028 | 0.112% |
| 50 | 2.5010 | 0.039% |

### Noise Robustness

| Noise Level | Predicted ε | Relative Error |
|-------------|-------------|----------------|
| 0%  | 2.4975 | 0.100% |
| 1%  | 2.5067 | 0.267% |
| 3%  | 2.5237 | 0.947% |
| 5%  | 2.5451 | 1.805% |
| 10% | 2.5999 | 3.996% |

---

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.

2. Baldan, M., Di Barba, P., & Lowther, D. A. (2023). Physics-informed neural networks for inverse electromagnetic problems. *IEEE Transactions on Magnetics*, 59(5), 1–5.

3. Lagaris, I. E., Likas, A., & Fotiadis, D. I. (1998). Artificial neural networks for solving ordinary and partial differential equations. *IEEE Transactions on Neural Networks*, 9(5), 987–1000.
