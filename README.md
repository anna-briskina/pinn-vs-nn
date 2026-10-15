# Physics-Informed Neural Networks vs. Standard Neural Networks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anna-briskina/pinn-vs-nn/blob/main/PINN_vs_NN.ipynb)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)

A from-scratch PyTorch implementation of **Physics-Informed Neural Networks (PINNs)**, compared head-to-head with an otherwise identical standard neural network. The only difference between the two models is one term in the loss: the residual of the governing differential equation.

![Oscillator training](figures/04_oscillator_training.gif)

## Why PINNs?

A standard NN learns only from data, so it has no idea what to do where data is missing. A PINN is also penalised for violating the laws of physics at thousands of unlabelled "collocation" points, computed with automatic differentiation:

```
L_PINN = L_data  +  λ_ic/bc · L_ic/bc  +  λ_phys · (1/N_c) Σⱼ r_θ(xⱼ)²
```

- **L_data**: fit the measurements
- **L_ic/bc**: satisfy the initial and boundary conditions
- **physics term**: the equation's residual r must be zero at every collocation point

## Experiments

| # | Problem | Equation | Task |
|---|---------|----------|------|
| 1 | Damped harmonic oscillator | `m·u'' + c·u' + k·u = 0` | Extrapolate from 10 noisy points in the first 40% of the time window |
| 2 | Damped oscillator (inverse) | same, damping `c` unknown | Discover the damping coefficient from noisy data |
| 3 | 1-D heat equation | `∂u/∂t = α·∂²u/∂x²` | Reconstruct the full space-time field from 5 sensors |

Every comparison is fair: same architecture, initial weights, data, optimiser and number of steps.

## Results

<!-- Fill in from the summary cell after running the notebook -->

| Relative L2 error | Standard NN | PINN |
|-------------------|:-----------:|:----:|
| Oscillator, extrapolation (t > 0.4) | –% | –% |
| Heat equation, extrapolation (t > 0.5) | –% | –% |
| Inverse problem: learned c (true = 4.0) | n/a | – |

**Oscillator:** both models fit the data, but only the PINN follows the physics beyond it.
![Oscillator comparison](figures/02_oscillator_comparison.png)

**Heat equation:** field reconstruction and error maps.
![Heat comparison](figures/08_heat_comparison.png)

**Inverse problem:** the damping coefficient converges to its true value.
![Inverse problem](figures/06_inverse_problem.png)

## Key takeaways

- **Physics generalises where data can't.** The baseline matches the PINN inside the observed region and fails outside it.
- **Low training loss ≠ valid model.** The baseline fits its data while violating the governing equation by orders of magnitude.
- **Inverse problems for free.** Unknown physical parameters can be learned alongside the network weights.
- **Practical details matter.** Non-dimensionalising the residual, `tanh` activations (ReLU has zero second derivative) and Adam → L-BFGS training are what make it work.

## Run it

```bash
git clone https://github.com/anna-briskina/pinn-vs-nn.git
cd pinn-vs-nn
pip install -r requirements.txt
jupyter notebook PINN_vs_NN.ipynb
```

Runs in about 3–6 minutes on a laptop CPU (under 2 on a GPU). All figures are saved to `figures/`.

## Repository structure

```
├── PINN_vs_NN.ipynb   # full notebook: theory, code, results
├── figures/           # generated plots and training GIF
├── requirements.txt
└── README.md
```

## References

- Raissi, Perdikaris & Karniadakis (2019). *Physics-informed neural networks.* J. Comput. Phys. 378, 686–707.
- Karniadakis et al. (2021). *Physics-informed machine learning.* Nature Reviews Physics 3, 422–440.
- Wang, Teng & Perdikaris (2021). *Understanding and mitigating gradient flow pathologies in PINNs.* SIAM J. Sci. Comput.
