
# Fourier Neural Operator for Heston Option Pricing

## Overview

This project investigates the use of a **Fourier Neural Operator (FNO)** as a surrogate model for numerical option pricing under the **Heston stochastic volatility model**.

The central objective is to learn the mapping

$$
(\text{Heston parameters}, S, v)
\;\longrightarrow\;
V(S,v)
$$

where the input consists of the model parameters together with the underlying-asset and variance coordinates, and the output is the corresponding **Heston option-pricing surface**.

Rather than solving the Heston pricing problem independently for every new parameter set using a conventional numerical PDE solver, the FNO approach learns an operator that can approximate the solution across a family of parameter configurations. This makes the problem an example of **operator learning for computational finance**.

The project is designed around three related questions:

1. Can an FNO learn accurate Heston pricing surfaces from numerically generated reference solutions?
2. How well does the learned model generalise to parameter configurations that were not used during training?
3. Can the resulting surrogate provide substantially faster inference once training has been completed?

---

## Research Context

The **Heston model** captures stochastic volatility by allowing the instantaneous variance to evolve as a mean-reverting stochastic process. Pricing under this model leads to a two-dimensional PDE in the underlying asset price \(S\) and variance \(v\).

A conventional numerical approach repeatedly solves this PDE for different parameter configurations. This can become computationally expensive when large numbers of pricing surfaces are required.

This project therefore explores a different formulation:

> **Learn the solution operator itself rather than repeatedly solving the PDE from scratch.**

The reference pricing surfaces used for supervised learning are generated numerically and supplied to the FNO as training targets.

The broader motivation is to investigate whether **operator-learning methods can act as reusable surrogate models for computationally intensive quantitative-finance calculations.**

---

## Methodology

### 1. Reference Data

The model is trained on numerically generated Heston pricing surfaces.

Each sample contains:

* a vector of **five Heston model parameters**;
* an underlying-price grid \(S\);
* a variance grid \(v\);
* the corresponding reference option-pricing surface.

The dataset is represented internally as:

```text
Parameters : [N, 5]
Price data : [N, H, W]
```

The FNO input is constructed by broadcasting the five Heston parameters over the spatial grid and appending the \(S\) and \(v\) coordinate channels:

```text
Input shape  : [batch, 7, H, W]
Output shape : [batch, 1, H, W]
```

Thus the seven input channels consist of:

```text
5 × Heston parameter channels
1 × underlying-price coordinate S
1 × variance coordinate v
```

This design allows the neural operator to condition the predicted pricing surface on both the model parameters and the spatial coordinates.

---

## 2. Fourier Neural Operator

The core model is implemented in `src/heston_fno/models/fno.py`.

The project uses the `neuraloperator` implementation of the Fourier Neural Operator through a lightweight `HestonFNO` wrapper.

The main architecture parameters are configurable through YAML:

```yaml
model:
  n_modes: [8, 8]
  hidden_channels: 32
  in_channels: 7
  out_channels: 1
```

The FNO operates in Fourier space, where learned spectral transformations are used to capture global relationships across the pricing surface.

Unlike a conventional pointwise neural network, the objective is therefore not simply to predict an individual option price, but to approximate an entire **function-valued solution**.

---

## 3. Data Preprocessing

The preprocessing pipeline performs:

1. deterministic train/validation/test splitting;
2. computation of channel-wise statistics from the training data;
3. standardisation of input and target tensors;
4. application of the same fitted transformations to validation and test samples.

The default dataset split is:

```text
80% training
10% validation
10% testing
```

The split is generated using a fixed random seed to improve reproducibility.

Training and evaluation configuration is centralised in:

```text
configs/fno.yaml
```

This makes experiments easier to reproduce without changing the Python implementation.

---

## 4. Training

Training is separated into reusable components under:

```text
src/heston_fno/training/
```

The training pipeline provides:

* mini-batch optimisation;
* validation after each epoch;
* training and validation loss tracking;
* epoch and wall-clock timing;
* best-validation-model checkpointing.

The current training script can be launched using:

```bash
python scripts/train_fno.py
```

A saved checkpoint contains the model state together with relevant experiment metadata, including the configuration and fitted normalisation statistics.

---

## 5. Pricing Evaluation

The project includes dedicated pricing-evaluation utilities under:

```text
src/heston_fno/evaluation/
```

Implemented error measures include:

* Mean Absolute Error (MAE)
* Mean Relative Error (MRE)
* Root Mean Squared Error (RMSE)
* Maximum Absolute Error
* Relative \(L_2\) error

For example:

```bash
python scripts/evaluate.py
```

The evaluation pipeline reconstructs the physical pricing scale before calculating pricing errors when target normalisation has been used.

This distinction is important because model optimisation takes place on normalised tensors, whereas the final assessment should be interpreted in the original pricing units.

---

## 6. Pricing-Surface Analysis

The model is intended to produce an entire pricing surface rather than an isolated scalar prediction.

For a given Heston parameter vector, the network predicts

$$
\hat V(S,v)
$$

over the supplied \(S\)- and \(v\)-grids.

The repository contains utilities for generating these predictions through:

```text
src/heston_fno/evaluation/pricing.py
```

The notebooks under `notebooks/` provide an interactive environment for inspecting the dataset, model outputs, and experimental results.

---

## 7. Greeks and Sensitivity Analysis

A further objective of the project is to assess whether sensitivities can be recovered from the learned pricing surface.

The repository includes automatic-differentiation-based utilities for:

### Delta

$$
\Delta = \frac{\partial V}{\partial S}
$$

### Vega-like sensitivity

$$
\text{Vega} \approx \frac{\partial V}{\partial v_0}
$$

where \(v_0\) denotes the initial variance parameter.

The implementation is contained in:

```text
src/heston_fno/evaluation/greeks.py
```

Reference sensitivities can also be estimated numerically from reference pricing surfaces using finite differences.

This allows the project to examine not only whether prices are reproduced accurately, but also whether the learned operator preserves economically relevant **first-order sensitivities**.

---

## 8. Benchmarking

The repository contains benchmarking infrastructure for measuring FNO inference performance.

The FNO benchmark records:

* average inference time;
* batch throughput.

For example:

```bash
python scripts/benchmark.py
```

The project also contains an ADI benchmarking module under:

```text
src/heston_fno/benchmarking/adi.py
```

This provides a framework for timing a conventional **Hundsdorfer–Verwer ADI** reference solver.

The intention is to make the computational comparison explicit:

```text
Numerical PDE solver
        ↓
Reference pricing surface

             versus

Fourier Neural Operator
        ↓
Learned pricing surface
```

Any reported speed-up should be interpreted only after matching the numerical accuracy, hardware, grid resolution, batch size, and timing methodology of the respective approaches.

---

## 9. Generalisation and Alternative References

The project also contains utilities for comparing FNO predictions with independently generated reference surfaces:

```text
src/heston_fno/evaluation/generalization.py
```

The current interface is designed to support comparison against alternative pricing references, including a Carr–Madan-style reference workflow.

A complete quantitative comparison should use an independently implemented and independently evaluated reference method rather than reusing the training target as the reference.

---

## Repository Structure

```text
FNO-for-Heston-Pricing-Engine-/
│
├── configs/
│   └── fno.yaml
│
├── notebooks/
│   ├── 01_dataset.ipynb
│   ├── 02_fno.ipynb
│   ├── colab_train.ipynb
│   └── results.ipynb
│
├── scripts/
│   ├── train_fno.py
│   ├── evaluate.py
│   ├── benchmark.py
│   ├── prepare_dataset.py
│   ├── inspect_dataset.py
│   ├── test_generalization.py
│   ├── test_evaluation_helpers.py
│   └── colab_run.sh
│
├── src/
│   └── heston_fno/
│       ├── data/
│       │   ├── dataset.py
│       │   ├── normalization.py
│       │   └── preprocessing.py
│       │
│       ├── models/
│       │   └── fno.py
│       │
│       ├── training/
│       │   ├── losses.py
│       │   └── train.py
│       │
│       ├── evaluation/
│       │   ├── errors.py
│       │   ├── generalization.py
│       │   ├── greeks.py
│       │   └── pricing.py
│       │
│       └── benchmarking/
│           ├── fno.py
│           ├── adi.py
│           └── speedup.py
│
├── tests/
│   ├── test_benchmarking.py
│   ├── test_dataset.py
│   ├── test_errors.py
│   └── test_model.py
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Installation

The project requires Python 3.10 or later.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

For editable installation of the package:

```bash
pip install -e .
```

The principal dependencies are:

* PyTorch
* `neuraloperator`
* NumPy
* PyYAML
* PyTest

---

## Data

The pricing dataset is intentionally excluded from version control because of its size.

The expected directory is:

```text
data/final/
├── inputs.pt
├── targets.pt
├── s_grid.pt
└── v_grid.pt
```

The repository therefore contains the **data-loading and preprocessing pipeline**, while the generated tensor dataset is maintained separately.

---

## Reproducibility

Experiment configuration is stored in:

```text
configs/fno.yaml
```

Key parameters include:

```yaml
training:
  batch_size: 16
  lr: 0.001
  epochs: 10
  seed: 42
  split: [0.8, 0.1, 0.1]
```

Using a fixed seed, recorded configuration, saved checkpoint, and stored normalisation statistics provides a reproducible basis for subsequent experiments.

For GPU experiments, the repository also includes:

```bash
scripts/colab_run.sh
```

which performs environment checks, installs the package, verifies the dataset, runs a GPU smoke test, and launches training.

---

## Validation and Testing

Unit tests are organised under:

```text
tests/
```

and can be executed with:

```bash
pytest
```

Additional diagnostic scripts are provided for dataset inspection, evaluation helpers, generalisation experiments, and inference benchmarking.

---

## Current Experimental Status

The repository should be viewed as an **ongoing research implementation**, rather than as a claim that FNO has already outperformed traditional Heston pricing methods.

The experimental workflow is structured around the following progression:

```text
Reference numerical solutions
            ↓
       Data preprocessing
            ↓
      FNO model training
            ↓
     Held-out test evaluation
            ↓
   Pricing-surface comparison
            ↓
       Greek comparison
            ↓
 Computational benchmarking
```

Final conclusions should be drawn from independently generated test data and controlled comparisons of accuracy and computational cost.

---

## Research Directions

The implementation provides a foundation for several extensions:

* increasing the size and diversity of the training parameter space;
* hyperparameter and spectral-mode studies;
* systematic out-of-distribution generalisation tests;
* comparison against independent Carr–Madan pricing references;
* comparison against ADI/PDE runtimes under matched numerical settings;
* quantitative comparison of Delta and Vega sensitivities;
* investigation of accuracy–latency trade-offs;
* larger-resolution pricing surfaces and more demanding parameter regimes.

---

## References

Heston, S. L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options*. The Review of Financial Studies, 6(2), 327–343.

Li, Z. et al. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations*. International Conference on Learning Representations (ICLR).

---

## Project Objective

The long-term objective is to assess whether **Fourier Neural Operators can provide accurate and computationally efficient surrogate models for stochastic-volatility option pricing**, while maintaining useful information about the resulting pricing surfaces and their sensitivities.

The project therefore sits at the intersection of:

**Quantitative Finance × Numerical PDEs × Scientific Machine Learning × Operator Learning**

> **Key idea:** instead of solving the Heston pricing PDE from scratch for every parameter configuration, learn an approximation to the solution operator that maps model parameters and spatial coordinates to the corresponding pricing surface.
