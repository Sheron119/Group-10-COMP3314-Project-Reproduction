# Group-10-COMP3314-Project-Reproduction

This repository contains a Python wrapper script (`repro.py`) designed to reproduce the empirical results of the NeurIPS 2024 paper **"Optimal Batched Best Arm Identification"**. 

The goal of this project was to run the authors' algorithms (Tri-BBAI and Opt-BBAI) on constrained hardware (a standard laptop) and compare the resulting Sample Complexity, Batch Complexity, and Recall against the paper's reported values.

## Paper Details
*   **Title:** Optimal Batched Best Arm Identification
*   **Authors:** Tianyuan Jin, Yu Yang, Jing Tang, Xiaokui Xiao, Pan Xu
*   **Conference:** NeurIPS 2024
*   **Original Codebase:** https://github.com/panxulab/Optimal-Batched-Best-Arm-Identification

---

## Setup and Run Instructions

### Prerequisites
Ensure you have Python 3.8+ installed. You will need the following libraries:
```bash
pip install numpy pandas scipy
```

### How to Run
1. Clone this repository to your local machine.
2. Open a terminal and navigate to the project directory.
3. Run the reproduction script:
```bash
python repro.py
```
**Note on Execution Time:** The script is configured to run **50 trials** per setting across **3 delta values** (`10^-1`, `10^-3`, `10^-5`) for both Uniform and Normal distributions. It takes approximately **6-10 minutes** to complete one distribution.

*(To run the Normal distribution instead of the default Uniform distribution, open `repro.py` and change `INSTANCE = "uniform"` to `INSTANCE = "normal"` at the bottom of the script).*

---

## Code Attribution
### 1. Paper's Author Contributions (Core Mathematics)
The mathematical foundation of this project was sourced directly from the authors' official repository. 
The following functions were copied directly (with minor safety patches as noted):
*   `bernouli_kl_divergence`, `chernoff`, `beta_func_TS`, `Ifonc`, `cost`, `xkofy`, `muddle`, `aux`, and `optimal_weights`.
*   The four-stage structural logic of `Tri_BBAI` and `Opt_BBAI`.

### 2. Our Modifications and Contributions
We built the wrapper script (`repro.py`) to make the authors' code runnable on constrained hardware. My original contributions include:
*   **Simulator:** `pull()` function simulate Bernoulli 0/1 rewards and track sample counts.
*   **Experiment Loop:** The nested loops to test multiple algorithms, multiple \(\delta\) values (0.1, 0.001, 0.00001), and generate the CSV results using Pandas.
*   **Bug Fixes for Overflow & Division by Zero Error:** 
    *   Added `1e-16` to denominators in `aux()` to prevent `NaN` division-by-zero errors when two arms tie.
    *   Replaced unbounded `while` loops in `xkofy()` with bounded `for` loops and loosened precision from `1e-11` to `1e-5` for speed.
    *   Added a hard cap (`L2 = 2000`) to prevent `float64` overflow in Stage 2.
*   **Safety Limits:** Added `if q > 10: break` and `if r > 10: break` to the `while` loops in the algorithms to prevent infinite execution.
*   **Data Generation:** `generate_means()` is used to create the Uniform and Normal bandit instances.

---

## Reproduced Results

**Data Format:** Each cell displays **Batch Complexity / Sample Complexity / Recall**.

| Algorithm | Distribution | \(\delta = 10^{-1}\) | \(\delta = 10^{-3}\) | \(\delta = 10^{-5}\) |
| :--- | :--- | :--- | :--- | :--- |
| **Tri-BBAI** | Uniform | 4.38 / 4,447 / 70% | 4.36 / 6,284 / 82% | 4.26 / 10,832 / 96% |
| **Tri-BBAI** | Normal | 4.16 / 3,968 / 94% | 4.14 / 5,306 / 96% | 3.98 / 8,833 / 100% |
| **Opt-BBAI** | Uniform | 3.92 / 2,194 / 60% | 4.28 / 2,900 / 74% | 4.16 / 3,923 / 76% |
| **Opt-BBAI** | Normal | 3.88 / 2,189 / 76% | 4.04 / 2,819 / 92% | 3.48 / 2,750 / 94% |
