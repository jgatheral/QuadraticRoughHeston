# Quadratic Rough Heston

We implement the hybrid simulation scheme described in the paper *Quadratic Rough Heston*, which is available [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5239929).

This repository includes user-friendly Jupyter notebooks in both R and Python to demonstrate how to simulate and apply the Quadratic Rough Heston (QR Heston) model:

- **`forward_variance_curve_construction.ipynb`**: Demonstrates the construction of the forward variance curve using implied volatility data.
- **`qrheston_simulation.ipynb`**: Simulates the QR Heston model and visualizes SPX and VIX smiles, replicating Figures 1 and 2 from the paper.
- **`qrheston_ssr.ipynb`**: Computes the SSR (Sum of Squared Residuals), reproducing Figure 3 from the paper.

## Python Code Installation

Follow these steps to set up the environment and install the required dependencies:

1. Clone the repository:
   ```bash
   git clone https://github.com/jgatheral/QuadraticRoughHeston.git
   ```

2. Go to the `Python` directory:
   ```bash
   cd QuadraticRoughHeston/Python
   ```

3. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```

4. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

5. Install the dependencies:
   ```bash
   pip install .
   ```

6. You can then launch Jupyter Lab by running:
   ```bash
   jupyter lab
   ```

---

* Alternatively, if you have [uv](https://docs.astral.sh/uv/) installed, you can simplify the process by running the following after step 2:
   ```bash
   uv sync
   ```

---