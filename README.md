# Quadratic Rough Heston

We implement the hybrid simulation scheme described in the paper *The SSR under Quadratic Rough Heston*, available [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5239929).

This repository includes user-friendly Jupyter notebooks in both R and Python to demonstrate how to simulate and apply the Quadratic Rough Heston (QR Heston) model:

- **`forward_variance_curve_construction.ipynb`**: Demonstrates the construction of the forward variance curve using implied volatility data.
- **`qrheston_simulation.ipynb`**: Simulates the QR Heston model and visualizes SPX and VIX smiles.
- **`qrheston_ssr.ipynb`**: Computes the SSR in the QR Heston model.

## Python Code Installation

Follow these steps to set up a virtual environment and install the required dependencies:

1. Clone the repository:
   ```bash
   git clone https://github.com/jgatheral/QuadraticRoughHeston.git
   ```

2. Go to the project directory:
   ```bash
   cd QuadraticRoughHeston
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
<<<<<<< HEAD
=======

## Using Google Colab

You can also run the notebooks directly in Google Colab. To do this, follow these steps:
1. Open the notebook you want to run in Google Colab. For example, to run the `qrheston_simulation.ipynb`, go to
https://colab.research.google.com/github/jgatheral/QuadraticRoughHeston/blob/main/qrheston_simulation.ipynb
1. Create a cell at the top of the notebook and run the following:
```python
!git clone https://colab.research.google.com/github/jgatheral/QuadraticRoughHeston
%cd QuadraticRoughHeston/
!pip install numfracpy
```
>>>>>>> b888b6f (update README and move python files at root)
