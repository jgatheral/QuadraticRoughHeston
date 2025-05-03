# Quadratic Rough Heston

We implement the hybrid simulation scheme from the paper Quadratic Rough Heston 
available [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5239929)

We provide easy-to-follow Jupyter notebooks in R and Python that show how 
to simulate and use the quadratic rough Heston (QR Heston) model.

- `forward_variance_curve_construction.ipynb` shows how to construct the forward variance curve from implied volatility data.

- `qrheston_simulation.ipynb` simulates the QR Heston model and plots SPX and VIX smiles, reproducing Figures 1 and 2 of the paper.

- `qrheston_ssr.ipynb` shows how to compute the SSR, reproducing Figure 3 of the paper.

## Installation for Python code

To set up the environment and install dependencies, follow these steps:

0. Clone the repository and cd into `/Python`

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:

    ```bash
    source .venv/bin/activate
    ```

3. Install the required dependencies:

   ```bash
   pip install .
   ```
