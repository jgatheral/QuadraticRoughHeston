# Quadratic Rough Heston

Python and R implementations accompanying *Quadratic Rough Heston: SPX, VIX, and the SSR*.

Model code lives in `src/` and `R/`; market data and supplied calibration parameters
live in `data/` for 20241219, 20250416, 20250620, and 20250702.

| Calculation | Python | R |
| --- | --- | --- |
| Forward variance curves | [forward_variance.ipynb](forward_variance.ipynb) | [forward_variance.ipynb](R/forward_variance.ipynb) |
| SPX/VIX smiles | [simulation_spx_vix.ipynb](simulation_spx_vix.ipynb) | [simulation_spx_vix.ipynb](R/simulation_spx_vix.ipynb) |
| Rømer comparison and step convergence | [scheme_comparison.ipynb](scheme_comparison.ipynb) | — |
| SSR term structures | [ssr.ipynb](ssr.ipynb) | [ssr.ipynb](R/ssr.ipynb) |

Use Python 3.11–3.13 and run from the repository root:

```bash
uv sync
uv run jupyter lab
```

For R notebooks, install the dependencies and register the kernel:

```r
install.packages(c("IRkernel", "gsl", "MittagLeffleR", "statmod"))
IRkernel::installspec()
```

Run notebooks from top to bottom. Simulations default to 300,000 paths, 300 steps.
Reduce paths and steps for quick checks.
