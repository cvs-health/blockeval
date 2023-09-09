# Randomized Block Design Analysis

Python module designed to analyze randomized block design. Blocks can be population segments derived from a predictive model, an uplift score, or any business rule. Individuals within each block have to be randomized in a treated and control group. The sample size ratio between treated and control can differ between blocks. When combining blocks, the Weighted Average Treatment Effect is calculated to mitigate the confounding effect of blocks and avoid Simpson's paradox.

With *blockeval*, users can:
- Estimate the treatment effect for each block
- Roll-up results to get effects for groups of blocks or the entire campaign
- Compare the treatment effects between blocks or groups of blocks (uplift)
- Resolve Simpson’s paradox
- Provide p-values and confidence intervals
- Handle non-normal distributions with bootstrapping

Results are also valid if the blocks are defined after the intervention in a completely randomized design (as opposed to a randomized block design).

## Contributors

- Mathieu d'Acremont
- Audrey Lee

## Installation

The latest version from PyPI can be installed using the following command:
```
pip install blockeval
```

## Test

To check the installation, open a Jupyter Notebook and try importing the package functions:
```
from blockeval.analysis import *
from blockeval.utils import campaign_simulation
```

## Quick Start

Please follow this [notebook](https://github.com/cvs-health/blockeval/blob/main/examples/quickstart.ipynb) for a quick introduction to the package.

## Medium Blog Post

We wrote a comprehensive review of *blockeval* in a Medium blog [post](https://medium.com/cvs-health-tech-blog/analyzing-randomized-block-design-and-uplift-campaigns-with-python-9a9dc5c8b064) with a companion [notebook](https://github.com/cvs-health/blockeval/blob/main/examples/medium_post.ipynb).
