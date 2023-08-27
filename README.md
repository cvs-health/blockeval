# Randomized Block Design Analysis

Module to analyze randomized block design in python. Blocks can be population segments derived from a predictive model, an uplift score, or any business rule. It assumes that individuals within each block are randomized in a treated and control groups. The sample size ratio between the treated and control groups can differ between blocks. When combining blocks, the Weighted Average Treatment Effect is calculated to avoid the counfounding effect of blocks and Simpson's paradox.

Users can:
- estimate the treatment effect for each block or group of blocks
- compare the treatment effect between blocks or group of blocks
- estimate the overall treatment effect of the campaign


## Contributors

- Mathieu d'Acremont
- Audrey Lee

## Installation

The latest version can be installed from PyPI:
```
pip install blockeval
```

## Test

In a jupyter notebook, check if you can import the package functions:
```
from blockeval.analysis import *
from blockeval.utils import campaign_simulation
```

## Example

This example [notebook](https://github.aetna.com/1636171/blockeval/blob/main/examples/tech_blog.ipynb) shows how to run a segment analysis on an uplift campaign and how to solve the Simpson's paradox.
