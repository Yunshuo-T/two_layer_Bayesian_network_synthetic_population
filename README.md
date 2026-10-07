# Two-layer Bayesian model for synthetic populations
This package provides a two-layer Bayesian model for synthetic population generation. Moreover, it also provides synthetic reconstruction methods (e.g. IPU, HIPF, GR and CrossEntropy) to calibrate the synthesis results to marginal target marginals.

This package consists of model training, population synthesis and calibration three parts.
# Model input requirements
## Model training input
Model training requires attributes and arguments:
```
train:
  raw_h_df: "french_h.csv"
  raw_p_df: "french_p.csv"
  h_nodes: ["TYPL", "VOIT"]
  p_nodes: ["SEXE", "CS1", "DIPL_15", "EMPL", "ETUD", "STAT_CONJ"]
  age_col: "AGE"
  gender_col: "SEXE"
  h_id_col: "H_ID"
  score: "aic-d"
```
## Population synthesis input 
```
synthesis:
  num_households: 1500
  pool_multiplier: 50
  max_retries: 10
  random_seed: 42
  max_workers: 8
  p_data_include_kid: true
```

## Calibration input
```
calibration:
  max_iterations: 100
  tolerance: 1.0e-6
  p_marginals: {
    "SEXE": {"1": 2042494, "2": 2210748}, 
    "CS1": {"1": 1469, "2": 103500, "3": 611888, "4": 557518, "5": 595924, "6": 304869, "7": 628168, "8": 1449906}, 
    "DIPL_15": {"A": 971873, "B": 500578, "C": 620487, "D": 1350186, "Z": 810118}, 
    "EMPL": {"11": 38876, "12": 21212, "13": 8839, "14": 10907, "15": 153798, "16": 1506973, "21": 110790, "22": 68924, "23": 1488, "ZZ": 2331435}, 
    "ETUD": {"1": 1121545, "2": 3131697}, "STAT_CONJ": {"A": 1397063, "B": 2856179}, 
    "age_cate": {"0": 310218, "1": 654153, "2": 707364, "3": 653433, "4": 592718, "5": 517486, "6": 403933, "7": 232916, "8": 181021}
    }
  h_marginals: {
    "TYPL": {"1": 18799, "2": 437277, "3": 14317, "4": 5943, "5": 1545, "6": 7737}, 
    "VOIT": {"0": 296715, "1": 156547, "2": 31430, "3": 14159}
    }
  p_attributes: ["SEXE", "CS1", "DIPL_15", "EMPL", "ETUD", "STAT_CONJ"]
  h_attributes: ["TYPL", "VOIT"]
  kwargs:
    method: "logit" #'linear', 'raking', or 'logit'
    bounds: (0.001,50.0)  # The bounds for logit method. Defaults to (0.1,5.0).
    alpha: 1e-3 # For CrossEntropy 
```

# Expected output
The output will be two `.csv` files for household and household members.

# Usage example
Need to complete

## Testing

Install the package with the test dependencies in your Python environment:

```console
python -m pip install -e ".[test]"
```

Run the pytest suite from the project root:

```console
python -m pytest -q
```

Tests live in `test.py` and use small example datasets and temporary directories.
To run one group, use e.g. `python -m pytest test.py::TestEncoding -q`.
