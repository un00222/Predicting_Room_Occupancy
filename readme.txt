# Room Occupancy Estimation — ML Models

A clean, modular implementation of multiple machine‑learning approaches for predicting room occupancy from sensor data. The repository includes **KNN**, **Closed‑Form Linear Regression**, **Linear Discriminant Analysis (LDA)**, **Proportional Odds (ordinal logistic)**, and a **Bagging ensemble**, with shared utilities for normalization and evaluation.

> This README formats the original project notes into a developer‑friendly guide you can drop into your repo.

---

##  Requirements

- **Python**: 3.2+ *(uses `concurrent.futures` in the bagging ensemble; tested on Python 3.11)*
- **Pip packages** (install if missing):
  ```bash
  pip install numpy pandas imbalanced-learn
  ```

> If you use a virtual environment:
> ```bash
> python -m venv .venv
> # Windows
> .\.venv\Scripts\activate
> # macOS/Linux
> source .venv/bin/activate
> pip install -r requirements.txt  # optional
> ```

Minimal `requirements.txt`:
```
numpy
pandas
imbalanced-learn
```

---

##  Data & Files

- **Dataset**: `occupancy_estimation.csv`
- Keep the following files **in the same directory** when running scripts:
  - `KNNClassification.py`, `closed_form_linear_regression.py`, `LDA.py`, `proportional_odds.py`, `bagging.py`
  - `utils.py`
  - `metrics.py` *(required by several scripts)*

Recommended layout:
```
project/
├─ occupancy_estimation.csv
├─ utils.py
├─ metrics.py
├─ KNNClassification.py
├─ LDA.py
├─ proportional_odds.py
├─ closed_form_linear_regression.py
└─ bagging.py
```

---

## Shared Utilities (`utils.py`)

> **Imports:** `numpy`, `pandas`, `imbalanced-learn`

Common preprocessing & evaluation helpers (not intended to run standalone).

### `normalize_data(...)` → normalized arrays
- Converts string‑parsable features to floats and **normalizes** them.
- Prints a **sample value** for features that **cannot** be converted to float.
- Optional `try_extra=True`:
  - Converts **dates** to one‑hot **day‑of‑week**
  - Converts **times** to integer **hour**

> **SMOTE option**: lines **75–81** can be uncommented to enable SMOTE; left off by default (ineffective for this dataset).

### `splitTrgValidation(...)` → `X_train, y_train, X_valid, y_valid`
- Splits data **2/3 training** and **1/3 validation**.

### `calcMetrics(y_true, y_pred)` → `precision, recall, f_measure, accuracy`
- Computes standard metrics.
- **Console output**: per‑class accuracy lines like  
  `Class 1: 295/350 = 84.29%`

### `calcDisplayMetrics(...)` → `None`
- Pretty table comparing **Training vs Validation**:
```
Metric     | Training Set | Validation Set
------------------------------------------
Precision  | 0.1234       | 0.1234
Recall     | 0.5678       | 0.5678
F_Measure  | 0.9876       | 0.9876
Accuracy   | 54.32%       | 54.32%
```

### `getNormalizedDataSets(...)` → `X_trg, Y_trg, X_valid, Y_valid`
- Convenience wrapper to generate splits.

### `print_confusion(y_true, y_pred)` → `None`
- Prints a labeled **confusion matrix** with totals:
```
        0    1    2    3  |    T
--------------------------------
 0 |    4    2    0    0  |    6
 1 |    1    3    2    0  |    6
 2 |    0    0    5    1  |    6
 3 |    0    0    3    3  |    6
--------------------------------
 T |    5    5   10    4  |   24
```

---

## K‑Nearest Neighbors (`KNNClassification.py`)

> **Imports:** `numpy`, `utils`  
> **Run location:** same directory as `occupancy_estimation.csv`, `utils.py`, `metrics.py`  
> **CLI parameters:** *none required*

### Usage (as a class)
```python
from KNNClassification import KNNClassifier
from utils import getNormalizedDataSets

X_trg, y_trg, X_val, y_val = getNormalizedDataSets(...)
model = KNNClassifier(k=5, weighting="uniform", eps=1e-12)  # weighting: "uniform" or "distance"
model.fit(X_trg, y_trg, X_val, y_val)
pred = model.predict(X_val)
```

### `main` behavior
- Tunes **k** for both `uniform` and `distance` weighting.
- Picks the **best** combo of `k` & weighting and trains final model.
- **Console outputs**:
  - `normalize_data` messages
  - Class distributions (`"Class 1: 295 samples"`)
  - Validation accuracy per k; best k summary
  - `calcMetrics` & `calcDisplayMetrics` tables

---

## Closed‑Form Linear Regression (`closed_form_linear_regression.py`)

> **Imports:** `numpy`, `utils`  
> **Run location:** same directory as data & utilities  
> **CLI parameters:** *none required*

### Usage (as a class)
```python
from closed_form_linear_regression import ClosedFormRegression  # example name
from utils import getNormalizedDataSets
X_trg, y_trg, X_val, y_val = getNormalizedDataSets(...)
model = ClosedFormRegression()
model.fit(X_trg, y_trg, X_val, y_val)
pred = model.predict(X_val)
```

### `main` behavior
- Trains **Closed‑Form Linear Regression** on normalized features (from `utils`) for `room_occupancy.csv`.
- Predicts classifications and computes **linear‑regression‑specific** metrics.
- **Console outputs**: `normalize_data` logs, LR metrics, `calcMetrics` & `calcDisplayMetrics`.

---

## Linear Discriminant Analysis (`LDA.py`)

> **Imports:** `numpy`, `utils`  
> **Run location:** same directory as data & utilities  
> **CLI parameters:** *none required*

### Usage (as a class)
```python
from LDA import lda
from utils import getNormalizedDataSets
X_trg, y_trg, X_val, y_val = getNormalizedDataSets(...)
model = lda()
model.fit(X_trg, y_trg, X_val, y_val)
pred = model.predict(X_val)
```

### `main` behavior
- Trains **LDA** on normalized data (`room_occupancy.csv`).
- Prints metrics via `utils` for predictions.

---

## Proportional Odds (Ordinal Logistic) — `proportional_odds.py`

> **Imports:** `numpy`, `utils`  
> **Run location:** same directory as data & utilities  
> **CLI parameters:** *none required*

### Usage (as a class)
```python
from proportional_odds import ProportionalOdds
from utils import getNormalizedDataSets
X_trg, y_trg, X_val, y_val = getNormalizedDataSets(...)

model = ProportionalOdds(learning_rate=1e-5, max_epochs=5000, tolerance=1e-10)
model.fit(X_trg, y_trg, X_val, y_val)  # reports val accuracy/loss every 1000 epochs
pred = model.predict(X_val)            # requires fit first (needs alpha/beta)
```

### `main` behavior
- Trains with `learning_rate=1e-5` and up to **20,000 epochs**.
- **Console outputs**:
  - `normalize_data` logs
  - **Every 1000 epochs**: `epoch: accuracy%, loss`
  - Final `calcMetrics` & `calcDisplayMetrics` table

---

## Bagging Ensemble (`bagging.py`)

> **Imports:** `numpy`, `utils`, and one of the base models (`KNNClassification`, `closed_form_linear_regression`, `LDA`, `proportional_odds`)  
> **Concurrency:** uses **`concurrent.futures`** to train subsets **in parallel** (all but 2 cores; uses 1 core on 1–2 core CPUs).  
> **CLI parameters:** *none required* — choose the base model by **uncommenting** the relevant lines in the script.

### Function 1: `bag(...)` → `pred_train, pred_val`
Trains an ensemble of base learners on multiple subsets and returns predictions.

**Parameters**
- `base_model_class`: one of `KNNClassifier`, `ClosedFormRegression`, `lda`, `ProportionalOdds`
- `num_subsets`: number of subset learners (voters)
- `X_train, Y_train, X_val`
- `max_workers`: parallelism (cores)
- `kwargs_con`: extra args for the base model **constructor**
- `kwargs_fit`: extra args for the base model **fit**

### Function 2: `determine_best_subset_count(...)` → `None`
Runs bagging with varying subset counts and prints the **best** voter count and metrics.

**Parameters**
- `base_model_class`, `X_train`, `Y_train`, `X_val`, `Y_val`
- `max_workers`, `min_subsets`, `max_subsets`
- `kwargs_con`, `kwargs_fit`

> **Note**: In `bagging.py` **main**, comment out **lines 135–136** when using `determine_best_subset_count` (it does not return predictions).

### Console output (bagging)
- `normalize_data` logs
- Core count used
- Base‑model training info
- `calcMetrics` & `calcDisplayMetrics`

**Unused (removed) code**
- `balance_training_data(...)` (over/under‑sampling) — removed; degraded results across all base models.

---

##  How to Run (Examples)

From the project directory:
```bash
# KNN
python KNNClassification.py

# Linear Discriminant Analysis
python LDA.py

# Proportional Odds (Ordinal Logistic)
python proportional_odds.py

# Closed‑Form Linear Regression
python closed_form_linear_regression.py

# Bagging (edit the script to choose base model first)
python bagging.py
```

---

## Tips & Notes

- Keep `occupancy_estimation.csv` small enough to share, or **exclude** it from Git with `.gitignore`.
- If enabling **SMOTE** in `utils.py`, document your rationale and sampling strategy.
- For reproducibility, fix random seeds and record library versions (`pip freeze > requirements.txt`).

---

#
