# Credit Card Fraud Prediction

**Detecting rare fraudulent transactions while minimising false fraud alerts**

This project investigates how effectively supervised machine learning can identify fraudulent credit card transactions in a highly imbalanced dataset. It compares **Logistic Regression**, **Random Forest**, and **XGBoost**, with particular attention to the trade-off between detecting fraud (**recall**) and avoiding unnecessary alerts (**precision**).

The final selected approach, **XGBoost with class weighting**, achieved **94.81% precision**, **76.84% recall**, **84.88% F1-score**, and **0.9722 ROC-AUC** on a held-out test set. At a fixed decision threshold of 0.50, it correctly detected **73 fraudulent transactions** while incorrectly flagging only **4 legitimate transactions**. It also missed **22 fraud cases**, an important limitation.

> **Research question:** How effectively can supervised machine learning models detect rare fraudulent credit card transactions while limiting incorrect fraud alerts?

## Project overview

| Item | Description |
|---|---|
| Task | Binary classification: fraudulent (`1`) vs legitimate (`0`) transactions |
| Dataset | Credit Card Fraud Detection — European cardholder transactions, September 2013 |
| Dataset size | 284,807 original transactions; 283,726 after removing exact duplicates |
| Fraud prevalence after cleaning | 473 fraud transactions, approximately **0.167%** |
| Features | 30 predictors: `Time`, `Amount`, and anonymised `V1`–`V28` |
| Models | Logistic Regression, Random Forest, XGBoost |
| Evaluation | Stratified train/validation/test split (60% / 20% / 20%) |
| Primary focus | Precision, recall, F1-score, ROC-AUC, average precision, and false-positive/false-negative counts |
| Selected model | XGBoost, evaluated on the held-out test set at threshold **0.50** |

## Final results — held-out test set

**XGBoost · 56,746 test transactions · 95 fraudulent transactions · threshold = 0.50**

| Metric | Result |
|---|---:|
| **Precision (fraud)** | **94.81%** |
| **Recall (fraud)** | **76.84%** |
| **F1-score (fraud)** | **84.88%** |
| **ROC-AUC** | **0.9722** |
| **Average precision (AP)** | **0.8345** |
| Accuracy | 99.95% |

**Test-set confusion matrix** (rows = actual class, columns = predicted class):

| | Predicted legitimate | Predicted fraud |
|---|---:|---:|
| **Actually legitimate** | 56,647 | **4** false positives |
| **Actually fraud** | **22** false negatives | **73** true positives |

The model correctly identified **73 of 95 fraud cases** and generated only **4 false fraud alerts** among **56,651 legitimate transactions**. However, **22 fraud cases went undetected**. Accuracy alone is not an appropriate success measure because nearly all transactions are legitimate.

## Dataset

**Original source:** [Credit Card Fraud Detection — Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains European cardholder transactions recorded over approximately two days in September 2013. The binary label `Class` indicates whether a transaction is fraudulent. `Time` denotes seconds elapsed since the first recorded transaction, while `Amount` represents transaction value. `V1`–`V28` are anonymised PCA-transformed features, so their underlying meanings are not available.

### Why `creditcard.csv` is not in this repository

The dataset is large (approximately **144 MB**) and exceeds GitHub's standard **100 MB per-file upload limit**. To keep the repository lightweight, **the CSV is not committed**.

**You do not need to download the CSV to run the notebook as written.** The notebook loads it directly from the public TensorFlow-hosted URL:

```python
import pandas as pd

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)
```

An internet connection is required for this step.

**Alternative — manual download from Kaggle:**

1. Open the [dataset page on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Select **Download** (sign in if required), then extract the ZIP archive.
3. Locate `creditcard.csv` and place it in the same directory as the notebook.
4. **Only if using the local file instead of the existing URL**, replace the loading cell with:

   ```python
   import pandas as pd
   df = pd.read_csv("creditcard.csv")
   ```

You can also download the dataset with the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) after configuring Kaggle authentication:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip
```

For a Git repository, add `creditcard.csv` to `.gitignore` to prevent accidental commits.

## Methodology

### 1. Data cleaning and exploratory data analysis

- Checked dataset dimensions, data types, missing values, duplicates, and basic validity of `Time` and `Amount`.
- Found **no missing values**; removed **1,081 exact duplicate rows**, leaving **283,726 transactions** and **473 fraud cases**. This cleaning choice also removes 19 fraud-labelled rows and is discussed as a limitation.
- Explored class imbalance; transaction amount distributions and log-transformed distributions; fraud vs legitimate amounts; temporal patterns; transaction-amount outliers; and correlations with the target.
- Kept high-value outliers in the data, rather than automatically deleting potentially informative fraud transactions.
- Used EDA-only helper variables for analysis but **excluded them from model training**.

### 2. Data splitting and preprocessing

- Used **30 original predictors**: `Time`, `Amount`, and `V1`–`V28`.
- Applied two stratified splits with `random_state=42`: **60% training**, **20% validation**, and **20% testing**.
- Training: **170,235** transactions; validation: **56,745**; test: **56,746**.
- Fitted `StandardScaler` on training data only, then transformed validation and test features for models requiring scaled inputs.
- Used `RandomUnderSampler(sampling_strategy=0.05)` **on training data only** for Logistic Regression and Random Forest. This produced **5,680 legitimate** and **284 fraudulent** training transactions.
- Kept the validation and test sets at their original class distributions.

### 3. Model training

**Logistic Regression — baseline.** Trained on the scaled, undersampled data with `max_iter=1000` to establish a simple comparison model.

**Random Forest.** Trained on the same scaled, undersampled data with `random_state=42` to capture potentially nonlinear fraud patterns.

**XGBoost.** Trained on the **full, unscaled training data**, retaining the legitimate transactions removed by undersampling in the other experiments. The manually specified settings are:

```python
XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=5,
    scale_pos_weight=10,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)
```

These hyperparameters were **manually selected**; no grid search, random search, or systematic threshold optimisation is documented in the notebook.

### 4. Validation comparison

All validation classification results below use a **fixed 0.50 threshold**.

| Model / imbalance strategy | Precision | Recall | F1 | ROC-AUC | Average precision | False positives | Missed fraud |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression + undersampling | 47.73% | **89.36%** | 62.22% | 0.9796 | 0.7917 | 92 | **10** |
| Random Forest + undersampling | 65.35% | 88.30% | 75.11% | 0.9591 | 0.8363 | 44 | 11 |
| **XGBoost + class weighting** | **90.91%** | 85.11% | **87.91%** | **0.9805** | **0.8881** | **8** | 14 |

XGBoost achieved the strongest **validation precision, F1-score, ROC-AUC, and average precision** among these three configurations, and reduced false alerts to **8** on the validation set. Logistic Regression had the highest **recall**, detecting 84 of 94 fraud cases but generating 92 false alerts.

**Important comparison caveat:** Logistic Regression and Random Forest use undersampled training data, whereas XGBoost uses the complete training set with class weighting. These results compare **modelling approaches**, not isolated algorithms under identical training conditions.

### 5. Final held-out test

The selected XGBoost model was evaluated **once on the separate test set** at the unchanged **0.50 threshold**. Validation precision was **90.91%** versus **94.81%** on test; recall decreased from **85.11%** to **76.84%**, and average precision decreased from **0.8881** to **0.8345**. With only 94–95 fraud examples in each evaluation split, a few predictions can materially change these values.

## How to run the project

### Option A — Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the `.ipynb` notebook from this repository (**File → Upload notebook**).
3. Run the cells in order (**Runtime → Run all**).
4. Ensure internet access is available for the public dataset URL. There is **no need to upload `creditcard.csv`**.

The notebook installs `xgboost` in a cell. If your environment does not already have the other packages, install them as shown below.

### Option B — local Jupyter Notebook

**Requirements:** Python 3 and Jupyter, with internet access for dataset loading.

Clone or download the repository, open a terminal in its folder, and install the dependencies:

```bash
python -m pip install notebook pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

Start Jupyter:

```bash
jupyter notebook
```

Open the project `.ipynb` file and select **Run All**. On Windows, if `python` is not recognised, try `py -m pip install ...` instead. The notebook's `%pip install -q xgboost` cell is Jupyter-specific.

### Dependencies

| Package | Use |
|---|---|
| `pandas`, `numpy` | Data loading and analysis |
| `matplotlib`, `seaborn` | EDA and evaluation visualisations |
| `scikit-learn` | Splitting, scaling, baseline models, and evaluation metrics |
| `imbalanced-learn` | Random undersampling |
| `xgboost` | Gradient-boosted tree classifier |
| `notebook` / Google Colab | Notebook execution |

*Exact dependency versions were not pinned in the uploaded notebook, so small numerical differences may occur across environments.*

## Key findings

1. **Fraud is extremely rare:** after deduplication, only about **0.167%** of transactions are fraudulent. A nearly perfect accuracy score can therefore hide poor fraud detection.
2. **Precision and recall tell different stories:** Logistic Regression had the highest validation recall, but at the cost of many more false alerts.
3. **XGBoost delivered the best overall validation precision–recall balance among the tested configurations**, with **90.91% precision** and **87.91% F1** at threshold 0.50.
4. **The final test result is promising but imperfect:** XGBoost produced only **4 false positives**, while failing to detect **22 of 95 fraud cases**.
5. **Operating thresholds matter:** the dataset lacks real investigation costs and fraud-loss amounts, so the fixed **0.50** threshold cannot be claimed as financially optimal.

## Limitations and future work

- **Historical dataset:** Transactions date to September 2013; results do not demonstrate performance on present-day payments.
- **Anonymised features:** `V1`–`V28` cannot be directly explained in business terms, and demographic fairness cannot be assessed from the available attributes.
- **Duplicate removal:** Exact duplicates might be legitimate repeated observations; results could change if they were retained.
- **Small fraud sample:** Only **473** fraud-labelled observations remained after cleaning, with **95** in the test split. Confidence intervals and repeated evaluations would help quantify uncertainty.
- **Random rather than chronological splits:** A future-time test would better assess resilience to changing fraud patterns.
- **Different training strategies:** Compare class weighting, undersampling, and other imbalance approaches under controlled conditions.
- **No systematic tuning or cost-based threshold selection:** Explore hyperparameter searches, probability calibration, threshold curves, and decision rules that incorporate investigation capacity and false-negative costs.
- **Not deployment validated:** The work is an offline experiment; no claim is made about production readiness or actual financial savings.

## Dataset attribution

Data: [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), originally made available by the ULB Machine Learning Group and collaborators. Please refer to the dataset page for its original description, terms, and citation information.

---

**Project scope:** An educational machine learning study of rare-event classification and the fraud-detection/false-alert trade-off, not a production fraud prevention system.
