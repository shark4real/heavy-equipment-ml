# Heavy Equipment Resale Price Prediction

This project predicts the resale price of used heavy equipment from auction-style transaction records. The workflow is implemented in `Heavy_equip_notebook.ipynb` and uses exploratory analysis, leakage-safe feature engineering, model comparison, LightGBM tuning, and cross-validated final predictions.

The target column is `TargetValue`, and the main evaluation metric is RMSLE. The notebook models `log1p(TargetValue)`, so RMSE on the transformed target corresponds directly to RMSLE on the original price scale.

## Project Structure

```text
.
├── Heavy_equip_notebook.ipynb   # End-to-end EDA, feature engineering, modeling, tuning, and submission
├── train.csv                    # Training data: 138,701 rows, 50 columns
├── test.csv                     # Local placeholder/empty file in this checkout
└── README.md
```

## Dataset

The training data contains 138,701 equipment sale records with:

- `TransactionID`: row identifier
- `TargetValue`: resale price to predict
- Equipment identifiers such as `AssetID` and `ProductConfigID`
- Product descriptors such as `Spec_FullDescriptor`, `Spec_BaseClass`, and `FunctionalClassification`
- Usage and age-related fields such as `ManufactureYear` and `OperationalHoursMeter`
- Sale metadata such as `TransactionDate`, `RegionCode`, and `VendorPartnerID`
- Several sparse categorical columns named `col*`

Training sale dates range from `1990-01-31` to `2013-04-28`. `TargetValue` ranges from `7,500` to `142,000`, with a median of `35,000`.

Important local data note: the notebook was written for the Kaggle competition input path, where `test.csv` has 15,000 rows and 49 columns. In this repository checkout, `test.csv` is effectively empty, so final submission generation requires replacing it with the real competition test file or running the notebook in the Kaggle environment.

## Approach

### 1. Exploratory Data Analysis

The notebook checks target distribution, missingness, date and age behavior, operating hours, and categorical cardinality. Key findings:

- Raw prices are right-skewed, while `log1p(TargetValue)` is much closer to symmetric.
- `col18` and `col19` are about 99.95% missing and are dropped.
- `ManufactureYear` contains sentinel-like values such as `1000` and `1001`.
- `OperationalHoursMeter` has many zero or missing values, so missingness indicators are useful.
- High-cardinality fields such as `AssetID`, `Spec_BaseClass`, and `Spec_FullDescriptor` need careful encoding.

### 2. Feature Engineering

Feature engineering is row-wise and shared between train and test to avoid leakage. The notebook creates:

- Date features: `SaleYear`, `SaleMonth`, `SaleQuarter`
- Age features: `AssetAge`, `AgeSquared`
- Usage features: `HoursPerYear`, `LogHours`, `UsagePerAge`
- Text descriptor signals: length, word count, digit count, uppercase count, slash count
- Missingness flags for important sparse fields

The original `TransactionDate` is dropped after date features are extracted.

### 3. Preprocessing

The notebook uses two preprocessing tracks:

- Ridge and Random Forest use a scikit-learn `ColumnTransformer` with median imputation and scaling for numeric features, plus one-hot encoding for low-cardinality categoricals.
- LightGBM uses native categorical handling, out-of-fold target encoding for selected high-cardinality columns, and frequency encoding.

Target encoding is built with K-fold out-of-fold logic so validation rows are not encoded using their own target values.

### 4. Modeling

Three model families are compared:

- Ridge Regression
- Random Forest Regressor
- LightGBM Regressor

LightGBM performs best, so it is tuned with `RandomizedSearchCV` and then used for the final 5-fold cross-validation workflow.

## Results

Validation results saved in the notebook:

| Model | Val RMSLE | Val MAE (log) | Val R2 |
| --- | ---: | ---: | ---: |
| LightGBM tuned | 0.1978 | 0.1408 | 0.9127 |
| LightGBM | 0.1984 | 0.1418 | 0.9121 |
| Random Forest | 0.3262 | 0.2456 | 0.7625 |
| Ridge Regression | 0.3680 | 0.2847 | 0.6977 |

Final 5-fold LightGBM out-of-fold RMSLE:

```text
0.1993
```

The tuned LightGBM parameters found in the notebook are:

```python
{
    "subsample": 1.0,
    "reg_lambda": 1.0,
    "num_leaves": 768,
    "min_child_samples": 15,
    "colsample_bytree": 0.9,
}
```

## Setup

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the main dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm jupyter
```

## How to Run

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
Heavy_equip_notebook.ipynb
```

If running locally, update the data-loading cell from the Kaggle paths to local files:

```python
train = pd.read_csv("train.csv", low_memory=False)
test = pd.read_csv("test.csv")
```

To generate `submission.csv`, make sure `test.csv` contains the real competition test data with the same feature columns as training data except `TargetValue`.

## Output

The final notebook cell writes:

```text
submission.csv
```

with two columns:

- `TransactionID`
- `TargetValue`

## Notes

- The notebook is designed to avoid target leakage during validation and final cross-validation.
- The current local `test.csv` is not sufficient for final prediction generation.
- The model target is log-transformed, and predictions are converted back with `expm1` before writing the submission.
