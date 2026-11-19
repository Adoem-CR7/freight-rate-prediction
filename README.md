# Freight Rate Prediction Challenge

Machine learning solution for Spotter's Freight Rate Prediction Challenge.

The objective is to predict freight transportation rates using shipment characteristics, including distance, equipment type, shipment weight, geographic information, and market indicators.

The solution uses LightGBM regression and includes separate models for the main validation dataset and the December prediction task.

## 1. Approach

### Data preprocessing

- Invalid shipment weights (≤ 0) are treated as missing values.
- Missing numerical values are handled natively by LightGBM.
- Categorical features are encoded consistently between training and inference.
- Date features are extracted from shipment dates.
- Extreme freight rates are retained in the training data.

### Model

Both models use LightGBM with an MAE regression objective.

Instead of predicting the total freight rate directly, the models predict the rate per mile:

`rate_per_mile = posted_rate / distance`

Training samples are weighted by shipment distance to align the optimization objective with absolute prediction errors in dollars.

Final predictions are calculated as:

`predicted_rate = predicted_rate_per_mile × distance`

### Feature sets

**Main model:** Distance, weight, market index, month, equipment, pickup/delivery locations, and geographic coordinates.

**December model:** Distance, weight, month, weekday, equipment, and pickup/delivery locations.

Separate models are used because the December dataset does not provide market indicators or geographic coordinates.

## 2. Validation

Model development uses three expanding-window chronological validation folds:

| Fold | Training period | Validation period |
|---|---|---|
| 1 | January–April 2025 | May–June 2025 |
| 2 | January–June 2025 | July–August 2025 |
| 3 | January–August 2025 | September–October 2025 |

This approach evaluates performance on future shipments while avoiding training on observations from later periods.

Models are compared using MAE, RMSE, and WAPE.

### Main model performance

| Metric | Average across folds |
|---|---:|
| MAE | $114.16 |
| RMSE | $631.15 |
| WAPE | 4.74% |

### December-compatible model performance

| Metric | Average across folds |
|---|---:|
| MAE | $135.73 |
| RMSE | $634.96 |
| WAPE | 5.63% |

The results are averages across the three chronological validation folds. Final hidden validation metrics are calculated by Spotter.

## 3. Installation

Python 3.12 is recommended.

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## 4. Generate predictions

From the project root directory, run:

```bash
python src/train_predict.py
```

This trains the final models using all available labeled development data and generates:

- `validation_predictions.csv` — 12,000 predictions for the main validation dataset.
- `data/december_predictions.csv` — 31 predictions for December 2025.

The original December input file remains unchanged.

## 5. Validate outputs and generate the December chart

Run the supplied scoring script:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_predictions.csv
```

The script validates both prediction files and generates:

`scorer_results/candidate_december.png`

The scorer checks submission formatting and generates the chart. It does not calculate prediction accuracy.

## 6. Exploratory analysis

The notebook `notebooks/01_exploration_and_modeling.ipynb` contains the exploratory data analysis, model comparisons, feature engineering experiments, hyperparameter tuning, and prediction-error analysis.

The standalone script `src/train_predict.py` is the final implementation used to generate the submission predictions.