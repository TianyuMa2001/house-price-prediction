"""Reusable training pipeline for Project Part II.

The model predicts ``log(Sale Price)`` as required by the assignment.  All
feature engineering is fitted inside the sklearn pipeline, so the exact same
transformations are applied to validation and contest data.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor


NUMERIC_FEATURES = [
    "Apartments",
    "Fireplaces",
    "Number of Commercial Units",
    "Longitude",
    "Latitude",
    "Age",
    "Age Decade",
    "Garage Indicator",
    "Bedrooms",
    "Rooms",
    "Bathrooms",
    "Log Building Square Feet",
    "Log Land Square Feet",
    "Log Lot Size",
    "Log Estimate (Land)",
    "Log Estimate (Building)",
    "Log Estimated Total Value",
    "Building to Land Ratio",
]

CATEGORICAL_FEATURES = [
    "Property Class",
    "Neighborhood Code",
    "Town Code",
    "Wall Material",
    "Roof Material",
    "Basement",
    "Basement Finish",
    "Central Heating",
    "Other Heating",
    "Central Air",
    "Attic Type",
    "Attic Finish",
    "Design Plan",
    "Cathedral Ceiling",
    "Construction Quality",
    "Site Desirability",
    "Garage 1 Size",
    "Garage 1 Material",
    "Garage 1 Attachment",
    "Garage 1 Area",
    "Garage 2 Size",
    "Garage 2 Material",
    "Garage 2 Attachment",
    "Garage 2 Area",
    "Porch",
    "Repair Condition",
    "Multi Property Indicator",
    "Modeling Group",
    "Use",
    "O'Hare Noise",
    "Floodplain",
    "Road Proximity",
    "Sale Year",
    "Sale Quarter of Year",
    "Sale Month of Year",
    "Most Recent Sale",
    "Pure Market Filter",
    "Town and Neighborhood",
]


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create stable numeric features and discard accidental CSV indexes."""

    def __init__(self, include_estimates=True):
        self.include_estimates = include_estimates

    def fit(self, X, y=None):
        validate_schema(X, include_estimates=self.include_estimates)
        return self

    def transform(self, X):
        validate_schema(X, include_estimates=getattr(self, 'include_estimates', True))
        data = pd.DataFrame(X).copy()
        data = data.loc[:, ~data.columns.astype(str).str.startswith("Unnamed:")]
        data = data.drop(columns=["Sale Price"], errors="ignore")

        description = data.get("Description", pd.Series("", index=data.index)).fillna("")
        patterns = {
            "Bedrooms": r"([0-9]+(?:\.[0-9]+)?) of which are bedrooms",
            "Rooms": r"([0-9]+(?:\.[0-9]+)?) rooms",
            "Bathrooms": r"([0-9]+(?:\.[0-9]+)?) of which are bathrooms",
        }
        for name, pattern in patterns.items():
            data[name] = pd.to_numeric(description.str.extract(pattern, expand=False), errors="coerce")

        for column in ['Estimate (Land)', 'Estimate (Building)']:
            if not getattr(self, 'include_estimates', True):
                data[column] = np.nan
        land_estimate = pd.to_numeric(data.get("Estimate (Land)"), errors="coerce")
        building_estimate = pd.to_numeric(data.get("Estimate (Building)"), errors="coerce")
        data["Estimated Total Value"] = land_estimate + building_estimate

        for source in [
            "Building Square Feet",
            "Land Square Feet",
            "Lot Size",
            "Estimate (Land)",
            "Estimate (Building)",
            "Estimated Total Value",
        ]:
            values = pd.to_numeric(data.get(source), errors="coerce").clip(lower=0)
            data[f"Log {source}"] = np.log1p(values)

        building_area = pd.to_numeric(data.get("Building Square Feet"), errors="coerce")
        land_area = pd.to_numeric(data.get("Land Square Feet"), errors="coerce")
        data["Building to Land Ratio"] = building_area / land_area.replace(0, np.nan)
        return data


RAW_NUMERIC = ['Apartments', 'Fireplaces', 'Number of Commercial Units', 'Longitude',
               'Latitude', 'Age', 'Age Decade', 'Garage Indicator',
               'Building Square Feet', 'Land Square Feet', 'Lot Size']


def validate_schema(data, include_estimates=True):
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError('Expected a nonempty pandas DataFrame.')
    if data.columns.duplicated().any():
        raise ValueError('Duplicate column names are not allowed.')
    numeric = RAW_NUMERIC + (['Estimate (Land)', 'Estimate (Building)'] if include_estimates else [])
    missing = sorted(set(numeric + CATEGORICAL_FEATURES + ['Description']) - set(data.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    for column in numeric + [c for c in CATEGORICAL_FEATURES if c != 'Modeling Group']:
        converted = pd.to_numeric(data[column], errors='coerce')
        if (data[column].notna() & converted.isna()).any() or np.isinf(converted).any():
            raise ValueError(f'Invalid numeric values in {column}')
    if not data['Description'].dropna().map(lambda value: isinstance(value, str)).all():
        raise ValueError('Description must contain strings or missing values.')


def create_pipeline(random_state: int = 42, model_kind='histgb', include_estimates=True) -> Pipeline:
    """Return a complete, unfitted model for log-scale sale prices."""

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    if model_kind == 'ridge':
        numeric_pipeline.steps.append(('scale', StandardScaler()))
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, [c for c in NUMERIC_FEATURES if include_estimates or 'Estimat' not in c]),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingRegressor(
        learning_rate=0.075,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=0.2,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=random_state,
    )
    if model_kind == 'ridge':
        categorical_pipeline.set_params(encoder=OneHotEncoder(handle_unknown='ignore', sparse_output=True))
        model = Ridge(alpha=10.0, solver='lsqr')
    elif model_kind == 'median':
        model = DummyRegressor(strategy='median')
    elif model_kind != 'histgb':
        raise ValueError(f'Unknown model: {model_kind}')
    return Pipeline(
        [
            ("features", HousingFeatureEngineer(include_estimates=include_estimates)),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def split_features_target(data: pd.DataFrame):
    """Remove invalid sales and return features plus the log target."""

    if "Sale Price" not in data:
        raise ValueError("Training data must contain a 'Sale Price' column.")
    prices = pd.to_numeric(data["Sale Price"], errors="coerce")
    valid = prices.ge(500) & np.isfinite(prices)
    X = data.loc[valid].drop(columns=["Sale Price"])
    y = np.log(prices.loc[valid].to_numpy())
    return X, y


def validate_pipeline(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Fit a holdout model and return log- and dollar-scale RMSE values."""

    X, y = split_features_target(data)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipeline = create_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_valid)
    return pipeline, {
        "log_rmse": float(mean_squared_error(y_valid, predicted) **0.5),
        "dollar_rmse": float(mean_squared_error(np.exp(y_valid), np.exp(predicted))**0.5),
        "validation_rows": int(len(y_valid)),
    }


def train_and_export(
    training_data: pd.DataFrame,
    contest_data: pd.DataFrame,
    model_path="pipeline.joblib.gz",
    predictions_path="predictions.csv",
    random_state: int = 42,
):
    """Train on all valid rows, save the model, and export contest predictions."""

    X, y = split_features_target(training_data)
    pipeline = create_pipeline(random_state=random_state)
    pipeline.fit(X, y)
    joblib.dump(pipeline, model_path, compress=("gzip", 3))

    log_predictions = pipeline.predict(contest_data)
    predictions = pd.DataFrame({"Sale Price": log_predictions})
    predictions.to_csv(predictions_path)
    return pipeline, predictions


def load_csv(path) -> pd.DataFrame:
    """Load a project CSV while tolerating one or more saved index columns."""

    data = pd.read_csv(Path(path))
    return data.loc[:, ~data.columns.astype(str).str.startswith("Unnamed:")]
