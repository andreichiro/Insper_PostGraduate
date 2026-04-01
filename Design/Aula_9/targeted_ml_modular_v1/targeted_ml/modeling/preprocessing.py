from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class FeatureSchema:
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)


def build_column_transformer(schema: FeatureSchema) -> ColumnTransformer:
    transformers = []
    if schema.numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                schema.numeric_features,
            )
        )
    if schema.categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                schema.categorical_features,
            )
        )
    transformer = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    try:
        transformer.set_output(transform="pandas")
    except Exception:
        pass
    return transformer


def build_feature_schema(feature_registry: pd.DataFrame, feature_names: list[str]) -> FeatureSchema:
    registry = feature_registry.set_index("feature_name")
    numeric_features = [name for name in feature_names if name in registry.index and registry.loc[name, "feature_type"] == "numeric"]
    categorical_features = [name for name in feature_names if name in registry.index and registry.loc[name, "feature_type"] == "categorical"]
    return FeatureSchema(numeric_features=numeric_features, categorical_features=categorical_features)


def build_preprocessor_from_registry(feature_registry: pd.DataFrame, feature_names: list[str]) -> ColumnTransformer:
    return build_column_transformer(build_feature_schema(feature_registry, feature_names))
