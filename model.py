from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score


def train_conversion_model(df: pd.DataFrame, schema: Dict) -> Tuple[Pipeline, List[str], Dict]:
    """
    Train a simple Logistic Regression model to predict conversion.
    Returns the fitted pipeline, feature names (conceptual), and metrics.
    """
    if 'converted' not in df.columns:
        raise ValueError('No target column "converted" available for modeling.')

    y = df['converted']
    # Select candidate features
    features = []
    numeric_features = []
    categorical_features = []

    if 'sales' in df.columns:
        features.append('sales')
        numeric_features.append('sales')
    if 'impressions' in df.columns:
        features.append('impressions')
        numeric_features.append('impressions')
    if 'clicks' in df.columns:
        features.append('clicks')
        numeric_features.append('clicks')
    if 'campaign' in df.columns:
        features.append('campaign')
        categorical_features.append('campaign')

    if not features:
        raise ValueError('No suitable features found for modeling.')

    X = df[features]

    preprocess = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    clf = LogisticRegression(max_iter=1000)
    pipe = Pipeline(steps=[('preprocess', preprocess), ('clf', clf)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        'roc_auc': float(roc_auc_score(y_test, y_prob)) if len(set(y_test)) > 1 else None,
        'accuracy': float(accuracy_score(y_test, y_pred))
    }

    return pipe, features, metrics


def predict_conversion_probabilities(model: Pipeline, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    X = df[features].copy()
    probs = model.predict_proba(X)[:, 1]
    out = df.copy()
    out['prob_conversion'] = probs
    return out
