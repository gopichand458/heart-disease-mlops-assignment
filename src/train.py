"""Train Logistic Regression and Random Forest with CV, tune, log to MLflow."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor, load_xy

SEED = 42
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = Path("models/model.joblib")

CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, random_state=SEED),
        {"clf__C": [0.01, 0.1, 1, 10]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED),
        {"clf__n_estimators": [100, 300], "clf__max_depth": [3, 5, None]},
    ),
}


def main() -> None:
    X, y = load_xy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    mlflow.set_experiment("heart-disease")
    best_name, best_auc, best_model = None, -1.0, None

    for name, (clf, grid) in CANDIDATES.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        with mlflow.start_run(run_name=name):
            search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc")
            search.fit(X_tr, y_tr)
            model = search.best_estimator_

            cv_scores = cross_validate(
                model, X_tr, y_tr, cv=cv,
                scoring=["accuracy", "precision", "recall", "roc_auc"])
            pred = model.predict(X_te)
            proba = model.predict_proba(X_te)[:, 1]
            metrics = {
                "cv_accuracy": cv_scores["test_accuracy"].mean(),
                "cv_precision": cv_scores["test_precision"].mean(),
                "cv_recall": cv_scores["test_recall"].mean(),
                "cv_roc_auc": cv_scores["test_roc_auc"].mean(),
                "test_accuracy": accuracy_score(y_te, pred),
                "test_precision": precision_score(y_te, pred),
                "test_recall": recall_score(y_te, pred),
                "test_roc_auc": roc_auc_score(y_te, proba),
            }
            mlflow.log_param("model", name)
            mlflow.log_params(search.best_params_)
            mlflow.log_param("seed", SEED)
            mlflow.log_metrics(metrics)

            for kind, display in (
                ("roc_curve", RocCurveDisplay.from_estimator),
                ("confusion_matrix", ConfusionMatrixDisplay.from_estimator),
            ):
                display(model, X_te, y_te)
                path = ARTIFACT_DIR / f"{name}_{kind}.png"
                plt.title(f"{name}: {kind}")
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close()
                mlflow.log_artifact(str(path))

            mlflow.sklearn.log_model(
                model, name="model",
                skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"])
            print(name, {k: round(v, 3) for k, v in metrics.items()})
            if metrics["cv_roc_auc"] > best_auc:
                best_name, best_auc, best_model = name, metrics["cv_roc_auc"], model

    joblib.dump(best_model, MODEL_PATH)
    Path("models/metadata.json").write_text(
        json.dumps({"best_model": best_name, "cv_roc_auc": best_auc}))
    print(f"Best: {best_name} (cv ROC-AUC {best_auc:.3f}) saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
