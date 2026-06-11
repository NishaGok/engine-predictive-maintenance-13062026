
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt
import seaborn as sns

from huggingface_hub import (
    HfApi,
    create_repo
)

from huggingface_hub.utils import (
    RepositoryNotFoundError
)

from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    GradientBoostingClassifier
)

from sklearn.model_selection import (
    GridSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# MLflow Setup
# --------------------------------------------------

mlflow.set_experiment(
    "Predictive_Maintenance_Experiment"
)

# --------------------------------------------------
# Hugging Face Setup
# --------------------------------------------------

api = HfApi(
    token=os.getenv("HF_TOKEN1")
)

# --------------------------------------------------
# Load Train/Test Data
# --------------------------------------------------

Xtrain = pd.read_csv(
    "hf://datasets/"
    "NishaGok/engine-predictive-maintenance-13062026/"
    "Xtrain.csv"
)

Xtest = pd.read_csv(
    "hf://datasets/"
    "NishaGok/engine-predictive-maintenance-13062026/"
    "Xtest.csv"
)

ytrain = pd.read_csv(
    "hf://datasets/"
    "NishaGok/engine-predictive-maintenance-13062026/"
    "ytrain.csv"
).squeeze()

ytest = pd.read_csv(
    "hf://datasets/"
    "NishaGok/engine-predictive-maintenance-13062026/"
    "ytest.csv"
).squeeze()

print("Train/Test datasets loaded")

# --------------------------------------------------
# Model Pipeline
# --------------------------------------------------

pipeline = Pipeline([
    (
        "model",
        GradientBoostingClassifier(
            random_state=42
        )
    )
])

# --------------------------------------------------
# Hyperparameter Grid
# --------------------------------------------------

param_grid = {

    "model__n_estimators": [100, 200],

    "model__learning_rate": [0.05, 0.1],

    "model__max_depth": [3, 5],

    "model__min_samples_split": [2, 5],

    "model__min_samples_leaf": [1, 2]
}

# --------------------------------------------------
# Grid Search
# --------------------------------------------------

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="f1",
    verbose=1,
    n_jobs=-1
)

# --------------------------------------------------
# Training + Tracking
# --------------------------------------------------

with mlflow.start_run(
    run_name="GradientBoosting_Tuning"
):

    # --------------------------
    # Train
    # --------------------------

    grid_search.fit(
        Xtrain,
        ytrain
    )

    # --------------------------
    # Best Model
    # --------------------------

    best_model = (
        grid_search.best_estimator_
    )

    # --------------------------
    # Log Parameters
    # --------------------------

    mlflow.log_params(
        grid_search.best_params_
    )

    mlflow.log_metric(
        "best_cv_score",
        grid_search.best_score_
    )

    print(
        "Best Parameters:",
        grid_search.best_params_
    )

    # --------------------------
    # Predictions
    # --------------------------

    preds = best_model.predict(
        Xtest
    )

    probs = best_model.predict_proba(
        Xtest
    )[:, 1]

    # --------------------------
    # Metrics
    # --------------------------

    final_accuracy = accuracy_score(
        ytest,
        preds
    )

    final_precision = precision_score(
        ytest,
        preds
    )

    final_recall = recall_score(
        ytest,
        preds
    )

    final_f1 = f1_score(
        ytest,
        preds
    )

    final_roc_auc = roc_auc_score(
        ytest,
        probs
    )

    mlflow.log_metric(
        "accuracy",
        final_accuracy
    )

    mlflow.log_metric(
        "precision",
        final_precision
    )

    mlflow.log_metric(
        "recall",
        final_recall
    )

    mlflow.log_metric(
        "f1_score",
        final_f1
    )

    mlflow.log_metric(
        "roc_auc_score",
        final_roc_auc
    )

    print(
        classification_report(
            ytest,
            preds
        )
    )

    # --------------------------
    # Confusion Matrix
    # --------------------------

    cm = confusion_matrix(
        ytest,
        preds
    )

    plt.figure(
        figsize=(6,4)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    plt.savefig(
        "confusion_matrix.png"
    )

    mlflow.log_artifact(
        "confusion_matrix.png"
    )

    # --------------------------
    # Save Model
    # --------------------------

    model_path = (
        "best_engine_predictive_model.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    mlflow.log_artifact(
        model_path
    )

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="gradient_boosting_model",
        input_example=Xtrain.iloc[:5]
    )

# --------------------------------------------------
# Create HF Model Repo
# --------------------------------------------------

repo_id = (
    "NishaGok/"
    "engine-predictive-maintenance-model-13062026"
)

repo_type = "model"

try:

    api.repo_info(
        repo_id=repo_id,
        repo_type=repo_type
    )

    print(
        "Model repo already exists"
    )

except RepositoryNotFoundError:

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False
    )

    print(
        "Model repo created"
    )

# --------------------------------------------------
# Upload Model
# --------------------------------------------------

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type="model"
)

print(
    "Model uploaded successfully"
)
