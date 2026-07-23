# ===========================
# Imports
# ===========================
import os
import joblib
import pandas as pd
import xgboost as xgb

from huggingface_hub import (
    HfApi,
    hf_hub_download,
    create_repo
)
from huggingface_hub.utils import RepositoryNotFoundError

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# ===========================
# Hugging Face Configuration
# ===========================

HF_TOKEN = os.getenv("HF_TOKEN")

DATASET_REPO = "Aishawarya/Bank-Customer-churn"
MODEL_REPO = "Aishawarya/churn-model"

api = HfApi(token=HF_TOKEN)

# ===========================
# Download Dataset Files
# ===========================

Xtrain_path = hf_hub_download(
    repo_id=DATASET_REPO,
    filename="Xtrain.csv",
    repo_type="dataset",
    token=HF_TOKEN,
)

Xtest_path = hf_hub_download(
    repo_id=DATASET_REPO,
    filename="Xtest.csv",
    repo_type="dataset",
    token=HF_TOKEN,
)

ytrain_path = hf_hub_download(
    repo_id=DATASET_REPO,
    filename="ytrain.csv",
    repo_type="dataset",
    token=HF_TOKEN,
)

ytest_path = hf_hub_download(
    repo_id=DATASET_REPO,
    filename="ytest.csv",
    repo_type="dataset",
    token=HF_TOKEN,
)

print("Dataset downloaded successfully!")

# ===========================
# Load Data
# ===========================

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Convert target from DataFrame to Series
ytrain = ytrain.iloc[:, 0]
ytest = ytest.iloc[:, 0]

# ===========================
# Feature Lists
# ===========================

numeric_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

categorical_features = [
    "Geography",
]

# ===========================
# Handle Class Imbalance
# ===========================

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# ===========================
# Preprocessing
# ===========================

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# ===========================
# Model
# ===========================

xgb_model = xgb.XGBClassifier(
    random_state=42,
    scale_pos_weight=class_weight,
)

model_pipeline = make_pipeline(
    preprocessor,
    xgb_model,
)

param_grid = {
    "xgbclassifier__n_estimators": [50, 75, 100, 125, 150],
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__colsample_bytree": [0.4, 0.5, 0.6],
    "xgbclassifier__colsample_bylevel": [0.4, 0.5, 0.6],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__reg_lambda": [0.4, 0.5, 0.6],
}

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
)

print("Training started...")

grid_search.fit(Xtrain, ytrain)

print("Training completed!")

best_model = grid_search.best_estimator_

# ===========================
# Evaluation
# ===========================

threshold = 0.45

train_probs = best_model.predict_proba(Xtrain)[:, 1]
train_preds = (train_probs >= threshold).astype(int)

test_probs = best_model.predict_proba(Xtest)[:, 1]
test_preds = (test_probs >= threshold).astype(int)

print("\nTraining Report")
print(classification_report(ytrain, train_preds))

print("\nTesting Report")
print(classification_report(ytest, test_preds))

# ===========================
# Save Model
# ===========================

MODEL_FILE = "best_churn_model.joblib"

joblib.dump(best_model, MODEL_FILE)

print("Model saved successfully!")

# ===========================
# Upload Model to Hugging Face
# ===========================

try:
    api.repo_info(
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    print("Model repository already exists.")

except RepositoryNotFoundError:
    print("Creating model repository...")
    create_repo(
        repo_id=MODEL_REPO,
        repo_type="model",
        private=False,
        token=HF_TOKEN,
    )

api.upload_file(
    path_or_fileobj=MODEL_FILE,
    path_in_repo=MODEL_FILE,
    repo_id=MODEL_REPO,
    repo_type="model",
)

print("Model uploaded successfully!")
