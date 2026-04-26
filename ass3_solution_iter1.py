"""
ASSIGNMENT 3 - Kaggle Playground Series S6E4
Predicting Irrigation Need

What this script does:
1. Loads train.csv, test.csv, sample_submission.csv
2. Tries required ML models:
   - Decision Tree
   - Naive Bayes
   - K-Means as a classifier
   - Logistic Regression
   - HistGradientBoosting as an advanced model
3. Uses balanced accuracy because Kaggle uses balanced accuracy
4. Saves confusion matrix images for your report
5. Chooses the best validation model
6. Trains that best model again on the full train.csv
7. Creates outputs/submission.csv for Kaggle

How to run:
    python ass3_solution.py

If your computer is slow, change QUICK_TEST = True below first, then run.
For final Kaggle submission, QUICK_TEST must be False.
"""

# =========================
# 0. SETTINGS YOU CAN EDIT
# =========================

QUICK_TEST = False
# True  = use only a small sample to check that code works
# False = use full training data for final assignment submission

QUICK_TEST_ROWS = 50000

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

# K-Fold is used for the report. To keep it practical, it uses a stratified sample by default.
RUN_KFOLD_FOR_BEST_MODEL = True
KFOLD_SAMPLE_ROWS = 150000  # set to None to run K-Fold on full data
KFOLD_SPLITS = 5

# Optional slower model. Keep False unless your machine is strong.
RUN_RANDOM_FOREST = False


# =========================
# 1. IMPORT LIBRARIES
# =========================

import os
import re
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from sklearn.cluster import MiniBatchKMeans
except Exception as e:
    raise ImportError("scikit-learn is required. Install it with: pip install scikit-learn") from e

warnings.filterwarnings("ignore")


# =========================
# 2. HELPER FUNCTIONS
# =========================

def find_data_folder():
    """
    Finds where train.csv/test.csv/sample_submission.csv are stored.
    Works for:
    - Kaggle notebooks
    - Local folder
    - ChatGPT /mnt/data folder
    """
    possible_folders = [
        Path("."),
        Path("/kaggle/input/playground-series-s6e4"),
        Path("/mnt/data"),
    ]

    for folder in possible_folders:
        if (
            (folder / "train.csv").exists()
            and (folder / "test.csv").exists()
            and (folder / "sample_submission.csv").exists()
        ):
            return folder

    raise FileNotFoundError(
        "Could not find train.csv, test.csv, and sample_submission.csv.\n"
        "Put this Python file in the same folder as those 3 CSV files, then run again."
    )


def make_one_hot_encoder():
    """
    Handles both old and new scikit-learn versions.
    New versions use sparse_output.
    Old versions use sparse.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def clean_filename(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def stratified_sample(X, y, n_rows, random_state=42):
    """
    Takes a stratified sample so class proportions stay similar.
    """
    if n_rows is None or n_rows >= len(X):
        return X, y

    sample_fraction = n_rows / len(X)
    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=sample_fraction,
        random_state=random_state,
        stratify=y,
    )
    return X_sample, y_sample


class KMeansAsClassifier(BaseEstimator, ClassifierMixin):
    """
    K-Means is an unsupervised clustering algorithm, not naturally a classifier.
    This wrapper turns clusters into class predictions:
    1. Fit K-Means with 3 clusters
    2. For each cluster, find the most common true label in that cluster
    3. Predict that majority label for future rows in the same cluster

    This is mainly included because the assignment asks you to try K-Means.
    It is usually weaker than real supervised classifiers.
    """

    def __init__(self, n_clusters=3, random_state=42, batch_size=4096):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.batch_size = batch_size

    def fit(self, X, y):
        y_array = np.asarray(y)
        self.classes_ = np.unique(y_array)
        self.default_class_ = Counter(y_array).most_common(1)[0][0]

        self.kmeans_ = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            batch_size=self.batch_size,
            n_init=10,
        )

        cluster_ids = self.kmeans_.fit_predict(X)

        self.cluster_to_class_ = {}
        for cluster_id in range(self.n_clusters):
            labels_in_cluster = y_array[cluster_ids == cluster_id]
            if len(labels_in_cluster) == 0:
                self.cluster_to_class_[cluster_id] = self.default_class_
            else:
                self.cluster_to_class_[cluster_id] = Counter(labels_in_cluster).most_common(1)[0][0]

        return self

    def predict(self, X):
        cluster_ids = self.kmeans_.predict(X)
        return np.array([
            self.cluster_to_class_.get(cluster_id, self.default_class_)
            for cluster_id in cluster_ids
        ])


def save_confusion_matrix(y_true, y_pred, labels, model_name, balanced_acc, output_dir):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, values_format="d", xticks_rotation=45)
    ax.set_title(f"{model_name}\nBalanced Accuracy = {balanced_acc:.4f}")

    filename = output_dir / f"confusion_matrix_{clean_filename(model_name)}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close(fig)

    return filename


def evaluate_model(model_name, model, X_train, X_valid, y_train, y_valid, labels, output_dir):
    print("\n" + "=" * 80)
    print(f"Training model: {model_name}")
    print("=" * 80)

    model.fit(X_train, y_train)
    preds = model.predict(X_valid)

    bal_acc = balanced_accuracy_score(y_valid, preds)
    normal_acc = accuracy_score(y_valid, preds)

    print(f"Normal accuracy:   {normal_acc:.5f}")
    print(f"Balanced accuracy: {bal_acc:.5f}")
    print("\nClassification report:")
    print(classification_report(y_valid, preds, labels=labels))

    cm_file = save_confusion_matrix(
        y_true=y_valid,
        y_pred=preds,
        labels=labels,
        model_name=model_name,
        balanced_acc=bal_acc,
        output_dir=output_dir,
    )

    print(f"Saved confusion matrix: {cm_file}")

    return {
        "model_name": model_name,
        "balanced_accuracy": bal_acc,
        "normal_accuracy": normal_acc,
        "confusion_matrix_file": str(cm_file),
    }


# =========================
# 3. LOAD DATA
# =========================

data_folder = find_data_folder()
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

print(f"Using data folder: {data_folder.resolve()}")

train = pd.read_csv(data_folder / "train.csv")
test = pd.read_csv(data_folder / "test.csv")
sample_submission = pd.read_csv(data_folder / "sample_submission.csv")

print("\nTrain shape:", train.shape)
print("Test shape:", test.shape)
print("Sample submission shape:", sample_submission.shape)

TARGET_COLUMN = "Irrigation_Need"
ID_COLUMN = "id"

if TARGET_COLUMN not in train.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in train.csv")

print("\nTarget distribution:")
print(train[TARGET_COLUMN].value_counts())

print("\nMissing values in train.csv:")
print(train.isna().sum())

# For a quick debug run only
if QUICK_TEST:
    print("\nQUICK_TEST is True, so using a smaller stratified sample.")
    X_temp = train.drop(columns=[TARGET_COLUMN])
    y_temp = train[TARGET_COLUMN]
    X_temp, y_temp = stratified_sample(X_temp, y_temp, QUICK_TEST_ROWS, RANDOM_STATE)
    train = pd.concat([X_temp, y_temp], axis=1)
    print("Quick-test train shape:", train.shape)

# Split features and target
X = train.drop(columns=[TARGET_COLUMN])
y = train[TARGET_COLUMN]

test_ids = test[ID_COLUMN].copy()

# Do not use id as a model feature
X = X.drop(columns=[ID_COLUMN], errors="ignore")
X_test = test.drop(columns=[ID_COLUMN], errors="ignore")

labels = ["Low", "Medium", "High"]

# Detect numeric and categorical columns automatically
numeric_columns = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("\nNumeric columns:")
print(numeric_columns)

print("\nCategorical columns:")
print(categorical_columns)


# =========================
# 4. PREPROCESSING
# =========================

numeric_basic = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

numeric_scaled = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_ordinal = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

categorical_onehot = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", make_one_hot_encoder()),
])

# Good for tree-style models and Naive Bayes
ordinal_preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_basic, numeric_columns),
        ("cat", categorical_ordinal, categorical_columns),
    ],
    remainder="drop",
)

# Good for Logistic Regression and K-Means
onehot_scaled_preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_scaled, numeric_columns),
        ("cat", categorical_onehot, categorical_columns),
    ],
    remainder="drop",
)


# =========================
# 5. TRAIN/VALIDATION SPLIT
# =========================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining rows:", X_train.shape[0])
print("Validation rows:", X_valid.shape[0])


# =========================
# 6. MODELS TO TRY
# =========================

models = {
    "Decision Tree": Pipeline(steps=[
        ("preprocess", ordinal_preprocess),
        ("model", DecisionTreeClassifier(
            max_depth=14,
            min_samples_leaf=80,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ]),

    "Naive Bayes": Pipeline(steps=[
        ("preprocess", ordinal_preprocess),
        ("model", GaussianNB()),
    ]),

    "K-Means as Classifier": Pipeline(steps=[
        ("preprocess", onehot_scaled_preprocess),
        ("model", KMeansAsClassifier(
            n_clusters=3,
            random_state=RANDOM_STATE,
            batch_size=4096,
        )),
    ]),

    "Logistic Regression": Pipeline(steps=[
        ("preprocess", onehot_scaled_preprocess),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ]),

    "HistGradientBoosting Advanced": Pipeline(steps=[
        ("preprocess", ordinal_preprocess),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=350,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=RANDOM_STATE,
        )),
    ]),
}

if RUN_RANDOM_FOREST:
    models["Random Forest Advanced"] = Pipeline(steps=[
        ("preprocess", ordinal_preprocess),
        ("model", RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


# =========================
# 7. EVALUATE MODELS
# =========================

results = []

for model_name, model in models.items():
    result = evaluate_model(
        model_name=model_name,
        model=model,
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid,
        labels=labels,
        output_dir=output_dir,
    )
    results.append(result)

results_df = pd.DataFrame(results).sort_values("balanced_accuracy", ascending=False)
results_file = output_dir / "validation_scores.csv"
results_df.to_csv(results_file, index=False)

print("\n" + "=" * 80)
print("VALIDATION RESULTS - sorted by balanced accuracy")
print("=" * 80)
print(results_df)
print(f"\nSaved validation scores: {results_file}")


# =========================
# 8. K-FOLD CV FOR BEST MODEL
# =========================

best_model_name = results_df.iloc[0]["model_name"]
best_validation_score = results_df.iloc[0]["balanced_accuracy"]

print("\nBest validation model:", best_model_name)
print("Best validation balanced accuracy:", best_validation_score)

kfold_scores = None

if RUN_KFOLD_FOR_BEST_MODEL:
    print("\n" + "=" * 80)
    print(f"Running {KFOLD_SPLITS}-Fold CV for best model: {best_model_name}")
    print("=" * 80)

    X_cv, y_cv = stratified_sample(X, y, KFOLD_SAMPLE_ROWS, RANDOM_STATE)
    print("K-Fold rows used:", X_cv.shape[0])

    cv = StratifiedKFold(
        n_splits=KFOLD_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scorer = make_scorer(balanced_accuracy_score)

    best_model_for_cv = clone(models[best_model_name])
    kfold_scores = cross_val_score(
        best_model_for_cv,
        X_cv,
        y_cv,
        scoring=scorer,
        cv=cv,
        n_jobs=None,
    )

    kfold_file = output_dir / "kfold_scores.csv"
    pd.DataFrame({
        "fold": list(range(1, len(kfold_scores) + 1)),
        "balanced_accuracy": kfold_scores,
    }).to_csv(kfold_file, index=False)

    print("K-Fold balanced accuracy scores:", kfold_scores)
    print("K-Fold mean:", kfold_scores.mean())
    print("K-Fold std:", kfold_scores.std())
    print(f"Saved K-Fold scores: {kfold_file}")


# =========================
# 9. TRAIN BEST MODEL ON FULL TRAINING DATA
# =========================

print("\n" + "=" * 80)
print(f"Training final model on full train.csv: {best_model_name}")
print("=" * 80)

final_model = clone(models[best_model_name])
final_model.fit(X, y)

print("Predicting test.csv...")
test_predictions = final_model.predict(X_test)

# Create submission in exact Kaggle format
submission = sample_submission.copy()
submission[ID_COLUMN] = test_ids
submission[TARGET_COLUMN] = test_predictions

# Safety checks
allowed_labels = set(labels)
bad_predictions = set(submission[TARGET_COLUMN].unique()) - allowed_labels
if bad_predictions:
    raise ValueError(f"Unexpected predictions found: {bad_predictions}")

submission_file = output_dir / "submission.csv"
submission.to_csv(submission_file, index=False)

print("\nSubmission preview:")
print(submission.head())

print("\nPrediction distribution:")
print(submission[TARGET_COLUMN].value_counts())

print(f"\nSaved Kaggle submission file: {submission_file}")


# =========================
# 10. SAVE REPORT NOTES
# =========================

report_notes = output_dir / "report_notes.txt"

with open(report_notes, "w", encoding="utf-8") as f:
    f.write("ASSIGNMENT 3 REPORT NOTES\n")
    f.write("=========================\n\n")

    f.write("Competition Overview\n")
    f.write("--------------------\n")
    f.write("Competition: Playground Series - Season 6, Episode 4: Predicting Irrigation Need\n")
    f.write("Problem Type: Multiclass classification\n")
    f.write("Target: Irrigation_Need with classes Low, Medium, High\n")
    f.write("Evaluation Metric: Balanced Accuracy\n\n")

    f.write("Dataset\n")
    f.write("-------\n")
    f.write(f"Train shape: {train.shape}\n")
    f.write(f"Test shape: {test.shape}\n")
    f.write("Target distribution:\n")
    f.write(str(train[TARGET_COLUMN].value_counts()))
    f.write("\n\n")

    f.write("Preprocessing\n")
    f.write("-------------\n")
    f.write("Dropped id column from model features.\n")
    f.write("Numeric columns: median imputation; scaling for Logistic Regression and K-Means.\n")
    f.write("Categorical columns: ordinal encoding for tree/Naive Bayes/HistGradientBoosting; one-hot encoding for Logistic Regression and K-Means.\n")
    f.write("Train-validation split: stratified 80/20 split.\n\n")

    f.write("Validation Results\n")
    f.write("------------------\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")

    if kfold_scores is not None:
        f.write("K-Fold Cross Validation\n")
        f.write("-----------------------\n")
        f.write(f"Best model tested with {KFOLD_SPLITS}-Fold StratifiedKFold.\n")
        f.write(f"Rows used for K-Fold: {len(y_cv)}\n")
        f.write(f"Scores: {kfold_scores}\n")
        f.write(f"Mean balanced accuracy: {kfold_scores.mean():.5f}\n")
        f.write(f"Std: {kfold_scores.std():.5f}\n\n")
    else:
        f.write("K-Fold Cross Validation\n")
        f.write("-----------------------\n")
        f.write("K-Fold was disabled in the script.\n\n")

    f.write("Final Model\n")
    f.write("-----------\n")
    f.write(f"Selected model: {best_model_name}\n")
    f.write(f"Validation balanced accuracy: {best_validation_score:.5f}\n")
    f.write("Reason selected: highest validation balanced accuracy among attempted models.\n\n")

    f.write("Failed Attempts and Insights\n")
    f.write("----------------------------\n")
    f.write("Use the lower-performing models from validation_scores.csv as failed attempts.\n")
    f.write("Attach their confusion matrix PNG files from the outputs folder.\n")
    f.write("Common insight: K-Means is unsupervised, so its clusters may not match Low/Medium/High labels well.\n")
    f.write("Naive Bayes assumes feature independence, which is often unrealistic for tabular agricultural/environmental data.\n")
    f.write("Decision Tree may overfit or underfit depending on depth.\n\n")

    f.write("Leaderboard\n")
    f.write("-----------\n")
    f.write("After uploading submission.csv to Kaggle, write your Kaggle score and rank here.\n")
    f.write("Attach screenshot showing your name and score.\n")

print(f"Saved report notes: {report_notes}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print(f"Upload this file to Kaggle: {submission_file}")
print(f"Use these files in your report: {output_dir.resolve()}")
print("For the assignment, submit:")
print("1. This Python file")
print("2. outputs/submission.csv")
print("3. Kaggle leaderboard screenshot")
print("4. Your report with confusion matrices")
