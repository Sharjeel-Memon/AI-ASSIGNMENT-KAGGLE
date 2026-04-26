"""
ASSIGNMENT 3 - ITERATION 3 BOOSTED VERSION
Kaggle Playground Series S6E4 - Predicting Irrigation Need

This version is made to try for a better Kaggle score than Iteration 2.
It adds:
1. Extra feature engineering
2. Optional extra labeled file: irrigation_prediction.csv, if present
3. Tuned HistGradientBoosting models
4. Probability multiplier tuning for balanced accuracy
5. Multiple submission files to try on Kaggle

How to run:
    source venv/bin/activate
    python3 FILENAME

Upload first:
    outputs_iter3_boosted/submission.csv

If you still have daily submissions left, you may also try:
    outputs_iter3_boosted/submission_more_high.csv
    outputs_iter3_boosted/submission_less_high.csv
"""

# =========================
# 0. SETTINGS
# =========================

from pathlib import Path
import os
import re
import warnings

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
TARGET_COLUMN = "Irrigation_Need"
ID_COLUMN = "id"
OUTPUT_DIR = Path("outputs_iter3_boosted")

# Use the extra labeled file if it exists in your folder.
# Your extra file appears to have normal feature columns + Irrigation_Need labels.
USE_EXTRA_LABELED_DATA_IF_AVAILABLE = True
EXTRA_LABELED_FILENAME = "irrigation_prediction.csv"

# Turn this on only to test if script runs. Turn it back off for real Kaggle submission.
QUICK_TEST = False
QUICK_TEST_ROWS = 80000

# If your laptop is too slow, set RUN_SECOND_MODEL = False.
# The script will still make a stronger threshold-tuned submission.
RUN_SECOND_MODEL = True

# =========================
# 1. IMPORTS
# =========================

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")


# =========================
# 2. HELPER FUNCTIONS
# =========================

def find_data_folder():
    possible_folders = [
        Path("."),
        Path("/kaggle/input/playground-series-s6e4"),
        Path("/mnt/data"),
    ]
    for folder in possible_folders:
        if (folder / "train.csv").exists() and (folder / "test.csv").exists() and (folder / "sample_submission.csv").exists():
            return folder
    raise FileNotFoundError(
        "Could not find train.csv, test.csv, and sample_submission.csv. "
        "Put this file in the same folder as the CSV files."
    )


def clean_filename(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")


def stratified_sample_df(df, target_col, n_rows, random_state=42):
    if n_rows is None or n_rows >= len(df):
        return df
    sampled, _ = train_test_split(
        df,
        train_size=n_rows,
        random_state=random_state,
        stratify=df[target_col],
    )
    return sampled.reset_index(drop=True)


def add_features(df):
    """Add simple domain-inspired features for irrigation prediction."""
    df = df.copy()

    # Numeric safety helper
    def safe_col(name):
        return pd.to_numeric(df[name], errors="coerce")

    if all(c in df.columns for c in ["Temperature_C", "Sunlight_Hours", "Wind_Speed_kmh", "Humidity", "Soil_Moisture"]):
        df["Dryness_Index"] = (
            safe_col("Temperature_C") * safe_col("Sunlight_Hours") * (safe_col("Wind_Speed_kmh") + 1)
            / (safe_col("Humidity") + safe_col("Soil_Moisture") + 1)
        )

    if all(c in df.columns for c in ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture"]):
        df["Water_Availability"] = safe_col("Rainfall_mm") + safe_col("Previous_Irrigation_mm") + safe_col("Soil_Moisture")
        df["Rainfall_Irrigation_Gap"] = safe_col("Rainfall_mm") - safe_col("Previous_Irrigation_mm")

    if all(c in df.columns for c in ["Rainfall_mm", "Field_Area_hectare"]):
        df["Rainfall_per_Area"] = safe_col("Rainfall_mm") / (safe_col("Field_Area_hectare") + 1)

    if all(c in df.columns for c in ["Previous_Irrigation_mm", "Field_Area_hectare"]):
        df["Prev_Irrigation_per_Area"] = safe_col("Previous_Irrigation_mm") / (safe_col("Field_Area_hectare") + 1)

    if all(c in df.columns for c in ["Soil_Moisture", "Rainfall_mm"]):
        df["Moisture_Rain_Ratio"] = safe_col("Soil_Moisture") / (safe_col("Rainfall_mm") + 1)

    if all(c in df.columns for c in ["Temperature_C", "Humidity"]):
        df["Heat_Humidity_Ratio"] = safe_col("Temperature_C") / (safe_col("Humidity") + 1)

    if "Soil_pH" in df.columns:
        df["pH_Distance_from_Optimal"] = (safe_col("Soil_pH") - 6.5).abs()

    if all(c in df.columns for c in ["Electrical_Conductivity", "Organic_Carbon"]):
        df["EC_Organic_Ratio"] = safe_col("Electrical_Conductivity") / (safe_col("Organic_Carbon") + 0.1)

    # Categorical interaction features. These help tree models split on useful combinations.
    combo_pairs = [
        ("Crop_Type", "Season", "Crop_Season"),
        ("Crop_Type", "Crop_Growth_Stage", "Crop_Stage"),
        ("Soil_Type", "Crop_Type", "Soil_Crop"),
        ("Region", "Season", "Region_Season"),
        ("Irrigation_Type", "Water_Source", "Irrigation_Water"),
        ("Soil_Type", "Region", "Soil_Region"),
    ]

    for left, right, new_col in combo_pairs:
        if left in df.columns and right in df.columns:
            df[new_col] = df[left].astype(str) + "_" + df[right].astype(str)

    return df


def make_preprocessor(X):
    numeric_columns = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_preprocess = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_preprocess = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_preprocess, numeric_columns),
            ("cat", categorical_preprocess, categorical_columns),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_columns, categorical_columns


def save_confusion_matrix(y_true, y_pred, labels, model_name, balanced_acc, output_dir):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, values_format="d", xticks_rotation=45)
    ax.set_title(f"{model_name}\nBalanced Accuracy = {balanced_acc:.5f}")
    filename = output_dir / f"confusion_matrix_{clean_filename(model_name)}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close(fig)
    return filename


def predict_with_multipliers(probs, classes, multipliers):
    multiplier_array = np.array([multipliers.get(cls, 1.0) for cls in classes], dtype=float)
    adjusted = probs * multiplier_array
    return classes[np.argmax(adjusted, axis=1)]


def tune_probability_multipliers(probs, y_true, classes):
    """
    Balanced accuracy cares about each class equally.
    This searches for small probability multipliers that improve class balance.
    Low is fixed at 1.0 because multiplying all classes equally changes nothing.
    """
    best_score = -1.0
    best_multipliers = {"Low": 1.0, "Medium": 1.0, "High": 1.0}
    rows = []

    medium_grid = np.round(np.arange(0.80, 1.26, 0.025), 3)
    high_grid = np.round(np.arange(0.75, 2.31, 0.025), 3)

    for medium_mult in medium_grid:
        for high_mult in high_grid:
            multipliers = {"Low": 1.0, "Medium": float(medium_mult), "High": float(high_mult)}
            preds = predict_with_multipliers(probs, classes, multipliers)
            score = balanced_accuracy_score(y_true, preds)
            rows.append({
                "medium_multiplier": float(medium_mult),
                "high_multiplier": float(high_mult),
                "balanced_accuracy": float(score),
            })
            if score > best_score:
                best_score = score
                best_multipliers = multipliers

    tuning_df = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    return best_multipliers, best_score, tuning_df


def evaluate_model(model_name, model, X_train, y_train, X_valid, y_valid, labels, output_dir):
    print("\n" + "=" * 80)
    print(f"Training model: {model_name}")
    print("=" * 80)

    model.fit(X_train, y_train)

    normal_preds = model.predict(X_valid)
    normal_bal = balanced_accuracy_score(y_valid, normal_preds)
    normal_acc = accuracy_score(y_valid, normal_preds)

    print(f"Normal prediction accuracy:          {normal_acc:.5f}")
    print(f"Normal prediction balanced accuracy: {normal_bal:.5f}")

    probs = model.predict_proba(X_valid)
    classes = model.classes_

    best_multipliers, tuned_bal, tuning_df = tune_probability_multipliers(probs, y_valid, classes)
    tuned_preds = predict_with_multipliers(probs, classes, best_multipliers)
    tuned_acc = accuracy_score(y_valid, tuned_preds)

    print(f"Tuned prediction accuracy:           {tuned_acc:.5f}")
    print(f"Tuned prediction balanced accuracy:  {tuned_bal:.5f}")
    print("Best multipliers:", best_multipliers)
    print("\nClassification report after tuning:")
    print(classification_report(y_valid, tuned_preds, labels=labels))

    cm_file = save_confusion_matrix(
        y_true=y_valid,
        y_pred=tuned_preds,
        labels=labels,
        model_name=model_name + " Tuned",
        balanced_acc=tuned_bal,
        output_dir=output_dir,
    )

    tuning_file = output_dir / f"threshold_tuning_{clean_filename(model_name)}.csv"
    tuning_df.to_csv(tuning_file, index=False)

    return {
        "model_name": model_name,
        "normal_balanced_accuracy": float(normal_bal),
        "tuned_balanced_accuracy": float(tuned_bal),
        "normal_accuracy": float(normal_acc),
        "tuned_accuracy": float(tuned_acc),
        "best_multipliers": best_multipliers,
        "confusion_matrix_file": str(cm_file),
        "threshold_tuning_file": str(tuning_file),
        "trained_model": model,
    }


# =========================
# 3. LOAD DATA
# =========================

OUTPUT_DIR.mkdir(exist_ok=True)
data_folder = find_data_folder()

print(f"Using data folder: {data_folder.resolve()}")
print(f"Saving outputs to: {OUTPUT_DIR.resolve()}")

train = pd.read_csv(data_folder / "train.csv")
test = pd.read_csv(data_folder / "test.csv")
sample_submission = pd.read_csv(data_folder / "sample_submission.csv")

print("\nTrain shape:", train.shape)
print("Test shape:", test.shape)
print("Sample submission shape:", sample_submission.shape)

labels = ["Low", "Medium", "High"]

if TARGET_COLUMN not in train.columns:
    raise ValueError(f"Target column {TARGET_COLUMN!r} not found in train.csv")

if QUICK_TEST:
    print("\nQUICK_TEST is True. Using a smaller sample only for debugging.")
    train = stratified_sample_df(train, TARGET_COLUMN, QUICK_TEST_ROWS, RANDOM_STATE)
    print("Quick test train shape:", train.shape)

print("\nOriginal target distribution:")
print(train[TARGET_COLUMN].value_counts())

# =========================
# 4. OPTIONAL EXTRA LABELED DATA
# =========================

extra_train = None
extra_path = data_folder / EXTRA_LABELED_FILENAME

if USE_EXTRA_LABELED_DATA_IF_AVAILABLE and extra_path.exists():
    try:
        possible_extra = pd.read_csv(extra_path)
        raw_feature_columns = [c for c in train.columns if c not in [TARGET_COLUMN, ID_COLUMN]]
        has_target = TARGET_COLUMN in possible_extra.columns
        has_features = all(c in possible_extra.columns for c in raw_feature_columns)

        if has_target and has_features:
            extra_train = possible_extra[raw_feature_columns + [TARGET_COLUMN]].copy()
            print(f"\nExtra labeled data found and will be used: {EXTRA_LABELED_FILENAME}")
            print("Extra labeled data shape:", extra_train.shape)
            print("Extra target distribution:")
            print(extra_train[TARGET_COLUMN].value_counts())
        else:
            print(f"\n{EXTRA_LABELED_FILENAME} found, but columns do not match. Skipping it.")
    except Exception as exc:
        print(f"\nCould not read {EXTRA_LABELED_FILENAME}. Skipping it. Reason: {exc}")
else:
    print("\nNo extra labeled data used.")


# =========================
# 5. FEATURE ENGINEERING
# =========================

raw_feature_columns = [c for c in train.columns if c not in [TARGET_COLUMN, ID_COLUMN]]

X_original = train[raw_feature_columns].copy()
y_original = train[TARGET_COLUMN].copy()
X_test = test[raw_feature_columns].copy()
test_ids = test[ID_COLUMN].copy()

X_original = add_features(X_original)
X_test = add_features(X_test)

if extra_train is not None:
    X_extra = add_features(extra_train[raw_feature_columns].copy())
    y_extra = extra_train[TARGET_COLUMN].copy()
else:
    X_extra = None
    y_extra = None

# Make sure train/test columns match after feature engineering
X_test = X_test.reindex(columns=X_original.columns)
if X_extra is not None:
    X_extra = X_extra.reindex(columns=X_original.columns)

preprocessor, numeric_columns, categorical_columns = make_preprocessor(X_original)

print("\nFeature count after engineering:", X_original.shape[1])
print("Numeric columns:", len(numeric_columns))
print("Categorical columns:", len(categorical_columns))


# =========================
# 6. TRAIN/VALIDATION SPLIT
# =========================

X_train_base, X_valid, y_train_base, y_valid = train_test_split(
    X_original,
    y_original,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_original,
)

if X_extra is not None:
    X_train = pd.concat([X_train_base, X_extra], axis=0, ignore_index=True)
    y_train = pd.concat([y_train_base, y_extra], axis=0, ignore_index=True)
else:
    X_train = X_train_base
    y_train = y_train_base

print("\nTraining rows for validation models:", X_train.shape[0])
print("Validation rows:", X_valid.shape[0])


# =========================
# 7. MODELS
# =========================

models = {}

models["HistGradientBoosting V3 Balanced"] = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", HistGradientBoostingClassifier(
        learning_rate=0.045,
        max_iter=750,
        max_leaf_nodes=63,
        min_samples_leaf=35,
        l2_regularization=0.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )),
])

if RUN_SECOND_MODEL:
    models["HistGradientBoosting V3 Smooth"] = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=850,
            max_leaf_nodes=45,
            min_samples_leaf=45,
            l2_regularization=0.02,
            class_weight="balanced",
            random_state=RANDOM_STATE + 7,
        )),
    ])


# =========================
# 8. EVALUATE + THRESHOLD TUNE
# =========================

results = []
trained_models = {}

for model_name, model in models.items():
    result = evaluate_model(
        model_name=model_name,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        labels=labels,
        output_dir=OUTPUT_DIR,
    )
    trained_models[model_name] = result["trained_model"]
    result_for_table = {k: v for k, v in result.items() if k != "trained_model"}
    result_for_table["best_multipliers"] = str(result_for_table["best_multipliers"])
    results.append(result_for_table)

results_df = pd.DataFrame(results).sort_values("tuned_balanced_accuracy", ascending=False)
results_file = OUTPUT_DIR / "validation_scores_iter3.csv"
results_df.to_csv(results_file, index=False)

print("\n" + "=" * 80)
print("ITERATION 3 VALIDATION RESULTS")
print("=" * 80)
print(results_df)
print(f"\nSaved validation results: {results_file}")

best_model_name = results_df.iloc[0]["model_name"]
best_multipliers = eval(results_df.iloc[0]["best_multipliers"])
best_validation_score = float(results_df.iloc[0]["tuned_balanced_accuracy"])

print("\nBest Iteration 3 model:", best_model_name)
print("Best tuned validation balanced accuracy:", best_validation_score)
print("Best multipliers:", best_multipliers)


# =========================
# 9. TRAIN FINAL MODEL ON FULL DATA
# =========================

print("\n" + "=" * 80)
print(f"Training final full-data model: {best_model_name}")
print("=" * 80)

if X_extra is not None:
    X_full = pd.concat([X_original, X_extra], axis=0, ignore_index=True)
    y_full = pd.concat([y_original, y_extra], axis=0, ignore_index=True)
else:
    X_full = X_original
    y_full = y_original

final_model = clone(models[best_model_name])
final_model.fit(X_full, y_full)

print("Predicting test probabilities...")
test_probs = final_model.predict_proba(X_test)
classes = final_model.classes_


def make_submission(filename, multipliers):
    preds = predict_with_multipliers(test_probs, classes, multipliers)
    submission = sample_submission.copy()
    submission[ID_COLUMN] = test_ids
    submission[TARGET_COLUMN] = preds

    bad_labels = set(submission[TARGET_COLUMN].unique()) - set(labels)
    if bad_labels:
        raise ValueError(f"Bad prediction labels found: {bad_labels}")

    out_file = OUTPUT_DIR / filename
    submission.to_csv(out_file, index=False)
    print(f"\nSaved {out_file}")
    print("Prediction distribution:")
    print(submission[TARGET_COLUMN].value_counts())
    return out_file

# Main submission = best validation multipliers
main_file = make_submission("submission.csv", best_multipliers)

# Two small variants. These sometimes do better on Kaggle public LB.
more_high_multipliers = dict(best_multipliers)
more_high_multipliers["High"] = float(more_high_multipliers.get("High", 1.0) * 1.05)
more_high_multipliers["Medium"] = float(more_high_multipliers.get("Medium", 1.0) * 0.99)
make_submission("submission_more_high.csv", more_high_multipliers)

less_high_multipliers = dict(best_multipliers)
less_high_multipliers["High"] = float(less_high_multipliers.get("High", 1.0) * 0.95)
less_high_multipliers["Medium"] = float(less_high_multipliers.get("Medium", 1.0) * 1.01)
make_submission("submission_less_high.csv", less_high_multipliers)


# =========================
# 10. REPORT NOTES
# =========================

report_notes = OUTPUT_DIR / "report_notes_iter3.txt"
with open(report_notes, "w", encoding="utf-8") as f:
    f.write("ASSIGNMENT 3 - ITERATION 3 REPORT NOTES\n")
    f.write("=======================================\n\n")
    f.write("Iteration 3 improvement summary:\n")
    f.write("- Added engineered numeric features such as Dryness_Index, Water_Availability, and pH_Distance_from_Optimal.\n")
    f.write("- Added categorical combination features such as Crop_Season, Crop_Stage, and Soil_Crop.\n")
    if X_extra is not None:
        f.write(f"- Used extra labeled data file: {EXTRA_LABELED_FILENAME}.\n")
    else:
        f.write("- No extra labeled data was used.\n")
    f.write("- Used HistGradientBoostingClassifier with class_weight='balanced'.\n")
    f.write("- Tuned prediction probability multipliers on validation data to directly improve balanced accuracy.\n\n")

    f.write("Validation results:\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")
    f.write(f"Final selected model: {best_model_name}\n")
    f.write(f"Final validation balanced accuracy: {best_validation_score:.5f}\n")
    f.write(f"Probability multipliers: {best_multipliers}\n\n")
    f.write("Upload order suggestion:\n")
    f.write("1. submission.csv\n")
    f.write("2. submission_more_high.csv if you still have submissions and want to test a slightly stronger High-class correction\n")
    f.write("3. submission_less_high.csv if the first two do not improve\n")

print(f"\nSaved report notes: {report_notes}")

print("\n" + "=" * 80)
print("DONE - ITERATION 3 BOOSTED")
print("=" * 80)
print(f"Upload first: {main_file}")
print("If you have more Kaggle submissions left, try these next:")
print(f"2nd: {OUTPUT_DIR / 'submission_more_high.csv'}")
print(f"3rd: {OUTPUT_DIR / 'submission_less_high.csv'}")
print(f"All Iteration 3 files saved in: {OUTPUT_DIR.resolve()}")
