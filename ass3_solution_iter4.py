"""
ASSIGNMENT 3 - ITERATION 4 ENSEMBLE/LADDER VERSION
Kaggle Playground Series S6E4 - Predicting Irrigation Need

Goal: try to beat Iteration 3 public score 0.96823.

What this version does:
1. Uses the same feature engineering as Iteration 3.
2. Uses irrigation_prediction.csv as extra labeled data if available.
3. Trains several HistGradientBoosting models with slightly different settings.
4. Averages their probabilities (soft-voting ensemble).
5. Tunes class probability multipliers on validation balanced accuracy.
6. Creates several Kaggle submissions with small High-class ladder variants.

How to run:
    source venv/bin/activate
    python ass3_solution_iter4_ensemble.py

Upload order suggestion:
    1. outputs_iter4_ensemble/submission_iter4_ensemble.csv
    2. outputs_iter4_ensemble/submission_iter4_high_plus_02.csv
    3. outputs_iter4_ensemble/submission_iter4_high_plus_05.csv
    4. outputs_iter4_ensemble/submission_iter4_high_plus_08.csv
    5. outputs_iter4_ensemble/submission_iter4_medium_plus_high_plus.csv
"""

from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
TARGET_COLUMN = "Irrigation_Need"
ID_COLUMN = "id"
OUTPUT_DIR = Path("outputs_iter4_ensemble")
EXTRA_LABELED_FILENAME = "irrigation_prediction.csv"
USE_EXTRA_LABELED_DATA_IF_AVAILABLE = True

# If your laptop is struggling, set this to True just to test the script.
QUICK_TEST = False
QUICK_TEST_ROWS = 100000

LABELS = ["Low", "Medium", "High"]


def find_data_folder():
    possible_folders = [Path("."), Path("/kaggle/input/playground-series-s6e4"), Path("/mnt/data")]
    for folder in possible_folders:
        if (folder / "train.csv").exists() and (folder / "test.csv").exists() and (folder / "sample_submission.csv").exists():
            return folder
    raise FileNotFoundError("Put this script in the same folder as train.csv, test.csv, and sample_submission.csv")


def clean_filename(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")


def stratified_sample_df(df, target_col, n_rows, random_state=42):
    if n_rows is None or n_rows >= len(df):
        return df
    sampled, _ = train_test_split(df, train_size=n_rows, random_state=random_state, stratify=df[target_col])
    return sampled.reset_index(drop=True)


def add_features(df):
    df = df.copy()

    def safe_col(name):
        return pd.to_numeric(df[name], errors="coerce")

    # Core domain features
    if all(c in df.columns for c in ["Temperature_C", "Sunlight_Hours", "Wind_Speed_kmh", "Humidity", "Soil_Moisture"]):
        df["Dryness_Index"] = safe_col("Temperature_C") * safe_col("Sunlight_Hours") * (safe_col("Wind_Speed_kmh") + 1) / (safe_col("Humidity") + safe_col("Soil_Moisture") + 1)
        df["Evaporation_Pressure"] = safe_col("Temperature_C") * safe_col("Sunlight_Hours") / (safe_col("Humidity") + 1)
        df["Wind_Dryness"] = safe_col("Wind_Speed_kmh") * safe_col("Temperature_C") / (safe_col("Soil_Moisture") + 1)

    if all(c in df.columns for c in ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture"]):
        df["Water_Availability"] = safe_col("Rainfall_mm") + safe_col("Previous_Irrigation_mm") + safe_col("Soil_Moisture")
        df["Rainfall_Irrigation_Gap"] = safe_col("Rainfall_mm") - safe_col("Previous_Irrigation_mm")
        df["Total_Water_Input"] = safe_col("Rainfall_mm") + safe_col("Previous_Irrigation_mm")
        df["Water_Input_x_Moisture"] = (safe_col("Rainfall_mm") + safe_col("Previous_Irrigation_mm")) * safe_col("Soil_Moisture")

    if all(c in df.columns for c in ["Rainfall_mm", "Field_Area_hectare"]):
        df["Rainfall_per_Area"] = safe_col("Rainfall_mm") / (safe_col("Field_Area_hectare") + 1)

    if all(c in df.columns for c in ["Previous_Irrigation_mm", "Field_Area_hectare"]):
        df["Prev_Irrigation_per_Area"] = safe_col("Previous_Irrigation_mm") / (safe_col("Field_Area_hectare") + 1)

    if all(c in df.columns for c in ["Soil_Moisture", "Rainfall_mm"]):
        df["Moisture_Rain_Ratio"] = safe_col("Soil_Moisture") / (safe_col("Rainfall_mm") + 1)

    if all(c in df.columns for c in ["Temperature_C", "Humidity"]):
        df["Heat_Humidity_Ratio"] = safe_col("Temperature_C") / (safe_col("Humidity") + 1)
        df["Temp_x_Humidity"] = safe_col("Temperature_C") * safe_col("Humidity")

    if all(c in df.columns for c in ["Temperature_C", "Soil_Moisture"]):
        df["Temp_x_SoilMoisture"] = safe_col("Temperature_C") * safe_col("Soil_Moisture")

    if "Soil_pH" in df.columns:
        df["pH_Distance_from_Optimal"] = (safe_col("Soil_pH") - 6.5).abs()
        df["pH_Acidic_Flag"] = (safe_col("Soil_pH") < 6.0).astype(int)
        df["pH_Alkaline_Flag"] = (safe_col("Soil_pH") > 7.5).astype(int)

    if all(c in df.columns for c in ["Electrical_Conductivity", "Organic_Carbon"]):
        df["EC_Organic_Ratio"] = safe_col("Electrical_Conductivity") / (safe_col("Organic_Carbon") + 0.1)
        df["EC_x_Organic"] = safe_col("Electrical_Conductivity") * safe_col("Organic_Carbon")

    # Bin a few continuous features. Tree models sometimes benefit from coarse buckets.
    for col, bins in [
        ("Soil_Moisture", 8),
        ("Temperature_C", 8),
        ("Rainfall_mm", 8),
        ("Previous_Irrigation_mm", 8),
        ("Humidity", 8),
    ]:
        if col in df.columns:
            try:
                df[f"{col}_bin"] = pd.qcut(safe_col(col), q=bins, duplicates="drop").astype(str)
            except Exception:
                df[f"{col}_bin"] = "unknown"

    combo_pairs = [
        ("Crop_Type", "Season", "Crop_Season"),
        ("Crop_Type", "Crop_Growth_Stage", "Crop_Stage"),
        ("Soil_Type", "Crop_Type", "Soil_Crop"),
        ("Region", "Season", "Region_Season"),
        ("Irrigation_Type", "Water_Source", "Irrigation_Water"),
        ("Soil_Type", "Region", "Soil_Region"),
        ("Crop_Growth_Stage", "Season", "Stage_Season"),
        ("Mulching_Used", "Irrigation_Type", "Mulch_Irrigation"),
    ]
    for left, right, new_col in combo_pairs:
        if left in df.columns and right in df.columns:
            df[new_col] = df[left].astype(str) + "_" + df[right].astype(str)

    return df


def make_preprocessor(X):
    numeric_columns = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_preprocess = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
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


def predict_with_multipliers(probs, classes, multipliers):
    arr = np.array([multipliers.get(cls, 1.0) for cls in classes], dtype=float)
    adjusted = probs * arr
    return classes[np.argmax(adjusted, axis=1)]


def tune_probability_multipliers(probs, y_true, classes):
    best_score = -1.0
    best_multipliers = {"Low": 1.0, "Medium": 1.0, "High": 1.0}
    rows = []

    # Wider and finer than before around the area that helped your public LB.
    medium_grid = np.round(np.arange(0.78, 1.18, 0.02), 3)
    high_grid = np.round(np.arange(0.85, 2.80, 0.02), 3)

    for medium_mult in medium_grid:
        for high_mult in high_grid:
            mult = {"Low": 1.0, "Medium": float(medium_mult), "High": float(high_mult)}
            preds = predict_with_multipliers(probs, classes, mult)
            score = balanced_accuracy_score(y_true, preds)
            rows.append({"medium_multiplier": medium_mult, "high_multiplier": high_mult, "balanced_accuracy": score})
            if score > best_score:
                best_score = score
                best_multipliers = mult

    df = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    return best_multipliers, best_score, df


def save_confusion_matrix(y_true, y_pred, labels, name, score, output_dir):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(ax=ax, values_format="d", xticks_rotation=45)
    ax.set_title(f"{name}\nBalanced Accuracy = {score:.5f}")
    out = output_dir / f"confusion_matrix_{clean_filename(name)}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close(fig)
    return out


def average_probabilities(models, X, classes_reference=None):
    probs_list = []
    reference_classes = classes_reference

    for name, model in models.items():
        probs = model.predict_proba(X)
        classes = model.classes_
        if reference_classes is None:
            reference_classes = classes
            probs_list.append(probs)
        else:
            aligned = np.zeros((len(X), len(reference_classes)))
            for j, cls in enumerate(classes):
                target_idx = list(reference_classes).index(cls)
                aligned[:, target_idx] = probs[:, j]
            probs_list.append(aligned)

    avg = np.mean(probs_list, axis=0)
    return avg, np.array(reference_classes)


# =========================
# Main script
# =========================

OUTPUT_DIR.mkdir(exist_ok=True)
data_folder = find_data_folder()
print(f"Using data folder: {data_folder.resolve()}")
print(f"Saving outputs to: {OUTPUT_DIR.resolve()}")

train = pd.read_csv(data_folder / "train.csv")
test = pd.read_csv(data_folder / "test.csv")
sample_submission = pd.read_csv(data_folder / "sample_submission.csv")

if QUICK_TEST:
    print("QUICK_TEST is ON. Using small sample only.")
    train = stratified_sample_df(train, TARGET_COLUMN, QUICK_TEST_ROWS, RANDOM_STATE)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Target distribution:")
print(train[TARGET_COLUMN].value_counts())

raw_feature_columns = [c for c in train.columns if c not in [TARGET_COLUMN, ID_COLUMN]]
X_original = add_features(train[raw_feature_columns].copy())
y_original = train[TARGET_COLUMN].copy()
X_test = add_features(test[raw_feature_columns].copy()).reindex(columns=X_original.columns)
test_ids = test[ID_COLUMN].copy()

# Optional extra labeled data
X_extra, y_extra = None, None
extra_path = data_folder / EXTRA_LABELED_FILENAME
if USE_EXTRA_LABELED_DATA_IF_AVAILABLE and extra_path.exists():
    try:
        extra = pd.read_csv(extra_path)
        if TARGET_COLUMN in extra.columns and all(c in extra.columns for c in raw_feature_columns):
            X_extra = add_features(extra[raw_feature_columns].copy()).reindex(columns=X_original.columns)
            y_extra = extra[TARGET_COLUMN].copy()
            print(f"Using extra labeled data: {EXTRA_LABELED_FILENAME}, shape={extra.shape}")
            print(y_extra.value_counts())
        else:
            print("Extra file exists but columns do not match, skipping.")
    except Exception as exc:
        print("Could not read extra file, skipping:", exc)

preprocessor, numeric_columns, categorical_columns = make_preprocessor(X_original)
print("Feature count:", X_original.shape[1])
print("Numeric features:", len(numeric_columns), "Categorical features:", len(categorical_columns))

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
    X_train, y_train = X_train_base, y_train_base

print("Training rows:", X_train.shape[0])
print("Validation rows:", X_valid.shape[0])

# Ensemble models. All are sklearn-only, so no extra installation needed.
model_configs = {
    "hgb_balanced_main": HistGradientBoostingClassifier(
        learning_rate=0.040,
        max_iter=850,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        l2_regularization=0.00,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "hgb_balanced_smooth": HistGradientBoostingClassifier(
        learning_rate=0.032,
        max_iter=1000,
        max_leaf_nodes=45,
        min_samples_leaf=45,
        l2_regularization=0.03,
        class_weight="balanced",
        random_state=RANDOM_STATE + 11,
    ),
    "hgb_high_sensitive": HistGradientBoostingClassifier(
        learning_rate=0.045,
        max_iter=800,
        max_leaf_nodes=75,
        min_samples_leaf=25,
        l2_regularization=0.01,
        class_weight="balanced",
        random_state=RANDOM_STATE + 23,
    ),
    "hgb_less_weighted": HistGradientBoostingClassifier(
        learning_rate=0.045,
        max_iter=700,
        max_leaf_nodes=55,
        min_samples_leaf=35,
        l2_regularization=0.01,
        class_weight="balanced",
        random_state=RANDOM_STATE + 37,
    ),
}

trained_validation_models = {}
validation_rows = []

for name, clf in model_configs.items():
    print("\n" + "=" * 80)
    print("Training validation model:", name)
    print("=" * 80)
    pipe = Pipeline(steps=[("preprocess", clone(preprocessor)), ("model", clf)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_valid)
    acc = accuracy_score(y_valid, preds)
    bal = balanced_accuracy_score(y_valid, preds)
    print(f"Accuracy: {acc:.5f}")
    print(f"Balanced accuracy: {bal:.5f}")
    validation_rows.append({"model": name, "accuracy": acc, "balanced_accuracy": bal})
    trained_validation_models[name] = pipe

validation_probs, classes = average_probabilities(trained_validation_models, X_valid)
base_ensemble_preds = classes[np.argmax(validation_probs, axis=1)]
base_ensemble_bal = balanced_accuracy_score(y_valid, base_ensemble_preds)
base_ensemble_acc = accuracy_score(y_valid, base_ensemble_preds)
print("\n" + "=" * 80)
print("Validation soft-voting ensemble")
print("=" * 80)
print(f"Ensemble accuracy: {base_ensemble_acc:.5f}")
print(f"Ensemble balanced accuracy before tuning: {base_ensemble_bal:.5f}")

best_multipliers, tuned_score, tuning_df = tune_probability_multipliers(validation_probs, y_valid, classes)
tuned_preds = predict_with_multipliers(validation_probs, classes, best_multipliers)
tuned_acc = accuracy_score(y_valid, tuned_preds)
print(f"Ensemble accuracy after tuning: {tuned_acc:.5f}")
print(f"Ensemble balanced accuracy after tuning: {tuned_score:.5f}")
print("Best multipliers:", best_multipliers)
print("\nClassification report:")
print(classification_report(y_valid, tuned_preds, labels=LABELS))

pd.DataFrame(validation_rows).sort_values("balanced_accuracy", ascending=False).to_csv(OUTPUT_DIR / "single_model_validation_scores.csv", index=False)
tuning_df.to_csv(OUTPUT_DIR / "ensemble_threshold_tuning.csv", index=False)
save_confusion_matrix(y_valid, tuned_preds, LABELS, "Iter4 Ensemble Tuned", tuned_score, OUTPUT_DIR)

# Train final full-data ensemble.
print("\n" + "=" * 80)
print("Training final full-data ensemble")
print("=" * 80)
if X_extra is not None:
    X_full = pd.concat([X_original, X_extra], axis=0, ignore_index=True)
    y_full = pd.concat([y_original, y_extra], axis=0, ignore_index=True)
else:
    X_full, y_full = X_original, y_original

final_models = {}
for name, clf in model_configs.items():
    print("Training final model:", name)
    pipe = Pipeline(steps=[("preprocess", clone(preprocessor)), ("model", clone(clf))])
    pipe.fit(X_full, y_full)
    final_models[name] = pipe

print("Predicting test probabilities with ensemble...")
test_probs, test_classes = average_probabilities(final_models, X_test, classes_reference=classes)


def make_submission(filename, multipliers):
    preds = predict_with_multipliers(test_probs, test_classes, multipliers)
    sub = sample_submission.copy()
    sub[ID_COLUMN] = test_ids
    sub[TARGET_COLUMN] = preds
    out = OUTPUT_DIR / filename
    sub.to_csv(out, index=False)
    print(f"Saved {out}")
    print(sub[TARGET_COLUMN].value_counts())
    return out

# Main best validation multipliers.
main_file = make_submission("submission_iter4_ensemble.csv", best_multipliers)

# Ladder variants around your currently best Iteration 3 result, which liked more High.
for suffix, high_factor, medium_factor in [
    ("high_plus_02", 1.02, 0.995),
    ("high_plus_05", 1.05, 0.990),
    ("high_plus_08", 1.08, 0.985),
    ("high_plus_12", 1.12, 0.980),
    ("medium_plus_high_plus", 1.06, 1.015),
]:
    mult = dict(best_multipliers)
    mult["High"] = float(mult.get("High", 1.0) * high_factor)
    mult["Medium"] = float(mult.get("Medium", 1.0) * medium_factor)
    make_submission(f"submission_iter4_{suffix}.csv", mult)

with open(OUTPUT_DIR / "report_notes_iter4.txt", "w", encoding="utf-8") as f:
    f.write("ASSIGNMENT 3 - ITERATION 4 REPORT NOTES\n")
    f.write("=======================================\n\n")
    f.write("Iteration 4 used a soft-voting ensemble of multiple HistGradientBoostingClassifier models.\n")
    f.write("The models used different tree complexity, regularization, and class weights.\n")
    f.write("The averaged probabilities were tuned using validation balanced accuracy.\n")
    f.write(f"Best validation multipliers: {best_multipliers}\n")
    f.write(f"Validation ensemble balanced accuracy after tuning: {tuned_score:.5f}\n")
    f.write("\nUpload order:\n")
    f.write("1. submission_iter4_ensemble.csv\n")
    f.write("2. submission_iter4_high_plus_02.csv\n")
    f.write("3. submission_iter4_high_plus_05.csv\n")
    f.write("4. submission_iter4_high_plus_08.csv\n")
    f.write("5. submission_iter4_medium_plus_high_plus.csv\n")

print("\n" + "=" * 80)
print("DONE - ITERATION 4 ENSEMBLE")
print("=" * 80)
print("Current best known from your screenshot: submission_more_high.csv with public score 0.96823")
print("Upload Iteration 4 files only if you still have daily submissions left.")
print("Try first:", main_file)
print("All files saved in:", OUTPUT_DIR.resolve())
