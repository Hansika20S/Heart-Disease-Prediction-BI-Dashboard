"""
bi_utils.py - helper functions for the Heart Disease BI dashboard.

Everything here REUSES the saved preprocessor/models from heart_disease_models.joblib.
The cleaning + 80:20 stratified split (random_state=42) replicate your Colab notebook,
so the test set used here is exactly the one your models were evaluated on.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "heart_disease_models.joblib"
DATA_PATH = BASE / "heart_disease.csv"

FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
NUMERIC = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
CATEGORICAL = ['cp', 'restecg', 'thal']
MODEL_KEYS = {"Decision Tree": "decision_tree", "KNN": "knn"}

SEX_MAP = {1: "Male", 0: "Female"}
CP_MAP = {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}
RESTECG_MAP = {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}
THAL_MAP = {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}
SLOPE_MAP = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
YESNO = {0: "No", 1: "Yes"}
LABEL = {0: "No Heart Disease", 1: "Heart Disease"}

FEATURE_NAMES = {
    'age': 'Age', 'sex': 'Sex', 'cp': 'Chest pain type', 'trestbps': 'Resting BP (mm Hg)',
    'chol': 'Cholesterol (mg/dl)', 'fbs': 'Fasting sugar > 120', 'restecg': 'Resting ECG',
    'thalach': 'Max heart rate', 'exang': 'Exercise angina', 'oldpeak': 'ST depression (oldpeak)',
    'slope': 'ST slope', 'ca': 'Major vessels (0-3)', 'thal': 'Thalassemia',
}


# ----------------------------------------------------------------------------- data
def load_raw(path=DATA_PATH):
    df = pd.read_csv(path)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_model_frame(raw):
    """Same cleaning as the notebook: drop duplicates, binary target (0 vs 1-4)."""
    df = raw.drop_duplicates().copy()
    df["severity"] = df["num"]                       # kept ONLY for BI charts
    df["target"] = (df["num"] != 0).astype(int)      # same target the models were trained on
    return df.drop(columns="num")


def data_quality(raw):
    return {"rows": len(raw), "cols": raw.shape[1], "duplicates": int(raw.duplicated().sum()),
            "missing": raw.isna().sum()[lambda s: s > 0].to_dict(),
            "severity_counts": raw["num"].value_counts().sort_index().to_dict()}


def add_display_columns(df):
    """Human-readable columns for charts (missing ca/thal filled with the mode, as in training)."""
    d = df.copy()
    for c in FEATURES:
        if d[c].isna().any():
            d[c] = d[c].fillna(d[c].mode()[0])
    d["Sex"] = d["sex"].map(SEX_MAP)
    d["Chest pain"] = d["cp"].map(CP_MAP)
    d["Resting ECG"] = d["restecg"].map(RESTECG_MAP)
    d["Thalassemia"] = d["thal"].map(THAL_MAP)
    d["ST slope"] = d["slope"].map(SLOPE_MAP)
    d["Exercise angina"] = d["exang"].map(YESNO)
    d["Fasting sugar > 120"] = d["fbs"].map(YESNO)
    d["Major vessels"] = d["ca"].astype(int).astype(str)
    d["Age group"] = pd.cut(d["age"], [0, 40, 50, 60, 70, 200], right=False,
                            labels=["Under 40", "40-49", "50-59", "60-69", "70+"])
    d["Cholesterol band"] = pd.cut(d["chol"], [0, 200, 240, 1000], right=False,
                                   labels=["Desirable (<200)", "Borderline (200-239)", "High (240+)"])
    d["Status"] = d["target"].map(LABEL)
    return d


def get_split(df_model, seed=42):
    X, y = df_model[FEATURES], df_model["target"]
    return train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)


# --------------------------------------------------------------------------- models
def load_models(path=MODEL_PATH):
    return joblib.load(path)


def _metrics(y, pred, proba):
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0), "F1 Score": f1_score(y, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y, proba)}


def evaluate_models(models, df_model):
    """Evaluate the SAVED models on the SAME test split as the notebook. No retraining."""
    pre = models["preprocessor"]
    X_tr, X_te, y_tr, y_te = get_split(df_model)
    Xte, Xtr = pre.transform(X_te), pre.transform(X_tr)      # transform only - never refit
    out = {"y_test": y_te, "X_test": X_te, "n_train": len(X_tr), "n_test": len(X_te), "models": {}}
    for name, key in MODEL_KEYS.items():
        m = models[key]
        pred, proba = m.predict(Xte), m.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        out["models"][name] = {**_metrics(y_te, pred, proba), "pred": pred, "proba": proba,
                               "cm": confusion_matrix(y_te, pred), "fpr": fpr, "tpr": tpr,
                               "Train Accuracy": accuracy_score(y_tr, m.predict(Xtr))}
    return out


def metrics_at_threshold(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0), "F1 Score": f1_score(y, pred, zero_division=0),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def pretty_feature(raw_name):
    name = raw_name.split("__", 1)[-1]
    for prefix, mp, title in (("cp", CP_MAP, "Chest pain"), ("restecg", RESTECG_MAP, "ECG"),
                              ("thal", THAL_MAP, "Thal")):
        if name.startswith(prefix + "_"):
            try:
                return f"{title}: {mp[int(float(name.split('_', 1)[1]))]}"
            except (ValueError, KeyError):
                pass
    return FEATURE_NAMES.get(name, name)


def feature_importance(models):
    names = [pretty_feature(n) for n in models["preprocessor"].get_feature_names_out()]
    imp = models["decision_tree"].feature_importances_
    return (pd.DataFrame({"Feature": names, "Importance": imp})
            .sort_values("Importance", ascending=False).reset_index(drop=True))


def cross_validate_models(models, df_model, folds=5):
    """OPTIONAL robustness check. Fits fresh clones inside CV; saved models are untouched."""
    X, y = df_model[FEATURES], df_model["target"]
    cv = StratifiedKFold(folds, shuffle=True, random_state=42)
    rows = []
    for name, key in MODEL_KEYS.items():
        pipe = Pipeline([("pre", clone(models["preprocessor"])), ("clf", clone(models[key]))])
        res = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
        row = {"Algorithm": name}
        for k, label in [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                         ("f1", "F1 Score"), ("roc_auc", "ROC-AUC")]:
            v = res[f"test_{k}"]
            row[label] = f"{v.mean():.3f} ± {v.std():.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- prediction
def predict_patient(models, patient):
    """patient: dict of the 13 features. Returns {model_name: (label, prob_disease)}."""
    Xp = models["preprocessor"].transform(pd.DataFrame([patient], columns=FEATURES))
    return {n: (int(models[k].predict(Xp)[0]), float(models[k].predict_proba(Xp)[0][1]))
            for n, k in MODEL_KEYS.items()}


def similar_patients(models, df_model, patient, k=5):
    """The k nearest TRAINING patients according to the saved KNN, with their real outcomes."""
    X_tr, _, y_tr, _ = get_split(df_model)
    Xp = models["preprocessor"].transform(pd.DataFrame([patient], columns=FEATURES))
    dist, idx = models["knn"].kneighbors(Xp, n_neighbors=k)
    out = X_tr.iloc[idx[0]].copy()
    out["Actual outcome"] = y_tr.iloc[idx[0]].map(LABEL).values
    out["Distance"] = dist[0].round(3)
    return out


def risk_band(p):
    return "Low" if p < 0.33 else ("Moderate" if p < 0.66 else "High")


# ----------------------------------------------------------------------- clustering
def segment_patients(models, df_model, k=3):
    """NEW data-mining step (unsupervised): K-Means on the already-preprocessed features."""
    Xp = models["preprocessor"].transform(df_model[FEATURES])
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xp)
    pcs = PCA(2, random_state=42).fit_transform(Xp)
    d = df_model.copy()
    d["Segment"] = ["Segment " + str(c + 1) for c in km.labels_]
    d["PC1"], d["PC2"] = pcs[:, 0], pcs[:, 1]
    prof = (d.groupby("Segment").agg(Patients=("target", "size"), Avg_age=("age", "mean"),
            Avg_chol=("chol", "mean"), Avg_max_HR=("thalach", "mean"), Avg_oldpeak=("oldpeak", "mean"),
            Disease_rate=("target", "mean")).round(2).reset_index())
    sil = {kk: silhouette_score(Xp, KMeans(kk, n_init=10, random_state=42).fit_predict(Xp))
           for kk in range(2, 7)}
    return d, prof, sil
