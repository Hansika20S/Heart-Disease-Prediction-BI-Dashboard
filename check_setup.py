"""Run BEFORE the app:  python check_setup.py   (from the project folder)"""
import numpy as np, sklearn
import bi_utils as bu

print("scikit-learn version here:", sklearn.__version__, "(joblib was saved with 1.6.1)")
raw = bu.load_raw(); dfm = bu.load_model_frame(raw)
print("rows:", len(raw), "-> after dedupe:", len(dfm), "| target balance:", dfm.target.value_counts().to_dict())
models = bu.load_models()
print("joblib keys:", list(models))
Xtr, Xte, ytr, yte = bu.get_split(dfm)
A = models["preprocessor"].transform(Xtr)
for k in ("decision_tree", "knn"):
    assert models[k].n_features_in_ == A.shape[1], f"{k}: feature count mismatch"
# proves the reproduced training split has the same row ORDER the KNN was fitted on
dist, idx = models["knn"].kneighbors(A[:10], n_neighbors=1)
assert (idx[:, 0] == np.arange(10)).all() and (dist < 1e-4).all(), "Split does not match the saved KNN"
print("Split reproduces your Colab training set: OK")
ev = bu.evaluate_models(models, dfm)
for n, m in ev["models"].items():
    print(f"{n:14s} acc={m['Accuracy']:.3f} prec={m['Precision']:.3f} rec={m['Recall']:.3f} "
          f"f1={m['F1 Score']:.3f} auc={m['ROC-AUC']:.3f}")
print("Compare these numbers with your Colab results. If they match, everything is wired correctly.")
