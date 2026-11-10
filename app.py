"""
Heart Disease BI & Data Mining Dashboard  (DMBI CA2)
Run:  streamlit run app.py
Reuses heart_disease_models.joblib (preprocessor + Decision Tree + KNN) - nothing is retrained.
"""
import inspect
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import bi_utils as bu

st.set_page_config(page_title="Heart Disease BI Dashboard", page_icon="❤️", layout="wide")

st.markdown("""
<style>
.kpi {border-left:5px solid #1769aa; background:rgba(128,128,128,0.08); padding:12px 16px;
      border-radius:8px; margin-bottom:8px;}
.kpi .v {font-size:1.7rem; font-weight:700; line-height:1.2;}
.kpi .l {font-size:0.82rem; opacity:0.75;}
.kpi.red {border-left-color:#c0392b;} .kpi.green {border-left-color:#27ae60;}
.kpi.amber {border-left-color:#e67e22;}
</style>""", unsafe_allow_html=True)

COLORS = {"No Heart Disease": "#27ae60", "Heart Disease": "#c0392b"}


# ------------------------------------------------------------------ small helpers
def show(fig):
    """Works on old and new Streamlit versions."""
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    try:
        if "width" in inspect.signature(st.plotly_chart).parameters:
            st.plotly_chart(fig, width="stretch")
            return
    except (TypeError, ValueError):
        pass
    st.plotly_chart(fig, use_container_width=True)


def kpi(col, label, value, color=""):
    col.markdown(f'<div class="kpi {color}"><div class="v">{value}</div>'
                 f'<div class="l">{label}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ cached loading
@st.cache_resource
def get_models():
    return bu.load_models()


@st.cache_data
def get_data():
    raw = bu.load_raw()
    return raw, bu.load_model_frame(raw)


@st.cache_data
def get_eval(_models):
    _, dfm = get_data()
    return bu.evaluate_models(_models, dfm)


@st.cache_data
def get_segments(_models, k):
    _, dfm = get_data()
    return bu.segment_patients(_models, dfm, k)


@st.cache_data
def get_cv(_models):
    _, dfm = get_data()
    return bu.cross_validate_models(_models, dfm)


try:
    models = get_models()
    raw, df_model = get_data()
    ev = get_eval(models)
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Keep app.py, bi_utils.py, heart_disease.csv and "
             "heart_disease_models.joblib in the same folder.")
    st.stop()
except Exception as e:  # typically an scikit-learn version mismatch
    st.error(f"Could not load the saved models: {e}")
    st.info("The .joblib was saved with scikit-learn 1.6.1 (Colab). Run: "
            "`pip install scikit-learn==1.6.1` and restart the app.")
    st.stop()

df_all = bu.add_display_columns(df_model)
perf = ev["models"]
best_name = max(perf, key=lambda n: perf[n]["F1 Score"])

# ------------------------------------------------------------------ sidebar
st.sidebar.title("❤️ Heart Disease BI")
PAGES = ["📊 Executive Dashboard", "🔎 Data Explorer", "⛏️ Data Mining Pipeline",
         "🧪 Model Evaluation", "🩺 Predict Patient Risk", "💡 Insights & About"]
page = st.sidebar.radio("Navigate", PAGES)

st.sidebar.markdown("---")
st.sidebar.subheader("Global filters")
st.sidebar.caption("Apply to the Dashboard and Data Explorer pages.")
age_lo, age_hi = st.sidebar.slider("Age range", int(df_all.age.min()), int(df_all.age.max()),
                                   (int(df_all.age.min()), int(df_all.age.max())))
sex_sel = st.sidebar.multiselect("Sex", ["Male", "Female"], default=["Male", "Female"])
cp_sel = st.sidebar.multiselect("Chest pain type", list(bu.CP_MAP.values()),
                                default=list(bu.CP_MAP.values()))
df = df_all[df_all.age.between(age_lo, age_hi) & df_all["Sex"].isin(sex_sel)
            & df_all["Chest pain"].isin(cp_sel)]
st.sidebar.caption(f"Showing {len(df)} of {len(df_all)} patients")
st.sidebar.markdown("---")
st.sidebar.caption("Dataset: UCI Cleveland Heart Disease · Algorithms: KNN, Decision Tree")

st.title("Heart Disease — Business Intelligence & Data Mining Dashboard")
st.info("Educational decision-support project (DMBI CA2). It does not provide a medical diagnosis.")

needs_data = page in (PAGES[0], PAGES[1])
if needs_data and df.empty:
    st.warning("No patients match the current filters. Widen the filters in the sidebar.")
    st.stop()

# ====================================================================== 1. EXECUTIVE DASHBOARD
if page == PAGES[0]:
    st.header("Executive Dashboard")
    prev = df.target.mean() * 100
    c = st.columns(5)
    kpi(c[0], "Patients in view", len(df))
    kpi(c[1], "Heart-disease prevalence", f"{prev:.1f}%", "red")
    kpi(c[2], "Average age", f"{df.age.mean():.1f}")
    kpi(c[3], "Avg cholesterol (mg/dl)", f"{df.chol.mean():.0f}", "amber")
    kpi(c[4], "Avg max heart rate", f"{df.thalach.mean():.0f}", "green")

    st.subheader(f"Model KPIs (held-out test set, n={ev['n_test']})")
    b = perf[best_name]
    c = st.columns(5)
    kpi(c[0], "Best model (by F1)", best_name)
    kpi(c[1], "Accuracy", f"{b['Accuracy']:.1%}")
    kpi(c[2], "Recall (disease caught)", f"{b['Recall']:.1%}", "red")
    kpi(c[3], "Precision", f"{b['Precision']:.1%}")
    kpi(c[4], "ROC-AUC", f"{b['ROC-AUC']:.3f}", "green")

    l, r = st.columns(2)
    with l:
        cnt = df["Status"].value_counts().reset_index()
        cnt.columns = ["Status", "Patients"]
        show(px.pie(cnt, names="Status", values="Patients", hole=0.55, color="Status",
                    color_discrete_map=COLORS, title="Heart disease vs no disease"))
    with r:
        sev = df.severity.astype(int).value_counts().sort_index().reset_index()
        sev.columns = ["Severity", "Patients"]
        sev["Severity"] = sev["Severity"].map(lambda x: "0 - None" if x == 0 else f"{x} - Disease")
        show(px.bar(sev, x="Severity", y="Patients", text="Patients",
                    title="Original severity levels (0-4) before binary recoding"))

    l, r = st.columns(2)
    with l:
        g = df.groupby("Age group", observed=True)["target"].agg(["mean", "size"]).reset_index()
        g["Prevalence %"] = (g["mean"] * 100).round(1)
        show(px.bar(g, x="Age group", y="Prevalence %", text="Prevalence %", hover_data=["size"],
                    title="Disease prevalence by age group"))
    with r:
        g = df.groupby("Chest pain")["target"].mean().mul(100).round(1).reset_index()
        g.columns = ["Chest pain", "Prevalence %"]
        show(px.bar(g.sort_values("Prevalence %"), x="Prevalence %", y="Chest pain", orientation="h",
                    text="Prevalence %", title="Disease prevalence by chest pain type"))

    st.subheader("Automatic insights for the current filter")
    ag = df.groupby("Age group", observed=True)["target"].mean()
    cpg = df.groupby("Chest pain")["target"].mean()
    sxg = df.groupby("Sex")["target"].mean()
    ins = [f"Overall prevalence is **{prev:.1f}%** across {len(df)} patients.",
           f"Highest-prevalence age group: **{ag.idxmax()}** ({ag.max():.0%}).",
           f"Highest-prevalence chest-pain type: **{cpg.idxmax()}** ({cpg.max():.0%}).",
           f"Highest-prevalence sex group: **{sxg.idxmax()}** ({sxg.max():.0%})."]
    if df.target.nunique() == 2:
        d = df.groupby("target")["thalach"].mean()
        ins.append(f"Patients with disease have an average max heart rate of **{d[1]:.0f}** vs "
                   f"**{d[0]:.0f}** without disease.")
    st.markdown("\n".join(f"- {i}" for i in ins))

# ====================================================================== 2. DATA EXPLORER
elif page == PAGES[1]:
    st.header("Data Explorer")
    t1, t2, t3, t4, t5 = st.tabs(["Distributions", "Relationships", "Correlation",
                                  "Drill-down pivot", "Raw data"])
    with t1:
        c1, c2 = st.columns([1, 3])
        feat = c1.selectbox("Numeric feature", bu.NUMERIC + ["Sex"],
                            format_func=lambda x: bu.FEATURE_NAMES.get(x, x))
        view = c1.radio("Chart", ["Histogram", "Box plot"])
        if feat == "Sex":
            show(px.histogram(df, x="Sex", color="Status", barmode="group", color_discrete_map=COLORS,
                              title="Gender-wise heart disease distribution"))
        elif view == "Histogram":
            show(px.histogram(df, x=feat, color="Status", nbins=20, marginal="rug", barmode="overlay",
                              opacity=0.65, color_discrete_map=COLORS,
                              title=f"{bu.FEATURE_NAMES[feat]} distribution"))
        else:
            show(px.box(df, x="Status", y=feat, color="Status", points="all",
                        color_discrete_map=COLORS, title=f"{bu.FEATURE_NAMES[feat]} vs heart disease"))
    with t2:
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X axis", bu.NUMERIC, index=0, format_func=bu.FEATURE_NAMES.get)
        y = c2.selectbox("Y axis", bu.NUMERIC, index=3, format_func=bu.FEATURE_NAMES.get)
        sz = c3.selectbox("Bubble size", ["None", "oldpeak", "chol", "age"])
        show(px.scatter(df, x=x, y=y, color="Status", color_discrete_map=COLORS,
                        size=None if sz == "None" else sz, opacity=0.7,
                        hover_data=["Sex", "Chest pain"], title=f"{bu.FEATURE_NAMES[y]} vs {bu.FEATURE_NAMES[x]}"))
    with t3:
        corr = df[bu.FEATURES + ["target"]].corr().round(2)
        show(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                       aspect="auto", title="Correlation matrix (incl. target)"))
        tc = corr["target"].drop("target").sort_values(key=abs, ascending=False)
        st.caption("Strongest linear correlations with the target: " +
                   ", ".join(f"{bu.FEATURE_NAMES[k]} ({v:+.2f})" for k, v in tc.head(3).items()))
    with t4:
        cats = ["Sex", "Chest pain", "Age group", "Cholesterol band", "Exercise angina", "ST slope",
                "Thalassemia", "Major vessels", "Resting ECG", "Fasting sugar > 120"]
        c1, c2 = st.columns(2)
        rw = c1.selectbox("Rows", cats, index=1)
        cl = c2.selectbox("Columns", [c for c in cats if c != rw], index=1)
        pv = df.pivot_table(index=rw, columns=cl, values="target", aggfunc="mean", observed=True).mul(100).round(1)
        cn = df.pivot_table(index=rw, columns=cl, values="target", aggfunc="size", observed=True)
        for t in (pv, cn):
            t.index, t.columns = t.index.astype(str), t.columns.astype(str)
        show(px.imshow(pv, text_auto=True, color_continuous_scale="Reds", aspect="auto",
                       title=f"Disease rate (%) by {rw} × {cl}"))
        st.caption("Patient counts behind each cell (small counts = unreliable rates):")
        st.dataframe(cn.fillna(0).astype(int))
    with t5:
        st.dataframe(df[bu.FEATURES + ["severity", "target"]], height=420)
        st.download_button("Download filtered data (CSV)", df.to_csv(index=False).encode(),
                           "filtered_heart_data.csv", "text/csv")

# ====================================================================== 3. DATA MINING PIPELINE
elif page == PAGES[2]:
    st.header("Data Mining Pipeline (KDD process)")
    q = bu.data_quality(raw)
    st.subheader("1. Data selection & quality check")
    c = st.columns(4)
    kpi(c[0], "Records", q["rows"]); kpi(c[1], "Input features", 13)
    kpi(c[2], "Duplicate rows", q["duplicates"]); kpi(c[3], "Cells with missing values", sum(q["missing"].values()), "amber")
    st.write("Missing values by column: " + (", ".join(f"**{k}**: {v}" for k, v in q["missing"].items()) or "none"))

    st.subheader("2. Target transformation")
    sev = pd.Series(q["severity_counts"]).rename_axis("Original 'num' value").reset_index(name="Patients")
    sev["Becomes"] = sev["Original 'num' value"].map(lambda x: "0 - No Heart Disease" if x == 0 else "1 - Heart Disease")
    l, r = st.columns(2)
    l.dataframe(sev, hide_index=True)
    r.markdown("Original values 1-4 are merged into a single **Heart Disease** class, giving a "
               f"near-balanced binary target ({df_model.target.mean():.0%} disease). "
               "Trade-off: the *severity* information is lost for modelling; it is only used for BI charts.")

    st.subheader("3. Preprocessing & split")
    st.markdown("""
| Step | Applied to | Fitted on |
|---|---|---|
| Mode imputation + StandardScaler | age, trestbps, chol, thalach, oldpeak, ca | training set only |
| Mode imputation + One-Hot Encoding | cp, restecg, thal | training set only |
| Passthrough | sex, fbs, exang, slope | – |
""")
    c = st.columns(3)
    kpi(c[0], "Training records (80%)", ev["n_train"]); kpi(c[1], "Test records (20%)", ev["n_test"])
    kpi(c[2], "Features after encoding", len(models["preprocessor"].get_feature_names_out()))
    st.caption("Stratified split, random_state=42 — reproduced here, so the test set is identical to Colab.")

    st.subheader("4. Classification (your saved models)")
    st.write("**Decision Tree** (max_depth=5) and **KNN** (k=5) were trained in Colab and are loaded "
             "from the .joblib file. See *Model Evaluation* for results.")
    imp = bu.feature_importance(models).head(10)
    show(px.bar(imp.iloc[::-1], x="Importance", y="Feature", orientation="h",
                title="Top 10 features — Decision Tree importance"))

    st.subheader("5. NEW — Unsupervised segmentation (K-Means)")
    st.caption("An extra data-mining technique added around your models. It uses your saved preprocessor "
               "but does not change the saved models.")
    k = st.slider("Number of patient segments (k)", 2, 6, 3)
    seg, prof, sil = get_segments(models, k)
    l, r = st.columns(2)
    with l:
        show(px.scatter(seg, x="PC1", y="PC2", color="Segment", symbol="Status",
                        title="Patient segments (PCA 2-D projection)"))
    with r:
        show(px.line(x=list(sil), y=list(sil.values()), markers=True,
                     labels={"x": "k", "y": "Silhouette score"}, title="Silhouette score by k (higher = better separated)"))
    prof["Disease_rate"] = (prof["Disease_rate"] * 100).round(0).astype(int).astype(str) + "%"
    st.dataframe(prof, hide_index=True)
    st.caption("BI reading: segments with a high disease rate and low max heart rate / high oldpeak are the "
               "profiles a screening programme would prioritise.")

# ====================================================================== 4. MODEL EVALUATION
elif page == PAGES[3]:
    st.header("Model Evaluation")
    st.caption(f"Saved models evaluated on the same {ev['n_test']}-patient test set as the notebook.")
    cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Train Accuracy"]
    tbl = pd.DataFrame({n: {c: perf[n][c] for c in cols} for n in perf}).T.round(3)
    st.dataframe(tbl.style.highlight_max(axis=0, color="#cfe8cf"))
    tl = tbl.reset_index().rename(columns={"index": "Algorithm"}).melt(
        "Algorithm", ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"], "Metric", "Score")
    show(px.bar(tl, x="Metric", y="Score", color="Algorithm", barmode="group", range_y=[0, 1],
                text_auto=".2f", title="Decision Tree vs KNN"))
    gap = perf["Decision Tree"]["Train Accuracy"] - perf["Decision Tree"]["Accuracy"]
    if gap > 0.10:
        st.warning(f"Decision Tree: train accuracy exceeds test accuracy by {gap:.0%} → sign of over-fitting.")

    t1, t2, t3, t4 = st.tabs(["Confusion matrices", "ROC curves", "Threshold & business cost", "Predictions"])
    with t1:
        cc = st.columns(2)
        for col, n in zip(cc, perf):
            with col:
                show(px.imshow(perf[n]["cm"], text_auto=True, color_continuous_scale="Blues",
                               x=["No Disease", "Heart Disease"], y=["No Disease", "Heart Disease"],
                               labels=dict(x="Predicted", y="Actual"), title=f"{n} — confusion matrix"))
    with t2:
        fig = go.Figure()
        for n in perf:
            fig.add_scatter(x=perf[n]["fpr"], y=perf[n]["tpr"], mode="lines",
                            name=f"{n} (AUC {perf[n]['ROC-AUC']:.3f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random guess")
        fig.update_layout(title="ROC curve comparison", xaxis_title="False positive rate",
                          yaxis_title="True positive rate")
        show(fig)
    with t3:
        st.write("In screening, **missing a sick patient (false negative)** is usually costlier than a "
                 "false alarm. Move the threshold and set the costs to see the trade-off.")
        c1, c2, c3 = st.columns(3)
        mname = c1.selectbox("Model", list(perf))
        thr = c2.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
        fn_cost = c3.number_input("Cost of a missed case (FN) ÷ cost of a false alarm (FP)", 1, 50, 5)
        m = bu.metrics_at_threshold(ev["y_test"], perf[mname]["proba"], thr)
        c = st.columns(5)
        kpi(c[0], "Recall", f"{m['Recall']:.1%}", "red"); kpi(c[1], "Precision", f"{m['Precision']:.1%}")
        kpi(c[2], "Accuracy", f"{m['Accuracy']:.1%}"); kpi(c[3], "Missed cases (FN)", m["FN"], "amber")
        kpi(c[4], "False alarms (FP)", m["FP"])
        grid = [i / 20 for i in range(1, 20)]
        sweep = pd.DataFrame([{"Threshold": t, **bu.metrics_at_threshold(ev["y_test"], perf[mname]["proba"], t)}
                              for t in grid])
        sweep["Weighted cost"] = sweep["FN"] * fn_cost + sweep["FP"]
        show(px.line(sweep, x="Threshold", y=["Precision", "Recall", "F1 Score"], title="Metrics vs threshold"))
        best = sweep.loc[sweep["Weighted cost"].idxmin()]
        st.info(f"With FN costing {fn_cost}× an FP, the lowest-cost threshold for {mname} on this test set is "
                f"**{best['Threshold']:.2f}** (FN={int(best['FN'])}, FP={int(best['FP'])}). "
                "Note: KNN (k=5) and the tree give coarse probabilities, so the curve is step-like.")
    with t4:
        res = pd.DataFrame({"Actual": ev["y_test"].map(bu.LABEL).values,
                            "Decision Tree": pd.Series(perf["Decision Tree"]["pred"]).map(bu.LABEL).values,
                            "KNN": pd.Series(perf["KNN"]["pred"]).map(bu.LABEL).values,
                            "KNN prob. (disease)": perf["KNN"]["proba"].round(2),
                            "DT prob. (disease)": perf["Decision Tree"]["proba"].round(2)},
                           index=ev["y_test"].index)
        res["Models agree"] = res["Decision Tree"] == res["KNN"]
        res["Both correct"] = (res["Decision Tree"] == res["Actual"]) & (res["KNN"] == res["Actual"])
        c1, c2 = st.columns(2)
        kpi(c1, "Models agree on", f"{res['Models agree'].mean():.0%} of test patients")
        kpi(c2, "Both models correct on", f"{res['Both correct'].mean():.0%} of test patients", "green")
        only = st.checkbox("Show only patients where the models disagree or a model is wrong", False)
        st.dataframe(res[~res["Both correct"]] if only else res, height=380)
    st.subheader("Robustness check (optional)")
    st.caption("A single 61-patient test split is noisy. 5-fold cross-validation refits *fresh copies* of the "
               "pipeline for the check; your saved models are not modified.")
    if st.button("Run 5-fold cross-validation"):
        with st.spinner("Cross-validating..."):
            st.dataframe(get_cv(models), hide_index=True)

# ====================================================================== 5. PREDICT
elif page == PAGES[4]:
    st.header("Predict Patient Risk")
    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 1, 120, 50)
            sex = st.selectbox("Gender", [1, 0], format_func=bu.SEX_MAP.get)
            cp = st.selectbox("Chest pain type", [1, 2, 3, 4], format_func=lambda x: f"{x} - {bu.CP_MAP[x]}")
            trestbps = st.number_input("Resting blood pressure (mm Hg)", 50, 250, 120)
            chol = st.number_input("Cholesterol (mg/dl)", 50, 700, 200)
        with c2:
            fbs = st.selectbox("Fasting blood sugar > 120 mg/dl?", [0, 1], format_func=bu.YESNO.get)
            restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: f"{x} - {bu.RESTECG_MAP[x]}")
            thalach = st.number_input("Maximum heart rate", 50, 250, 150)
            exang = st.selectbox("Exercise-induced angina?", [0, 1], format_func=bu.YESNO.get)
        with c3:
            oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 10.0, 1.0, 0.1)
            slope = st.selectbox("Slope of peak exercise ST", [1, 2, 3], format_func=lambda x: f"{x} - {bu.SLOPE_MAP[x]}")
            ca = st.selectbox("Major vessels coloured (0-3)", [0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", [3, 6, 7], format_func=bu.THAL_MAP.get)
        primary = st.radio("Primary model for the gauge", list(bu.MODEL_KEYS), horizontal=True)
        submitted = st.form_submit_button("Predict Heart Disease")

    if submitted:
        pt = dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs, restecg=restecg,
                  thalach=thalach, exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)
        res = bu.predict_patient(models, pt)
        st.divider(); st.subheader("Result")
        label, prob = res[primary]
        l, r = st.columns([1, 2])
        with l:
            g = go.Figure(go.Indicator(mode="gauge+number", value=prob * 100,
                          number={"suffix": "%"}, title={"text": f"{primary}: disease-class probability"},
                          gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#c0392b"},
                                 "steps": [{"range": [0, 33], "color": "#d5f0dd"},
                                           {"range": [33, 66], "color": "#fdebc8"},
                                           {"range": [66, 100], "color": "#f6d0cc"}]}))
            g.update_layout(height=280); show(g)
        with r:
            (st.error if label else st.success)(
                f"{primary}: {'Heart disease indicated' if label else 'No heart disease indicated'} "
                f"(model probability band: **{bu.risk_band(prob)}**)")
            cc = st.columns(2)
            for col, (n, (lb, p)) in zip(cc, res.items()):
                kpi(col, f"{n} → {'Disease' if lb else 'No disease'}", f"{p:.0%}", "red" if lb else "green")
            if res["Decision Tree"][0] == res["KNN"][0]:
                st.write("✅ Both models **agree** on this patient.")
            else:
                st.write("⚠️ The models **disagree** — treat this case as uncertain.")
            st.caption("KNN (k=5) probabilities can only be 0/20/40/60/80/100%, and a depth-5 tree is also coarse. "
                       "Probabilities are model scores, not confirmed medical risk.")
        st.subheader("Patient vs. historical averages")
        avg = df_model.groupby("target")[["age", "trestbps", "chol", "thalach", "oldpeak"]].mean().round(1)
        cmp = pd.DataFrame({"Patient": [age, trestbps, chol, thalach, oldpeak],
                            "Avg. without disease": avg.loc[0].values, "Avg. with disease": avg.loc[1].values},
                           index=[bu.FEATURE_NAMES[c] for c in avg.columns])
        st.dataframe(cmp)
        st.subheader("5 most similar historical patients (from the saved KNN)")
        st.dataframe(bu.similar_patients(models, df_model, pt).rename(columns=bu.FEATURE_NAMES))
        st.warning("For academic demonstration only. Please consult a qualified healthcare professional.")

# ====================================================================== 6. INSIGHTS
else:
    st.header("Insights & About")
    full = df_all
    cpr = full.groupby("Chest pain")["target"].mean()
    agr = full.groupby("Age group", observed=True)["target"].mean()
    sxr = full.groupby("Sex")["target"].mean()
    imp = bu.feature_importance(models)
    st.subheader("Key findings (computed from the data)")
    st.markdown(f"""
- **{full.target.mean():.0%}** of the {len(full)} patients have heart disease — a nearly balanced target.
- Prevalence rises with age: **{agr.iloc[0]:.0%}** in the youngest group vs **{agr.iloc[-1]:.0%}** in the oldest.
- **{cpr.idxmax()}** chest pain has the highest disease rate (**{cpr.max():.0%}**); **{cpr.idxmin()}** the lowest (**{cpr.min():.0%}**).
- **{sxr.idxmax()}** patients show a higher disease rate (**{sxr.max():.0%}**) than **{sxr.idxmin()}** (**{sxr.min():.0%}**).
- Top predictors in the Decision Tree: **{', '.join(imp.Feature.head(3))}**.
- Best model by F1 on the test set: **{best_name}** (Recall {perf[best_name]['Recall']:.0%}, ROC-AUC {perf[best_name]['ROC-AUC']:.2f}).
""")
    st.subheader("Business recommendations (from the dataset, not medical advice)")
    st.markdown("""
1. **Prioritise screening** of the high-prevalence profiles (age groups / chest-pain types above).
2. **Tune for recall**: a missed case costs more than a false alarm — use the threshold tool on *Model Evaluation*.
3. **Use the models as decision support**, flagging patients where the two models disagree for manual review.
""")
    st.subheader("Limitations")
    st.markdown("""
- Only 303 patients from a single source (Cleveland); the test set is just 61 patients, so metrics are noisy
  (see the cross-validation check).
- Severity (0-4) is collapsed into a binary target.
- Fixed hyper-parameters (k=5, max_depth=5); no tuning was performed.
- The tool is educational and is **not** a diagnostic device.
""")
    st.subheader("How this project maps to DMBI")
    st.markdown("""
| Topic | Where in this app |
|---|---|
| **Data Mining** – preprocessing, classification (KNN, Decision Tree), evaluation, clustering | Pipeline, Model Evaluation pages |
| **Business Intelligence** – KPIs, OLAP-style slice/dice & drill-down, dashboards, decision support | Dashboard, Data Explorer, Threshold tool, Predict page |
""")

st.divider()
st.caption("Heart Disease BI Dashboard | DMBI CA2 Mini Project | UCI Cleveland Dataset")
