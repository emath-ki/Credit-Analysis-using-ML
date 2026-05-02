# =============================================================================
# German Credit Risk Dashboard
# Run with: streamlit run app.py
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ──────────────────────────────────────────────────────────────
NAVY    = "#0B1F3A"
NAVY_LT = "#132945"
GOLD    = "#C9A84C"
GOLD_LT = "#E8C97A"
OFFWHT  = "#F5F1EB"
GREEN   = "#2E8B6E"
RED     = "#C0392B"
SLATE   = "#6B7A8D"
WHITE   = "#FFFFFF"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {{
      font-family: 'DM Sans', sans-serif;
      background-color: {NAVY};
      color: {OFFWHT};
  }}
  .stApp {{
      background: linear-gradient(160deg, {NAVY} 0%, #0D2440 60%, #0F1E35 100%);
  }}
  /* Sidebar */
  [data-testid="stSidebar"] {{
      background: {NAVY_LT} !important;
      border-right: 1px solid rgba(201,168,76,0.15);
  }}
  [data-testid="stSidebar"] * {{
      color: {OFFWHT} !important;
  }}
  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
      background: transparent;
      border-bottom: 1px solid rgba(201,168,76,0.2);
      gap: 4px;
  }}
  .stTabs [data-baseweb="tab"] {{
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(201,168,76,0.15);
      border-radius: 6px 6px 0 0;
      color: {SLATE} !important;
      font-size: 13px;
      font-weight: 500;
      padding: 8px 20px;
  }}
  .stTabs [aria-selected="true"] {{
      background: rgba(201,168,76,0.12) !important;
      border-color: {GOLD} !important;
      color: {GOLD} !important;
  }}
  /* Metric cards */
  .metric-card {{
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(201,168,76,0.15);
      border-radius: 10px;
      padding: 20px 24px;
      margin-bottom: 12px;
  }}
  .metric-label {{
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: {SLATE};
      margin-bottom: 4px;
  }}
  .metric-value {{
      font-size: 32px;
      font-weight: 700;
      color: {GOLD};
      line-height: 1;
  }}
  .metric-sub {{
      font-size: 12px;
      color: {SLATE};
      margin-top: 4px;
  }}
  /* Verdict boxes */
  .verdict-approve {{
      background: rgba(46,139,110,0.15);
      border: 1px solid {GREEN};
      border-left: 4px solid {GREEN};
      border-radius: 8px;
      padding: 16px 20px;
  }}
  .verdict-review {{
      background: rgba(201,168,76,0.1);
      border: 1px solid {GOLD};
      border-left: 4px solid {GOLD};
      border-radius: 8px;
      padding: 16px 20px;
  }}
  .verdict-decline {{
      background: rgba(192,57,43,0.15);
      border: 1px solid {RED};
      border-left: 4px solid {RED};
      border-radius: 8px;
      padding: 16px 20px;
  }}
  .verdict-title {{
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 6px;
  }}
  .info-box {{
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 8px;
      padding: 14px 18px;
      font-size: 13px;
      color: {OFFWHT};
      line-height: 1.6;
  }}
  /* Inputs */
  .stSlider > div > div > div > div {{
      background: {GOLD} !important;
  }}
  .stSelectbox [data-baseweb="select"] {{
      background: rgba(255,255,255,0.06);
      border-color: rgba(201,168,76,0.3);
  }}
  div[data-testid="stNumberInput"] input,
  div[data-testid="stTextInput"] input {{
      background: rgba(255,255,255,0.06);
      border-color: rgba(201,168,76,0.3);
      color: {OFFWHT};
  }}
  /* Page header */
  .page-header {{
      padding: 8px 0 20px 0;
      border-bottom: 1px solid rgba(201,168,76,0.2);
      margin-bottom: 24px;
  }}
  .page-title {{
      font-size: 26px;
      font-weight: 700;
      color: {OFFWHT};
      letter-spacing: -0.02em;
  }}
  .page-sub {{
      font-size: 13px;
      color: {SLATE};
      margin-top: 2px;
  }}
  /* Dashboard main title */
  .dashboard-title {{
      font-size: 32px;
      font-weight: 700;
      color: {OFFWHT};
      letter-spacing: -0.02em;
      margin-bottom: 4px;
  }}
  .dashboard-tagline {{
      font-size: 15px;
      color: {GOLD};
      font-weight: 500;
      margin-bottom: 8px;
  }}
  .dashboard-one-liner {{
      font-size: 14px;
      color: {SLATE};
      margin-bottom: 28px;
      line-height: 1.5;
      max-width: 85%;
  }}
  /* Footer */
  .footer {{
      text-align: center;
      padding: 32px 0 12px 0;
      font-size: 12px;
      color: {SLATE};
      border-top: 1px solid rgba(255,255,255,0.06);
      margin-top: 40px;
  }}
  h1, h2, h3 {{
      color: {OFFWHT} !important;
  }}
  .stMarkdown p {{
      color: {OFFWHT};
  }}
  /* Upload area */
  [data-testid="stFileUploadDropzone"] {{
      background: rgba(255,255,255,0.04) !important;
      border: 1px dashed rgba(201,168,76,0.3) !important;
      border-radius: 8px;
  }}
  /* Dataframe */
  [data-testid="stDataFrame"] {{
      border: 1px solid rgba(201,168,76,0.15);
      border-radius: 8px;
  }}
  .section-title {{
      font-size: 14px;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: {GOLD};
      margin: 20px 0 10px 0;
  }}
</style>
""", unsafe_allow_html=True)


# ── Feature metadata ───────────────────────────────────────────────────────────
# Full German Credit dataset feature definitions with human-readable labels
FEATURE_META = {
    "Attribute1": {
        "label": "Checking account",
        "type": "cat",
        "options": {
            "A11": "Negative balance (< 0 DM)",
            "A12": "Low balance (0–200 DM)",
            "A13": "Healthy balance (≥ 200 DM)",
            "A14": "No checking account",
        },
    },
    "Attribute2":  {"label": "Loan duration (months)", "type": "num", "min": 4,  "max": 72,  "default": 18},
    "Attribute3": {
        "label": "Credit history",
        "type": "cat",
        "options": {
            "A30": "No credits taken / all paid",
            "A31": "All credits paid at this bank",
            "A32": "Existing credits paid duly",
            "A33": "Delay in paying off in the past",
            "A34": "Critical / other credits exist",
        },
    },
    "Attribute4": {
        "label": "Loan purpose",
        "type": "cat",
        "options": {
            "A40": "Car (new)", "A41": "Car (used)", "A42": "Furniture/equipment",
            "A43": "Radio/television", "A44": "Domestic appliances", "A45": "Repairs",
            "A46": "Education", "A47": "Vacation (N/A)", "A48": "Retraining",
            "A49": "Business", "A410": "Other",
        },
    },
    "Attribute5":  {"label": "Credit amount (DM)", "type": "num", "min": 250,  "max": 18500, "default": 3000},
    "Attribute6": {
        "label": "Savings / bonds",
        "type": "cat",
        "options": {
            "A61": "< 100 DM", "A62": "100–500 DM", "A63": "500–1000 DM",
            "A64": "≥ 1000 DM", "A65": "Unknown / no savings",
        },
    },
    "Attribute7": {
        "label": "Employment since",
        "type": "cat",
        "options": {
            "A71": "Unemployed", "A72": "< 1 year", "A73": "1–4 years",
            "A74": "4–7 years", "A75": "≥ 7 years",
        },
    },
    "Attribute8":  {"label": "Instalment rate (% of income)", "type": "num", "min": 1, "max": 4, "default": 2},
    "Attribute9": {
        "label": "Personal status & sex",
        "type": "cat",
        "options": {
            "A91": "Male – divorced/separated", "A92": "Female – divorced/separated/married",
            "A93": "Male – single", "A94": "Male – married/widowed",
            "A95": "Female – single",
        },
    },
    "Attribute10": {
        "label": "Co-applicant / guarantor",
        "type": "cat",
        "options": {"A101": "None", "A102": "Co-applicant", "A103": "Guarantor"},
    },
    "Attribute11": {"label": "Present residence (years)", "type": "num", "min": 1, "max": 4, "default": 2},
    "Attribute12": {
        "label": "Property / collateral",
        "type": "cat",
        "options": {
            "A121": "Real estate", "A122": "Life insurance / savings",
            "A123": "Car / other", "A124": "None / unknown",
        },
    },
    "Attribute13": {"label": "Age (years)", "type": "num", "min": 19, "max": 75, "default": 35},
    "Attribute14": {
        "label": "Other instalment plans",
        "type": "cat",
        "options": {"A141": "Bank", "A142": "Stores", "A143": "None"},
    },
    "Attribute15": {
        "label": "Housing",
        "type": "cat",
        "options": {"A151": "Rent", "A152": "Own", "A153": "Free"},
    },
    "Attribute16": {"label": "Existing credits at this bank", "type": "num", "min": 1, "max": 4, "default": 1},
    "Attribute17": {
        "label": "Job type",
        "type": "cat",
        "options": {
            "A171": "Unemployed / unskilled (non-resident)",
            "A172": "Unskilled (resident)", "A173": "Skilled / official",
            "A174": "Management / self-employed / officer",
        },
    },
    "Attribute18": {"label": "Number of dependants", "type": "num", "min": 1, "max": 2, "default": 1},
    "Attribute19": {
        "label": "Telephone registered",
        "type": "cat",
        "options": {"A191": "None", "A192": "Yes, registered"},
    },
    "Attribute20": {
        "label": "Foreign worker",
        "type": "cat",
        "options": {"A201": "Yes", "A202": "No"},
    },
}

CATEGORICAL_COLS = [k for k, v in FEATURE_META.items() if v["type"] == "cat"]
NUMERIC_COLS     = [k for k, v in FEATURE_META.items() if v["type"] == "num"]

# ── Data generation (mirrors UCI Statlog German Credit statistics) ─────────────
@st.cache_data(show_spinner=False)
def generate_synthetic_data(n=1000, seed=42):
    """
    Reproduce the UCI German Credit dataset structure using the known
    class-conditional distributions from the original 1000-sample dataset.
    This lets the dashboard run standalone without the ucimlrepo package.
    """
    rng = np.random.default_rng(seed)

    records = []
    for i in range(n):
        is_bad = rng.random() < 0.30  # 70/30 Good/Bad split
        label  = 1 if is_bad else 0

        # Numeric features — calibrated to known dataset stats
        duration  = int(np.clip(rng.normal(24 if is_bad else 18, 10), 4, 72))
        amount    = int(np.clip(rng.normal(3800 if is_bad else 2800, 2200), 250, 18500))
        install   = int(rng.choice([1,2,3,4], p=[0.2,0.3,0.3,0.2]))
        residence = int(rng.choice([1,2,3,4], p=[0.1,0.3,0.4,0.2]))
        age       = int(np.clip(rng.normal(32 if is_bad else 36, 11), 19, 75))
        n_credits = int(rng.choice([1,2,3,4], p=[0.6,0.25,0.1,0.05]))
        dependants= int(rng.choice([1,2], p=[0.8,0.2]))

        # Categorical features — probabilities reflect known default-rate differences
        chk = rng.choice(["A11","A12","A13","A14"],
                         p=[0.35,0.27,0.06,0.32] if is_bad else [0.17,0.27,0.07,0.49])
        hist= rng.choice(["A30","A31","A32","A33","A34"],
                         p=[0.06,0.06,0.55,0.15,0.18] if is_bad else [0.04,0.05,0.53,0.08,0.30])
        purp= rng.choice(["A40","A41","A42","A43","A44","A45","A46","A47","A48","A49","A410"],
                         p=[0.10,0.08,0.18,0.28,0.01,0.02,0.05,0.00,0.02,0.17,0.09])
        sav = rng.choice(["A61","A62","A63","A64","A65"],
                         p=[0.60,0.14,0.08,0.07,0.11] if is_bad else [0.48,0.16,0.10,0.10,0.16])
        emp = rng.choice(["A71","A72","A73","A74","A75"],
                         p=[0.08,0.24,0.25,0.16,0.27])
        sex = rng.choice(["A91","A92","A93","A94","A95"],
                         p=[0.05,0.31,0.55,0.09,0.00])
        guar= rng.choice(["A101","A102","A103"], p=[0.91,0.04,0.05])
        prop= rng.choice(["A121","A122","A123","A124"],
                         p=[0.28,0.23,0.23,0.26] if is_bad else [0.38,0.25,0.22,0.15])
        oip = rng.choice(["A141","A142","A143"], p=[0.14,0.02,0.84])
        hous= rng.choice(["A151","A152","A153"], p=[0.18,0.71,0.11])
        job = rng.choice(["A171","A172","A173","A174"],
                         p=[0.02,0.20,0.63,0.15])
        tel = rng.choice(["A191","A192"], p=[0.60,0.40])
        fgn = rng.choice(["A201","A202"], p=[0.963,0.037])

        records.append({
            "Attribute1": chk,  "Attribute2": duration, "Attribute3": hist,
            "Attribute4": purp, "Attribute5": amount,   "Attribute6": sav,
            "Attribute7": emp,  "Attribute8": install,  "Attribute9": sex,
            "Attribute10": guar,"Attribute11": residence,"Attribute12": prop,
            "Attribute13": age, "Attribute14": oip,     "Attribute15": hous,
            "Attribute16": n_credits,"Attribute17": job,"Attribute18": dependants,
            "Attribute19": tel, "Attribute20": fgn,
            "default": label,
        })

    return pd.DataFrame(records)


# ── Model training pipeline ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_models():
    """
    Retrain all three models on the synthetic dataset.
    Returns: models dict, scaler, feature columns, evaluation results, test data.
    """
    df = generate_synthetic_data()
    X_raw = df.drop(columns=["default"])
    y     = df["default"]

    # One-hot encode with drop='first' (same as notebook)
    X_enc = pd.get_dummies(X_raw, columns=CATEGORICAL_COLS, drop_first=True).astype(float)
    feature_cols = X_enc.columns.tolist()

    # Stratified 60/20/20 split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_enc, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )

    # Scale (fit on train only)
    scaler = StandardScaler()
    scaler.fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic Regression
    lr_gs = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        {"C": [0.01, 0.1, 1, 10], "class_weight": [None, "balanced", {0:1,1:5}]},
        cv=cv, scoring="f1_weighted", n_jobs=-1
    )
    lr_gs.fit(Xtr, y_train)
    lr = lr_gs.best_estimator_

    # k-NN
    knn_gs = GridSearchCV(
        KNeighborsClassifier(),
        {"n_neighbors": [5, 7, 11, 15], "weights": ["uniform", "distance"]},
        cv=cv, scoring="f1_weighted", n_jobs=-1
    )
    knn_gs.fit(Xtr, y_train)
    knn = knn_gs.best_estimator_

    # Decision Tree
    dt_gs = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        {
            "criterion": ["gini", "entropy"],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5, 10],
            "class_weight": [None, "balanced", {0:1,1:5}],
        },
        cv=cv, scoring="f1_weighted", n_jobs=-1
    )
    dt_gs.fit(Xtr, y_train)
    dt = dt_gs.best_estimator_

    models = {
        "Logistic Regression": lr,
        "k-Nearest Neighbours": knn,
        "Decision Tree": dt,
    }

    # Evaluate all on test set
    results = {}
    for name, mdl in models.items():
        yp   = mdl.predict(Xte)
        yprb = mdl.predict_proba(Xte)[:, 1]
        fpr, tpr, thr = roc_curve(y_test, yprb, pos_label=1)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(y_test, yp, labels=[0,1])
        fn = int(cm[1,0]); fp = int(cm[0,1])
        cost = fn * 5 + fp * 1
        results[name] = {
            "accuracy":  accuracy_score(y_test, yp),
            "precision": precision_score(y_test, yp, pos_label=1, zero_division=0),
            "recall":    recall_score(y_test, yp, pos_label=1, zero_division=0),
            "f1":        f1_score(y_test, yp, pos_label=1, zero_division=0),
            "auc":       roc_auc,
            "cost":      cost,
            "fn":        fn,
            "fp":        fp,
            "cm":        cm,
            "fpr":       fpr,
            "tpr":       tpr,
            "y_prob":    yprb,
        }

    # Feature importances (LR coefficients, DT feature_importances_)
    lr_imp  = pd.Series(np.abs(lr.coef_[0]),  index=feature_cols)
    dt_imp  = pd.Series(dt.feature_importances_, index=feature_cols)

    return models, scaler, feature_cols, results, y_test, lr_imp, dt_imp, X_raw


def encode_input(raw_dict, feature_cols):
    """One-hot encode a single applicant dict to match the training feature space."""
    row = pd.DataFrame([raw_dict])
    row_enc = pd.get_dummies(row, columns=CATEGORICAL_COLS, drop_first=True).astype(float)
    # Align columns — missing OHE dummies become 0
    row_enc = row_enc.reindex(columns=feature_cols, fill_value=0.0)
    return row_enc


def pd_label(pd_score):
    if pd_score < 0.25:
        return "Low risk", GREEN, "✓ Approve"
    elif pd_score < 0.50:
        return "Medium risk", GOLD, "⚠ Review manually"
    else:
        return "High risk", RED, "✗ Decline"


# ── Build models on load ────────────────────────────────────────────────────────
with st.spinner("Fitting models on training data…"):
    models, scaler, feature_cols, results, y_test, lr_imp, dt_imp, X_raw_ref = build_models()

best_model_name = min(results, key=lambda k: results[k]["cost"])

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:16px 0 24px 0; border-bottom:1px solid rgba(201,168,76,0.2); margin-bottom:20px;'>
      <div style='font-size:11px; letter-spacing:0.1em; text-transform:uppercase; color:{SLATE}; margin-bottom:4px;'>Credit Risk</div>
      <div style='font-size:20px; font-weight:700; color:{OFFWHT};'>Intelligence</div>
      <div style='font-size:11px; color:{SLATE}; margin-top:4px;'>Statlog German Credit · UCI</div>
    </div>
    """, unsafe_allow_html=True)

    # ── NEW: Recruiter‑ready decision support paragraph ────────────────────────
    st.markdown(f"""
    <div class='info-box' style='margin-bottom:20px; background: rgba(201,168,76,0.05);'>
      <b>Decision support for credit officers</b><br>
      Instead of a 20‑minute manual review or a gut‑feel guess, enter 10–15 borrower attributes and receive an instant, explainable risk score. 
      The dashboard returns a recommended action – <b>Approve / Review / Decline</b> – powered by the best model from our offline evaluation. 
      No coding required. Just faster, more consistent underwriting.
    </div>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Active model",
        list(models.keys()),
        index=list(models.keys()).index(best_model_name),
        help="Switch between the three tuned classifiers."
    )
    st.markdown(f"""
    <div style='font-size:11px; color:{SLATE}; margin-top:-8px; margin-bottom:16px;'>
      ★ Recommended: <b style='color:{GOLD};'>{best_model_name}</b> (lowest cost-weighted error)
    </div>
    """, unsafe_allow_html=True)

    threshold = st.slider(
        "Decision threshold (PD)",
        min_value=0.10, max_value=0.70, value=0.35, step=0.01,
        help="Probability of default above which the model flags an application. "
             "The asymmetric 5:1 cost matrix makes 0.35 a more conservative, "
             "cost-optimal starting point than the default 0.5.",
    )
    st.markdown(f"""
    <div class='info-box' style='margin-top:8px;'>
      <b>Why not 0.5?</b> A missed default costs 5× a wrongly declined good applicant.
      The cost-optimal threshold typically falls in the 0.30–0.45 range.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    r = results[model_choice]
    st.markdown(f"<div class='section-title'>Live metrics — {model_choice}</div>", unsafe_allow_html=True)
    for label, val, suffix in [
        ("Accuracy",       f"{r['accuracy']:.1%}",  ""),
        ("Recall (Bad)",   f"{r['recall']:.1%}",    "← primary"),
        ("AUC-ROC",        f"{r['auc']:.3f}",       ""),
        ("Cost (5×FN+FP)", str(r['cost']),           ""),
    ]:
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between; padding:6px 0;
                    border-bottom:1px solid rgba(255,255,255,0.05); font-size:13px;'>
          <span style='color:{SLATE};'>{label}</span>
          <span style='color:{GOLD}; font-weight:600;'>{val} <span style='color:{SLATE};font-size:10px;'>{suffix}</span></span>
        </div>
        """, unsafe_allow_html=True)

# ── MAIN DASHBOARD TITLE (above tabs) ──────────────────────────────────────────
st.markdown(f"""
<div style='margin-bottom: 16px;'>
    <div class='dashboard-title'>Credit Risk Screener</div>
    <div class='dashboard-tagline'>Real‑time underwriting support by EMath</div>
    <div class='dashboard-one-liner'>Using a tuned <b>{best_model_name}</b>, enter borrower attributes and receive an instant, explainable risk score with a recommended action — Approve, Review, or Decline.</div>
</div>
""", unsafe_allow_html=True)

# ── Main tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "  📋  Credit Assessment  ",
    "  📈  Model Performance  ",
    "  📂  Batch Analysis  ",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — APPLICANT ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"""
    <div class='page-header'>
      <div class='page-title'>Applicant risk profile</div>
      <div class='page-sub'>Good to see you. Fill in an applicant's details and the model will return an instant PD estimate.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input form ─────────────────────────────────────────────────────────────
    with st.expander("▸  Applicant details", expanded=True):
        c1, c2, c3 = st.columns(3)
        raw_input = {}

        # Column 1 — financial profile
        with c1:
            st.markdown(f"<div class='section-title'>Financial profile</div>", unsafe_allow_html=True)
            meta = FEATURE_META["Attribute1"]
            raw_input["Attribute1"] = st.selectbox(
                meta["label"], list(meta["options"].keys()),
                format_func=lambda k: meta["options"][k], index=3
            )
            raw_input["Attribute2"] = st.slider(
                FEATURE_META["Attribute2"]["label"], 4, 72, 18, step=1
            )
            raw_input["Attribute5"] = st.number_input(
                FEATURE_META["Attribute5"]["label"],
                min_value=250, max_value=18500, value=3000, step=100
            )
            meta = FEATURE_META["Attribute6"]
            raw_input["Attribute6"] = st.selectbox(
                meta["label"], list(meta["options"].keys()),
                format_func=lambda k: meta["options"][k], index=0
            )
            raw_input["Attribute8"] = st.slider(
                FEATURE_META["Attribute8"]["label"], 1, 4, 2,
                help="Loan instalment as % of disposable income."
            )

        # Column 2 — credit history & purpose
        with c2:
            st.markdown(f"<div class='section-title'>Credit history & purpose</div>", unsafe_allow_html=True)
            meta = FEATURE_META["Attribute3"]
            raw_input["Attribute3"] = st.selectbox(
                meta["label"], list(meta["options"].keys()),
                format_func=lambda k: meta["options"][k], index=2
            )
            meta = FEATURE_META["Attribute4"]
            raw_input["Attribute4"] = st.selectbox(
                meta["label"], list(meta["options"].keys()),
                format_func=lambda k: meta["options"][k], index=3
            )
            raw_input["Attribute16"] = st.slider(
                FEATURE_META["Attribute16"]["label"], 1, 4, 1
            )
            meta = FEATURE_META["Attribute14"]
            raw_input["Attribute14"] = st.selectbox(
                meta["label"], list(meta["options"].keys()),
                format_func=lambda k: meta["options"][k], index=2
            )
            meta = FEATURE_META["Attribute10"]
            raw_input["Attribute10"] = st.selectbox(
                meta["label"], list(meta["options"].keys()),
                format_func=lambda k: meta["options"][k], index=0
            )

        # Column 3 — personal & employment
        with c3:
            st.markdown(f"<div class='section-title'>Personal & employment</div>", unsafe_allow_html=True)
            raw_input["Attribute13"] = st.slider(
                FEATURE_META["Attribute13"]["label"], 19, 75, 35
            )
            meta = FEATURE_META["Attribute7"]
            raw_input["Attribute7"] = st.selectbox(
                meta["label"], list(meta["options"].keys()),
                format_func=lambda k: meta["options"][k], index=2
            )
            meta = FEATURE_META["Attribute9"]
            raw_input["Attribute9"] = st.selectbox(
                meta["label"], list(meta["options"].keys()),
                format_func=lambda k: meta["options"][k], index=2
            )
            meta = FEATURE_META["Attribute15"]
            raw_input["Attribute15"] = st.selectbox(
                meta["label"], list(meta["options"].keys()),
                format_func=lambda k: meta["options"][k], index=1
            )
            meta = FEATURE_META["Attribute17"]
            raw_input["Attribute17"] = st.selectbox(
                meta["label"], list(meta["options"].keys()),
                format_func=lambda k: meta["options"][k], index=2
            )

    # Fill remaining features with defaults (not shown in UI for brevity)
    for attr in CATEGORICAL_COLS:
        if attr not in raw_input:
            raw_input[attr] = list(FEATURE_META[attr]["options"].keys())[0]
    for attr in NUMERIC_COLS:
        if attr not in raw_input:
            raw_input[attr] = FEATURE_META[attr]["default"]

    # ── Predict ────────────────────────────────────────────────────────────────
    row_enc  = encode_input(raw_input, feature_cols)
    row_sc   = scaler.transform(row_enc)
    mdl      = models[model_choice]
    pd_score = float(mdl.predict_proba(row_sc)[0, 1])
    pd_pct   = pd_score * 100
    risk_lbl, risk_col, verdict_lbl = pd_label(pd_score)

    # ── Results row ────────────────────────────────────────────────────────────
    st.markdown("---")
    ra, rb, rc = st.columns([1.4, 1.2, 1.4])

    # PD Gauge
    with ra:
        st.markdown(f"<div class='section-title'>Probability of default (PD)</div>", unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pd_pct,
            number={"suffix": "%", "font": {"size": 36, "color": OFFWHT}},
            gauge={
                "axis":  {"range": [0, 100], "tickfont": {"color": SLATE, "size": 10}},
                "bar":   {"color": risk_col, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 25],  "color": "rgba(46,139,110,0.15)"},
                    {"range": [25, 50], "color": "rgba(201,168,76,0.15)"},
                    {"range": [50,100], "color": "rgba(192,57,43,0.15)"},
                ],
                "threshold": {
                    "line": {"color": GOLD, "width": 2},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
        ))
        gauge.update_layout(
            height=220, margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color=OFFWHT,
        )
        st.plotly_chart(gauge, use_container_width=True)
        st.markdown(f"""
        <div style='text-align:center; font-size:12px; color:{SLATE}; margin-top:-12px;'>
          Gold line = active threshold ({threshold:.0%})
        </div>
        """, unsafe_allow_html=True)

    # Risk metrics
    with rb:
        st.markdown(f"<div class='section-title'>Risk signal</div>", unsafe_allow_html=True)
        for label_txt, val_txt in [
            ("PD score",      f"{pd_score:.3f}"),
            ("Risk tier",     risk_lbl),
            ("Model",         model_choice),
            ("Threshold",     f"{threshold:.2f}"),
        ]:
            colour = risk_col if label_txt == "Risk tier" else GOLD
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; padding:8px 0;
                        border-bottom:1px solid rgba(255,255,255,0.05); font-size:13px;'>
              <span style='color:{SLATE};'>{label_txt}</span>
              <span style='color:{colour}; font-weight:600;'>{val_txt}</span>
            </div>
            """, unsafe_allow_html=True)

    # Verdict
    with rc:
        st.markdown(f"<div class='section-title'>Recommendation</div>", unsafe_allow_html=True)
        if pd_score < 0.25:
            css_cls = "verdict-approve"
            icon = "✅"
            note = "PD is below the conservative threshold. Standard terms apply."
        elif pd_score < threshold:
            css_cls = "verdict-review"
            icon = "⚠️"
            note = "PD is moderate. Consider a risk premium or reduced credit limit before approving."
        else:
            css_cls = "verdict-decline"
            icon = "🚫"
            note = (f"PD exceeds the {threshold:.0%} threshold. "
                    "We recommend a second look or decline. "
                    "Missed defaults cost 5× more than false alarms.")

        st.markdown(f"""
        <div class='{css_cls}'>
          <div class='verdict-title'>{icon} {verdict_lbl}</div>
          <div style='font-size:13px; line-height:1.5; color:{OFFWHT};'>{note}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Feature-level explanation ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"<div class='section-title'>What's driving this assessment</div>", unsafe_allow_html=True)

    # Use LR coefficients as a proxy for feature contribution on this applicant
    coef_vec = np.array(models["Logistic Regression"].coef_[0])
    row_vals = row_enc[feature_cols].values[0]
    contrib  = coef_vec * row_vals  # element-wise contribution
    contrib_s = pd.Series(contrib, index=feature_cols)

    # Map encoded feature names back to friendly labels (best effort)
    def friendly_name(col):
        for base in FEATURE_META:
            if col.startswith(base):
                return FEATURE_META[base]["label"]
        return col

    contrib_labeled = contrib_s.copy()
    contrib_labeled.index = [friendly_name(c) for c in feature_cols]
    # Aggregate by label (multiple OHE dummies may share the same label)
    contrib_grouped = contrib_labeled.groupby(contrib_labeled.index).sum()
    top = contrib_grouped.abs().nlargest(10).index
    top_contrib = contrib_grouped[top].sort_values()

    colors = [RED if v > 0 else GREEN for v in top_contrib.values]
    fig_expl = go.Figure(go.Bar(
        x=top_contrib.values,
        y=top_contrib.index,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig_expl.add_vline(x=0, line_color=SLATE, line_width=1)
    fig_expl.update_layout(
        height=320,
        xaxis_title="Contribution to PD (positive = increases risk)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=OFFWHT,
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        margin=dict(l=0, r=0, t=10, b=30),
    )
    st.plotly_chart(fig_expl, use_container_width=True)
    st.markdown(f"""
    <div class='info-box'>
      <b>How to read this:</b> Red bars indicate features <em>increasing</em> default risk for this applicant;
      green bars <em>reduce</em> it. Based on Logistic Regression coefficients weighted by this applicant's values.
      Need more detail? Hover over any bar.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"""
    <div class='page-header'>
      <div class='page-title'>Model performance & insights</div>
      <div class='page-sub'>Held-out test set results across all three classifiers. The primary objective is minimising cost-weighted misclassification (5×FN + 1×FP).</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Summary table ──────────────────────────────────────────────────────────
    st.markdown(f"<div class='section-title'>Performance summary</div>", unsafe_allow_html=True)
    summary_rows = []
    for name, res in results.items():
        summary_rows.append({
            "Model":           name,
            "Accuracy":        f"{res['accuracy']:.1%}",
            "Recall (Bad)":    f"{res['recall']:.1%}",
            "Precision (Bad)": f"{res['precision']:.1%}",
            "F1 (Bad)":        f"{res['f1']:.1%}",
            "AUC-ROC":         f"{res['auc']:.3f}",
            "Total cost":      res['cost'],
            "FN (missed)":     res['fn'],
            "FP (declined)":   res['fp'],
        })
    sdf = pd.DataFrame(summary_rows)
    st.dataframe(sdf.set_index("Model"), use_container_width=True)

    st.markdown(f"""
    <div class='info-box' style='margin-top:8px;'>
      The model is deliberately biased toward catching defaults — that's by design for conservative lending.
      <b>Recall on the Bad class</b> is the metric that matters most; a missed default costs five times more than a false alarm.
      We recommend <b>{best_model_name}</b> for production use (lowest cost-weighted error on the test set).
    </div>
    """, unsafe_allow_html=True)

    # ── Charts row 1: Confusion matrix + ROC ──────────────────────────────────
    st.markdown("---")
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown(f"<div class='section-title'>Confusion matrix — {model_choice}</div>", unsafe_allow_html=True)
        cm  = results[model_choice]["cm"]
        n   = cm.sum()
        z   = cm.astype(float)
        pct = z / n * 100
        text= [[f"{int(cm[i,j])}<br>{pct[i,j]:.1f}%" for j in range(2)] for i in range(2)]
        fig_cm = go.Figure(go.Heatmap(
            z=z,
            x=["Predicted Good","Predicted Bad"],
            y=["Actual Good","Actual Bad"],
            text=text,
            texttemplate="%{text}",
            colorscale=[[0,"rgba(11,31,58,0.5)"], [0.5,SLATE], [1,GOLD]],
            showscale=False,
            hovertemplate="%{y} → %{x}: %{text}<extra></extra>",
        ))
        fig_cm.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=OFFWHT,
            xaxis=dict(side="bottom"),
            margin=dict(l=0,r=0,t=10,b=0),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with ch2:
        st.markdown(f"<div class='section-title'>ROC curve — all models</div>", unsafe_allow_html=True)
        fig_roc = go.Figure()
        colors_roc = [GREEN, GOLD, "#5B8CFF"]
        for (name, res), col in zip(results.items(), colors_roc):
            fig_roc.add_trace(go.Scatter(
                x=res["fpr"], y=res["tpr"],
                mode="lines",
                name=f"{name} (AUC={res['auc']:.3f})",
                line=dict(color=col, width=2.5),
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(color=SLATE, dash="dot", width=1),
            showlegend=False, name="Random",
        ))
        fig_roc.update_layout(
            height=300,
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            legend=dict(font_size=10, bgcolor="rgba(0,0,0,0)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=OFFWHT,
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            margin=dict(l=0,r=0,t=10,b=0),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── Charts row 2: Feature importance ──────────────────────────────────────
    st.markdown("---")
    fi1, fi2 = st.columns(2)

    with fi1:
        st.markdown(f"<div class='section-title'>Feature importance — Logistic Regression (|coef|)</div>", unsafe_allow_html=True)
        lr_top = lr_imp.nlargest(10).sort_values()
        fig_lr = go.Figure(go.Bar(
            x=lr_top.values,
            y=[friendly_name(c) for c in lr_top.index],
            orientation="h",
            marker_color=GOLD,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig_lr.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=OFFWHT,
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Absolute coefficient"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            margin=dict(l=0,r=0,t=10,b=0),
        )
        st.plotly_chart(fig_lr, use_container_width=True)

    with fi2:
        st.markdown(f"<div class='section-title'>Feature importance — Decision Tree (Gini gain)</div>", unsafe_allow_html=True)
        dt_top = dt_imp.nlargest(10).sort_values()
        fig_dt = go.Figure(go.Bar(
            x=dt_top.values,
            y=[friendly_name(c) for c in dt_top.index],
            orientation="h",
            marker_color=GREEN,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig_dt.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=OFFWHT,
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Gini importance"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            margin=dict(l=0,r=0,t=10,b=0),
        )
        st.plotly_chart(fig_dt, use_container_width=True)

    # ── Per-class precision/recall table ──────────────────────────────────────
    st.markdown("---")
    st.markdown(f"<div class='section-title'>Precision · Recall · F1 per class</div>", unsafe_allow_html=True)
    pr_rows = []
    for name, res in results.items():
        yp = (res["y_prob"] >= threshold).astype(int)
        for cls_idx, cls_name in [(0, "Good (0)"), (1, "Bad (1)")]:
            pr_rows.append({
                "Model": name,
                "Class": cls_name,
                "Precision": f"{precision_score(y_test, yp, pos_label=cls_idx, zero_division=0):.1%}",
                "Recall":    f"{recall_score(y_test, yp, pos_label=cls_idx, zero_division=0):.1%}",
                "F1":        f"{f1_score(y_test, yp, pos_label=cls_idx, zero_division=0):.1%}",
            })
    pr_df = pd.DataFrame(pr_rows)
    st.dataframe(pr_df.set_index(["Model","Class"]), use_container_width=True)
    st.markdown(f"""
    <div class='info-box' style='margin-top:8px;'>
      Metrics computed at the current threshold ({threshold:.2f}). Adjust the sidebar slider to see how recall vs precision
      trade off as the operating point shifts.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"""
    <div class='page-header'>
      <div class='page-title'>Batch applicant analysis</div>
      <div class='page-sub'>Upload a CSV of applicants — same 20 raw features as the training data — and get instant PD scores and decisions.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='info-box' style='margin-bottom:20px;'>
      <b>Expected format:</b> One row per applicant, columns named <code>Attribute1</code> through <code>Attribute20</code>,
      using the same coding as the UCI dataset (e.g. <code>A11</code>, <code>A12</code>… for checking account status).
      No target column required.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop your CSV here", type=["csv"])

    if uploaded:
        try:
            batch_raw = pd.read_csv(uploaded)
            st.markdown(f"<div style='font-size:13px; color:{SLATE}; margin-bottom:12px;'>{len(batch_raw):,} applicants loaded.</div>", unsafe_allow_html=True)

            # Encode and score
            batch_cat_cols = [c for c in CATEGORICAL_COLS if c in batch_raw.columns]
            batch_num_cols = [c for c in NUMERIC_COLS if c in batch_raw.columns]
            if not batch_cat_cols and not batch_num_cols:
                st.error("No recognised feature columns found. Check column names match Attribute1–Attribute20.")
            else:
                batch_enc = pd.get_dummies(batch_raw[batch_cat_cols + batch_num_cols],
                                           columns=batch_cat_cols, drop_first=True).astype(float)
                batch_enc = batch_enc.reindex(columns=feature_cols, fill_value=0.0)
                batch_sc  = scaler.transform(batch_enc)

                mdl_b     = models[model_choice]
                batch_pd  = mdl_b.predict_proba(batch_sc)[:, 1]
                batch_dec = pd.cut(batch_pd,
                                   bins=[0, 0.25, threshold, 1.0],
                                   labels=["Approve", "Review", "Decline"])

                out = batch_raw.copy()
                out["PD_score"]    = np.round(batch_pd, 4)
                out["Risk_tier"]   = ["Low" if p < 0.25 else "Medium" if p < threshold else "High" for p in batch_pd]
                out["Decision"]    = batch_dec

                # Summary metrics
                n_tot = len(out)
                n_app = (out["Decision"] == "Approve").sum()
                n_rev = (out["Decision"] == "Review").sum()
                n_dec = (out["Decision"] == "Decline").sum()

                m1, m2, m3, m4 = st.columns(4)
                for col, label, val, color in [
                    (m1, "Total applicants", f"{n_tot:,}", GOLD),
                    (m2, "Approve",           f"{n_app:,} ({n_app/n_tot:.0%})", GREEN),
                    (m3, "Manual review",     f"{n_rev:,} ({n_rev/n_tot:.0%})", GOLD),
                    (m4, "Decline",           f"{n_dec:,} ({n_dec/n_tot:.0%})", RED),
                ]:
                    col.markdown(f"""
                    <div class='metric-card'>
                      <div class='metric-label'>{label}</div>
                      <div class='metric-value' style='font-size:24px; color:{color};'>{val}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # PD distribution
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=batch_pd, nbinsx=40,
                    marker_color=GOLD, opacity=0.7,
                    name="PD distribution",
                    hovertemplate="PD: %{x:.2f} — Count: %{y}<extra></extra>",
                ))
                fig_dist.add_vline(x=threshold, line_color=RED, line_dash="dash",
                                   annotation_text=f"Threshold {threshold:.2f}", annotation_font_color=RED)
                fig_dist.update_layout(
                    height=260,
                    xaxis_title="Probability of default",
                    yaxis_title="Applicant count",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color=OFFWHT,
                    xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                    showlegend=False,
                    margin=dict(l=0,r=0,t=20,b=0),
                )
                st.plotly_chart(fig_dist, use_container_width=True)

                st.dataframe(out.head(50), use_container_width=True)

                # Download
                csv_buf = io.StringIO()
                out.to_csv(csv_buf, index=False)
                st.download_button(
                    "⬇  Download scored results (CSV)",
                    csv_buf.getvalue().encode(),
                    file_name="scored_applicants.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Could not process file: {e}")
    else:
        st.markdown(f"""
        <div style='text-align:center; padding:60px 0; color:{SLATE};'>
          <div style='font-size:40px; margin-bottom:12px;'>📂</div>
          <div style='font-size:14px;'>Upload a CSV to begin batch scoring.</div>
          <div style='font-size:12px; margin-top:6px;'>
            Need a sample file? Run the dashboard, go to the assessment tab, and manually export a few rows.
          </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='footer'>
  Decisions made with data - and a human in the loop. &nbsp;·&nbsp;
  UCI Statlog German Credit Dataset · Hofmann (1994) &nbsp;·&nbsp;
  Built with Streamlit + scikit-learn by Eshwaree Mathanki

</div>
""", unsafe_allow_html=True) 
