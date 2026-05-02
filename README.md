# German Credit Risk – Notebook + Interactive Dashboard

**End‑to‑end credit risk analysis:**  
- `German_Credit_Classification.ipynb` – model training, evaluation, and threshold analysis.  
- `dashboard/app.py` – a live Streamlit tool that lets loan officers score applicants in real time.

**Best model (by business cost):** Logistic Regression – lowest cost‑weighted misclassification on the test set.  
**Best model (by recall):** Decision Tree – catches more actual defaults.

The dashboard uses the **cost‑based champion** by default. The trade‑off is explained below.

---

## The stakes

Every day, retail banks make thousands of binary decisions: approve or decline. Get it right and you earn the net interest margin on a repaid loan. Get it wrong in one direction and you decline a creditworthy customer — an opportunity cost. Get it wrong in the other direction and you approve a borrower who defaults — a write‑off, recovery costs, and a drag on capital ratios that regulators notice.

The global consumer finance market stood at approximately **$1.4 trillion in 2024**, on a trajectory toward $2.9 trillion by 2035. Within that volume, credit risk is the primary lever separating profitable lending portfolios from ones that require provisions and regulatory intervention. Research on large consumer credit bureaus has documented that conventional scoring models systematically misclassify subprime borrowers – with actual default rates of 44–95% across risk tiers – implying that the cost of a miscalibrated model is not academic. It is measured in written‑off principal, in elevated cost of funds, and in the Basel III capital charges that follow a deteriorating loan book.

The problem is asymmetric by design. Missing a default (approving a borrower who subsequently fails) costs five times more than declining a borrower who would have repaid. This is not a modelling assumption. It is encoded directly in the cost matrix Hans Hofmann distributed with the dataset in 1994, and it reflects the basic economics of credit: a missed net interest margin is a fraction of a percent; a write‑off is a multiple of it.

---

## The dataset

The Statlog German Credit dataset, donated to the UCI Machine Learning Repository by Prof. Hans Hofmann of the University of Hamburg, captures 1,000 credit applicants across 20 attributes: checking account status, loan duration, credit history, loan purpose, savings level, employment tenure, instalment rate as a percentage of disposable income, personal status, collateral, age, and several others. Each applicant is labelled Good (700 cases) or Bad (300 cases) credit risk, based on observed repayment outcomes.

The 70/30 class split is representative of real retail credit portfolios. The explicit cost matrix — 5 units for approving a defaulter, 1 unit for declining a good borrower — makes this one of the few benchmark datasets that forces the analyst to think in business terms rather than statistical ones.

---

## What this model enables

**Risk‑adjusted pricing.** A classifier that outputs a probability rather than a binary label feeds directly into a rate‑setting engine. High‑confidence Good applicants receive the headline rate; borderline applicants receive a risk premium or a reduced limit. This converts a gate into a margin management tool.

**Threshold optimisation.** The standard 0.5 decision boundary is cost‑neutral by construction. It is almost never cost‑optimal when a 5:1 asymmetry governs the loss function. The notebook's threshold analysis surfaces the operating point that minimises cost‑weighted misclassification, which is typically in the 0.3–0.45 range — a result that has material P&L implications at portfolio scale.

**Regulatory compliance.** Under GDPR's right‑to‑explanation provisions and Basel III's model governance requirements, lenders must be able to explain individual credit decisions in terms a borrower can understand and a regulator can audit. A Decision Tree or a Logistic Regression with interpretable coefficients satisfies this requirement. A black‑box gradient‑boosted ensemble, however accurate, introduces model risk that many retail lenders cannot accept.

**Early‑warning monitoring.** Run against a cohort of recently approved loans on a rolling basis, the model's posterior probability outputs flag applicants whose creditworthiness has deteriorated — enabling proactive collections activity before accounts reach 90 days past due.

---

## Why accuracy is the wrong metric here

A model that predicts everyone Good achieves 70% accuracy — and catches zero defaults. The correct objectives are:

1. **Minimise cost‑weighted misclassification** (5×false negatives + 1×false positives).
2. **Maximise recall on the Bad class** — the proportion of actual defaults the model correctly flags.
3. **AUC‑ROC** — discrimination ability across all possible thresholds, independent of the chosen operating point.

This notebook operationalises all three throughout: from class‑weight grids in `GridSearchCV` (including the `{Good:1, Bad:5}` option that directly mirrors the cost matrix), through cost‑sensitive evaluation on the test set, to threshold sensitivity analysis and error decomposition.

---

## Notebook pipeline (technical overview)

| Step | Detail |
|------|--------|
| Data | UCI Statlog German Credit — 1,000 samples, 20 features (13 categorical, 7 numeric) |
| Encoding | One‑hot encoding with `drop='first'` to avoid the dummy variable trap |
| Split | Stratified 60 / 20 / 20 train‑val‑test |
| Scaling | `StandardScaler` fit on training data only |
| Models | Logistic Regression, k‑NN, Decision Tree – tuned via `GridSearchCV` with `StratifiedKFold` |
| Scoring | Weighted F1 for tuning; cost matrix evaluation on test set |
| Threshold | Sensitivity analysis across [0.1, 0.9] to identify cost‑optimal operating point |
| Error analysis | FN profile inspection: confidence distribution, feature comparison, account‑type breakdown |

---

## Dashboard – interactive risk scoring

We built a **Streamlit dashboard** (`dashboard/app.py`) that wraps the trained models into a tool a credit officer can use without writing code.

**What it does:**  
- **Single applicant assessment** – input loan amount, credit history, savings, age, etc. Instantly get probability of default (PD), risk tier (Low/Medium/High), and a recommended action (Approve / Review / Decline).  
- **Model performance tab** – see confusion matrices, ROC curves, feature importance, and per‑class metrics for all three models.  
- **Batch analysis** – upload a CSV of applicants, score them all at once, download results.

**How it works under the hood:**  
The dashboard retrains the same three model families on a synthetic dataset that mirrors the original credit statistics. It applies identical preprocessing (one‑hot encoding, scaling) and uses the same cost‑based evaluation. The default active model is the one that minimises **cost‑weighted misclassification** (5×FN + 1×FP) on the dashboard’s internal test set – which, consistent with the notebook, is **Logistic Regression**.

**References**
**Dataset**

Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77

**Market size**

Market Research Future. (2024). *Consumer Finance Market Size, Share and Growth by 2035*. https://www.marketresearchfuture.com/reports/consumer-finance-market-24293

**Asymmetric misclassification costs**

Albanesi, S., & Domossy, D. (2019). *Predicting consumer default: A deep learning approach* (NBER Working Paper 26165). National Bureau of Economic Research. https://www.nber.org/papers/w26165

Zhang, W., et al. (2021). How to identify early defaults in online lending: A cost-sensitive multi-layer learning framework. *Knowledge-Based Systems*, 221, 106963. https://doi.org/10.1016/j.knosys.2021.106963

**Credit scoring and model governance**

Bussmann, N., et al. (2021). Employing explainable AI to optimize the return target function of a loan portfolio. *Frontiers in Artificial Intelligence*, 4. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8239154/


**Run it locally:**  
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py


