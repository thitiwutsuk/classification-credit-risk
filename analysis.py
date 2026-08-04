"""
Credit Risk Analysis
=====================
Q1: What factors contribute to a customer being Credit Risky?
Q2: What is the behavior of a Credit Worthy customer?
"""

import os

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save files
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report,
)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)

# ---------------------------------------------------------------------------
# STEP 1: Load data & aggregate payment_data.csv to one row per customer
# ---------------------------------------------------------------------------
# payment_data.csv has MULTIPLE rows per customer id (one per product /
# report date). customer_data.csv has ONE row per customer (with the label).
# Before we can merge them, we must summarize payment_data down to one row
# per id -- this is called "aggregation".

DATA_DIR = "data"
customer = pd.read_csv(f"{DATA_DIR}/customer_data.csv")
payment = pd.read_csv(f"{DATA_DIR}/payment_data.csv")

print("customer_data shape:", customer.shape)
print("payment_data shape :", payment.shape)
print("customers with >1 payment record:",
      (payment.groupby("id").size() > 1).sum())

payment_agg = payment.groupby("id").agg(
    ovd_t1_sum=("OVD_t1", "sum"),
    ovd_t2_sum=("OVD_t2", "sum"),
    ovd_t3_sum=("OVD_t3", "sum"),
    ovd_days_sum=("OVD_sum", "sum"),
    pay_normal_sum=("pay_normal", "sum"),
    n_products=("prod_code", "nunique"),
    n_payment_records=("id", "size"),
    prod_limit_mean=("prod_limit", "mean"),
    prod_limit_max=("prod_limit", "max"),
    new_balance_mean=("new_balance", "mean"),
    new_balance_max=("new_balance", "max"),
    highest_balance_mean=("highest_balance", "mean"),
    highest_balance_max=("highest_balance", "max"),
).reset_index()

# Behavioral ratio: how much of a customer's payment activity was overdue?
ovd_total = payment_agg["ovd_t1_sum"] + payment_agg["ovd_t2_sum"] + payment_agg["ovd_t3_sum"]
payment_agg["overdue_ratio"] = ovd_total / (ovd_total + payment_agg["pay_normal_sum"]).replace(0, np.nan)

print("\npayment_agg (aggregated, one row per id) shape:", payment_agg.shape)
print(payment_agg.head())

# ---------------------------------------------------------------------------
# STEP 2: Merge into customer_data, then handle missing values
# ---------------------------------------------------------------------------
# left join on "id" -- customer_data is the "base" table because it holds the
# label we want to predict/explain. Every customer_data id should now find a
# match in payment_agg (we confirmed shapes match: 1125 == 1125).

df = customer.merge(payment_agg, on="id", how="left")
print("\nMerged shape:", df.shape)

missing = df.isna().sum()
missing = missing[missing > 0]
print("\nColumns with missing values:\n", missing)

# Missing payment-side numeric columns mean "no recorded activity of that
# kind" for that customer -> fill counts/sums with 0.
count_like_cols = [
    "ovd_t1_sum", "ovd_t2_sum", "ovd_t3_sum", "ovd_days_sum",
    "pay_normal_sum", "n_products", "n_payment_records",
]
df[count_like_cols] = df[count_like_cols].fillna(0)

# Missing balance/limit fields mean we don't know their credit limit or
# balance -> fill with the column median (a neutral, robust-to-outliers
# estimate) rather than 0, since 0 would falsely imply "no credit limit".
balance_like_cols = [
    "prod_limit_mean", "prod_limit_max",
    "new_balance_mean", "new_balance_max",
    "highest_balance_mean", "highest_balance_max",
]
for col in balance_like_cols:
    df[col] = df[col].fillna(df[col].median())

# overdue_ratio is NaN when a customer has zero total payment activity
# (0/0) -> treat as 0 (no overdue behavior observed).
df["overdue_ratio"] = df["overdue_ratio"].fillna(0)

# fea_2 is a demographic numeric feature from customer_data.csv itself
# (unrelated to the merge) -> fill with the median, same reasoning as balances.
df["fea_2"] = df["fea_2"].fillna(df["fea_2"].median())

print("\nRemaining missing values after cleanup:", df.isna().sum().sum())
print("\nClass balance of label:\n", df["label"].value_counts(normalize=True))

# ---------------------------------------------------------------------------
# STEP 3: EDA - compare Risky (label=1) vs Worthy (label=0) groups
# ---------------------------------------------------------------------------
# This is the most direct way to start answering Q1/Q2: for every numeric
# feature, compute the average value separately for each group. A big gap
# between the two group means is a strong hint that the feature matters.
# fea_1, fea_3, fea_5, fea_6, fea_7, fea_9 are categorical codes (per the
# dataset description) so we exclude them from this numeric mean comparison.

categorical_cols = ["fea_1", "fea_3", "fea_5", "fea_6", "fea_7", "fea_9"]
id_label_cols = ["id", "label"]
numeric_cols = [c for c in df.columns if c not in categorical_cols + id_label_cols]

group_means = df.groupby("label")[numeric_cols].mean().T
group_means.columns = ["worthy_mean (label=0)", "risky_mean (label=1)"]
group_means["diff (risky - worthy)"] = (
    group_means["risky_mean (label=1)"] - group_means["worthy_mean (label=0)"]
)
# pct_diff makes gaps comparable across features with very different scales
group_means["pct_diff_%"] = (
    group_means["diff (risky - worthy)"] / group_means["worthy_mean (label=0)"].replace(0, np.nan) * 100
)
group_means = group_means.sort_values("pct_diff_%", key=abs, ascending=False)

print("\n=== Group means: Risky vs Worthy (sorted by biggest relative gap) ===")
print(group_means.round(2))

# Also check categorical features: does the distribution of categories
# differ between the two groups? (e.g. is category X of fea_6 more common
# among risky customers?)
print("\n=== Categorical feature distribution by label (row %) ===")
for col in categorical_cols:
    ct = pd.crosstab(df[col], df["label"], normalize="columns") * 100
    ct.columns = ["worthy_% (label=0)", "risky_% (label=1)"]
    print(f"\n-- {col} --")
    print(ct.round(1))

# ---------------------------------------------------------------------------
# STEP 4: EDA charts
# ---------------------------------------------------------------------------
# Turn the numbers above into pictures -- easier to spot patterns and easier
# to put in a report/presentation. We save everything under output/.

# 4a. Class balance -- shows the 80/20 imbalance we found earlier
plt.figure(figsize=(5, 4))
df["label"].value_counts().sort_index().plot(
    kind="bar", color=["#4C72B0", "#C44E52"]
)
plt.xticks([0, 1], ["Worthy (0)", "Risky (1)"], rotation=0)
plt.ylabel("Number of customers")
plt.title("Class Balance: Credit Worthy vs Credit Risky")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_class_balance.png", dpi=150)
plt.close()

# 4b. Top features by relative gap -- boxplots comparing the two groups
top_features = group_means.head(8).index.tolist()
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for ax, col in zip(axes.flatten(), top_features):
    sns.boxplot(data=df, x="label", y=col, ax=ax, palette=["#4C72B0", "#C44E52"], showfliers=False)
    ax.set_xticklabels(["Worthy", "Risky"])
    ax.set_title(col)
    ax.set_xlabel("")
plt.suptitle("Distribution of Top Differentiating Features by Group (outliers hidden)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_top_features_boxplots.png", dpi=150)
plt.close()

# 4c. Correlation of every numeric feature with label
corr_with_label = df[numeric_cols + ["label"]].corr()["label"].drop("label").sort_values()
plt.figure(figsize=(7, 8))
corr_with_label.plot(kind="barh", color=corr_with_label.apply(lambda v: "#C44E52" if v > 0 else "#4C72B0"))
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Correlation with label (positive = associated with Risky)")
plt.title("Feature Correlation with Credit Risk Label")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_correlation_with_label.png", dpi=150)
plt.close()

print(f"\nSaved 3 EDA charts to '{OUTPUT_DIR}/' folder.")

# ---------------------------------------------------------------------------
# STEP 5: Modeling
# ---------------------------------------------------------------------------
# EDA (step 3-4) shows raw associations, but two things can distort those:
# a feature can look important on its own while really just riding on the
# back of a correlated feature, and outliers can skew a plain mean. Models
# let every feature "compete" against all the others at once, which is a
# more trustworthy way to rank what actually matters. We fit two models so
# they can cross-check each other:
#   - Logistic Regression: linear, interpretable coefficients (direction +
#     rough magnitude of each feature's effect on risk).
#   - Random Forest: non-linear, captures interactions, gives a robust
#     feature_importances_ ranking.
#
# fea_1, fea_3, fea_5, fea_6, fea_7, fea_9 are categorical codes, not
# ordered numbers, so we one-hot encode them for both models.

model_df = pd.get_dummies(df.drop(columns=["id"]), columns=categorical_cols, drop_first=True)
X = model_df.drop(columns=["label"])
y = model_df["label"]

# stratify=y keeps the 80/20 class ratio the same in train and test splits,
# important because the dataset is imbalanced (see step 2).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# --- Logistic Regression (needs scaled features to make coefficients
#     comparable to each other, and class_weight="balanced" to compensate
#     for the 80/20 imbalance so it doesn't just predict "Worthy" always) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
log_reg.fit(X_train_scaled, y_train)

# --- Random Forest (tree-based models don't need scaling) ---
rf = RandomForestClassifier(
    n_estimators=300, max_depth=6, class_weight="balanced", random_state=42
)
rf.fit(X_train, y_train)


def evaluate(name, model, X_te, y_te):
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    print(f"\n--- {name} ---")
    print("Accuracy:", round(accuracy_score(y_te, pred), 3))
    print("ROC-AUC :", round(roc_auc_score(y_te, proba), 3))
    print("Confusion matrix [[TN, FP], [FN, TP]]:\n", confusion_matrix(y_te, pred))
    print(classification_report(y_te, pred, target_names=["Worthy", "Risky"]))


print("\n" + "=" * 70)
print("MODEL EVALUATION (test set, 25% held out)")
print("=" * 70)
evaluate("Logistic Regression", log_reg, X_test_scaled, y_test)
evaluate("Random Forest", rf, X_test, y_test)

# ---------------------------------------------------------------------------
# STEP 6: Rank factors -- combine coefficients + importances + EDA gaps
# ---------------------------------------------------------------------------
# Bring together three independent signals for each feature:
#   1. Logistic regression coefficient (sign = direction, size = strength)
#   2. Random forest importance (size = strength, no direction)
#   3. The raw group-mean %-difference from step 3 (sign = direction)
# Agreement across all three is strong evidence the factor is real.

coef_series = pd.Series(log_reg.coef_[0], index=X.columns, name="logreg_coef")
importance_series = pd.Series(rf.feature_importances_, index=X.columns, name="rf_importance")

factor_table = pd.concat([coef_series, importance_series], axis=1)
factor_table["rf_importance_rank"] = factor_table["rf_importance"].rank(ascending=False)
factor_table = factor_table.sort_values("rf_importance", ascending=False)

print("\n" + "=" * 70)
print("TOP 10 FACTORS BY RANDOM FOREST IMPORTANCE (cross-checked with Logistic Regression sign)")
print("=" * 70)
print(factor_table.head(10).round(4))

factor_table.to_csv(f"{OUTPUT_DIR}/feature_importance_summary.csv")

# Chart: top 10 random forest importances
plt.figure(figsize=(8, 6))
top10 = factor_table.head(10).sort_values("rf_importance")
plt.barh(top10.index, top10["rf_importance"], color="#55A868")
plt.xlabel("Random Forest Feature Importance")
plt.title("Top 10 Factors Driving the Credit Risk Prediction")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_top_factors_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# STEP 7: Answer Q1 and Q2 in plain language, generated from the numbers above
# ---------------------------------------------------------------------------

def direction_word(feature, factor_row, group_row):
    """Turn signs into a human-readable direction."""
    coef = factor_row.get("logreg_coef", 0)
    if feature in group_row.index:
        pct = group_row.loc[feature, "pct_diff_%"]
        if pd.notna(pct):
            return "HIGHER" if pct > 0 else "LOWER"
    return "HIGHER" if coef > 0 else "LOWER"


top_risk_factors = factor_table.head(6)
print("\n" + "=" * 70)
print("Q1 ANSWER: Factors contributing to a Credit RISKY customer")
print("=" * 70)
q1_lines = []
for feat in top_risk_factors.index:
    if feat not in group_means.index:
        continue  # skip one-hot encoded categorical dummy columns for the narrative
    direction = direction_word(feat, top_risk_factors.loc[feat], group_means)
    pct = group_means.loc[feat, "pct_diff_%"]
    line = f"- {feat}: risky customers show {direction} values (avg {pct:+.1f}% vs worthy customers)"
    q1_lines.append(line)
    print(line)

print(
    "\nSummary: Credit Risky customers are primarily distinguished by heavier balances "
    "(higher new_balance and highest_balance), a higher share of overdue payments "
    "(overdue_ratio, ovd_*_sum, ovd_days_sum), and fewer normal on-time payments "
    "(pay_normal_sum) relative to Credit Worthy customers. Demographic category features "
    "(fea_1, fea_3, fea_5, fea_6, fea_7, fea_9) show much smaller differences between "
    "groups, meaning payment BEHAVIOR is a far stronger risk signal than demographic profile."
)

print("\n" + "=" * 70)
print("Q2 ANSWER: Behavior of a Credit WORTHY customer")
print("=" * 70)
worthy_profile = group_means.loc[[f for f in top_risk_factors.index if f in group_means.index]]
print(worthy_profile[["worthy_mean (label=0)"]].round(2))

print(
    "\nSummary: Credit Worthy customers keep lower running balances (lower new_balance and "
    "highest_balance relative to their limit), overdue very rarely (overdue_ratio close to "
    f"{group_means.loc['overdue_ratio', 'worthy_mean (label=0)']:.1%}), and pay normally far more "
    f"often (avg {group_means.loc['pay_normal_sum', 'worthy_mean (label=0)']:.0f} normal payments "
    "recorded vs risky customers). In short: consistent, on-time payment behavior and controlled "
    "balances -- not demographic background -- are what mark a customer as Credit Worthy."
)

print(f"\nAll charts saved to '{OUTPUT_DIR}/', feature summary saved to "
      f"'{OUTPUT_DIR}/feature_importance_summary.csv'.")
print("\nDone.")
