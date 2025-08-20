
import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "hr_data.csv"))
y = df["Attrition"]
X = df.drop(columns=["Attrition", "EmployeeID"])

num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

pre = ColumnTransformer([("num","passthrough", num_cols),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])

logreg = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:,1]

report = classification_report(y_test, y_pred, output_dict=True)
auc = roc_auc_score(y_test, y_prob)

with open(os.path.join(OUT_DIR,"01_logreg_metrics.json"), "w") as f:
    import json; json.dump({"classification_report": report, "roc_auc": auc}, f, indent=2)

fig = plt.figure(figsize=(6,5))
RocCurveDisplay.from_estimator(logreg, X_test, y_test)
plt.title("Logistic Regression ROC")
fig.savefig(os.path.join(OUT_DIR,"01_logreg_roc.png"), bbox_inches="tight"); plt.close(fig)

tree = Pipeline([("pre", pre), ("clf", DecisionTreeClassifier(max_depth=4, random_state=42))])
tree.fit(X_train, y_train)
y_prob_tree = tree.predict_proba(X_test)[:,1]
auc_tree = roc_auc_score(y_test, y_prob_tree)

with open(os.path.join(OUT_DIR,"02_tree_metrics.json"), "w") as f:
    import json; json.dump({"roc_auc": auc_tree}, f, indent=2)

ohe_features = tree.named_steps["pre"].transformers_[1][1].get_feature_names_out(tree.named_steps["pre"].transformers_[1][2])
feature_names = num_cols + list(ohe_features)
importances = tree.named_steps["clf"].feature_importances_
imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(15)

fig = plt.figure(figsize=(8,5))
imp.set_index("feature")["importance"].plot(kind="bar")
plt.title("Top Feature Importances (Decision Tree)"); plt.xlabel("Feature"); plt.ylabel("Importance")
fig.savefig(os.path.join(OUT_DIR,"03_feature_importance.png"), bbox_inches="tight"); plt.close(fig)

attr = df.groupby("Department")["Attrition"].mean().sort_values(ascending=False)
fig = plt.figure(figsize=(8,5))
attr.plot(kind="bar")
plt.title("Attrition Rate by Department"); plt.xlabel("Department"); plt.ylabel("Attrition Rate")
fig.savefig(os.path.join(OUT_DIR,"04_attrition_by_department.png"), bbox_inches="tight"); plt.close(fig)

summary_lines = [
    f"Logistic Regression ROC-AUC: {auc:.3f}",
    f"Decision Tree ROC-AUC: {auc_tree:.3f}",
    "Top drivers (tree-based):"
] + [f"- {r.feature}: {r.importance:.3f}" for _, r in imp.iterrows()]

with open(os.path.join(OUT_DIR,"executive_summary.txt"), "w") as f:
    f.write("\\n".join(summary_lines))
