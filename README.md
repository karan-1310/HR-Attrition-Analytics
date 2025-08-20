# HR Attrition Analytics

**Objective:** Understand and reduce attrition through descriptive analytics and simple, explainable models.

## Why this is useful (one line)
_Equips HR leaders with the key drivers of attrition and a practical way to flag at-risk employees for targeted retention actions._

## Methods (at a glance)
- Descriptive cuts: department/tenure/overtime
- **Logistic Regression** (baseline) + **Decision Tree** (interpretability)
- Feature importance chart to communicate drivers

## How to run
```bash
pip install -r requirements.txt
python src/run_analysis.py
```
Outputs are saved in `outputs/`:
- `01_logreg_roc.png`, `02_tree_metrics.json`
- `03_feature_importance.png`, `04_attrition_by_department.png`
- `01_logreg_metrics.json`, `executive_summary.txt`

## Interview highlights
- Tradeoff: calibrated baseline vs. explainable tree
- Translating model drivers into **retention pilots**
