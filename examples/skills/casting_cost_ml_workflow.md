# Casting Cost ML Workflow Skill

Use this workflow when the task is about predicting casting cost from tabular data such as part number, vendor, material, weight, OD, ID, wall thickness, and standard cost.

## Workflow

1. Confirm target column and remove obvious leakage columns.
2. Normalize column names and inspect missing values.
3. Split features into numeric and categorical columns.
4. Engineer casting geometry proxies where available:
   - OD / ID ratio
   - wall thickness / OD
   - annular area proxy: OD^2 - ID^2
   - approximate volume proxy: annular area * wall thickness
   - weight per geometry proxy
5. Build preprocessing pipeline:
   - median imputation for numeric columns
   - most-frequent imputation for categorical columns
   - one-hot encoding for low/medium-cardinality categories
6. Test several models:
   - DummyRegressor baseline
   - Ridge/ElasticNet
   - RandomForestRegressor
   - ExtraTreesRegressor
   - HistGradientBoostingRegressor
   - optional XGBoost/LightGBM/CatBoost if installed
7. Evaluate using train/test split and cross-validation.
8. Report R², MAE, RMSE, and residual outliers.
9. Check leakage, overfitting, and vendor/material dominance.
10. Save the best pipeline with joblib and produce feature importance/permutation importance.

## Verification checklist

- Never report training R² as final quality.
- Compare against a simple baseline.
- Confirm target column is not used as an input feature.
- Use grouped validation by part family/vendor if repeated part numbers exist.
- Explain why R² is high or low in business terms.
