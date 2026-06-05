I want to build a machine learning model for engine casting cost prediction.

Available columns:
- part number
- weight
- material used
- factory standard cost
- part material
- vendor
- outer diameter
- inner diameter
- wall thickness

Not all geometry attributes exist for every row.

Create a practical ML plan and Python script design that:
1. Loads an Excel file.
2. Cleans column names.
3. Handles missing values.
4. Encodes vendor and material.
5. Engineers geometry features from OD, ID, wall thickness.
6. Tests multiple models.
7. Evaluates R2, MAE, RMSE using train/test split and cross-validation.
8. Checks data leakage.
9. Explains which model is most likely to work best and why.
10. Shows how to save the best model.

Assume the target is factory standard cost.
