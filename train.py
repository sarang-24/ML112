"""Train, compare and analyse 5 regression models for product demand prediction."""
import json, warnings
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg") # Use Agg backend for matplotlib to generate plots without a display
import matplotlib.pyplot as plt, seaborn as sns

# Import scikit-learn modules for preprocessing, pipeline, and models
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore") # Ignore warning messages for cleaner output

# Define feature columns and target variable
CAT = ["product_category", "season"] # Categorical features (text/categories)
NUM = ["historical_demand", "product_price", "discount", "promotion", "holiday", "previous_month_sales"] # Numerical features
TARGET = "demand" # The value we are trying to predict

# Load the dataset
df = pd.read_csv("data/product_demand.csv")
print("Missing values:\n", df.isna().sum().sum(), "\nDuplicates:", df.duplicated().sum())

# Clean the dataset by removing duplicates and missing values
df = df.drop_duplicates().dropna()

# ---------- 1. Exploratory Data Analysis (EDA) ----------
# Generate and save a few basic plots to understand the data
sns.set_style("whitegrid")
fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))

# Plot 1: Average demand by category
sns.barplot(data=df, x="product_category", y=TARGET, ax=ax[0], estimator=np.mean, errorbar=None)
ax[0].set_title("Average demand by category"); ax[0].tick_params(axis="x", rotation=30)

# Plot 2: Heatmap of seasonal demand per category
sns.heatmap(df.pivot_table(index="product_category", columns="season", values=TARGET), annot=True, fmt=".0f", cmap="YlGnBu", ax=ax[1])
ax[1].set_title("Seasonal demand pattern")

# Plot 3: Impact of promotion on demand
sns.barplot(data=df, x="promotion", y=TARGET, ax=ax[2], errorbar=None); ax[2].set_title("Promotion vs demand (0=No, 1=Yes)")

# Save the plots as an image
plt.tight_layout(); plt.savefig("outputs/eda.png", dpi=130); plt.close()

# ---------- 2. Data Splitting & Preprocessing ----------
X, y = df[CAT + NUM], df[TARGET] # Separate features (X) and target (y)

# Split the data: 80% for training the models, 20% for testing them on unseen data
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a preprocessor to handle our features before feeding them to the models
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT), # Convert categories into numbers (0s and 1s)
    ("num", StandardScaler(), NUM), # Scale numerical features so they have a mean of 0 and variance of 1
])

# Define the 5 different Machine Learning models we want to compare
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=250, learning_rate=0.08, max_depth=3, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.06, max_depth=4, subsample=0.9, colsample_bytree=0.9, random_state=42),
}

# ---------- 3. Model Training and Evaluation ----------
rows, fitted, preds = [], {}, {}

# Train and test each model one by one
for name, m in models.items():
    # Create a pipeline that first preprocesses the data, then trains the model
    pipe = Pipeline([("pre", pre), ("model", m)]).fit(X_tr, y_tr)
    
    # Predict demand on the unseen test data
    p = pipe.predict(X_te)
    
    # Calculate error metrics (MAE, MSE, RMSE) and accuracy (R2)
    mse = mean_squared_error(y_te, p)
    rows.append({"Model": name, "MAE": mean_absolute_error(y_te, p), "MSE": mse,
                 "RMSE": np.sqrt(mse), "R2": r2_score(y_te, p)})
    
    # Store the trained model and its predictions
    fitted[name], preds[name] = pipe, p

# Create a DataFrame of the results, sorted by R2 (higher is better)
res = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
res.to_csv("outputs/model_comparison.csv", index=False)
print("\n", res.round(3).to_string(index=False))

# Identify and save the best performing model
best = res.loc[0, "Model"]
joblib.dump(fitted[best], "models/best_model.joblib")
print("\nBest model:", best)

# ---------- 4. Comparison Plots ----------
# Plot the error metrics for all models to visually compare them
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
for a, metric in zip(ax, ["MAE", "MSE", "RMSE", "R2"]):
    sns.barplot(data=res, x="Model", y=metric, ax=a, errorbar=None); a.set_title(metric)
    a.tick_params(axis="x", rotation=40); a.set_xlabel("")
plt.tight_layout(); plt.savefig("outputs/model_comparison.png", dpi=130); plt.close()

# Plot Actual vs Predicted values for the best model to see how close the predictions are
plt.figure(figsize=(6, 6))
plt.scatter(y_te, preds[best], alpha=.35, s=12)
lim = [0, max(y_te.max(), preds[best].max())]; plt.plot(lim, lim, "r--") # Perfect prediction line
plt.xlabel("Actual demand"); plt.ylabel("Predicted demand"); plt.title(f"Actual vs Predicted ({best})")
plt.tight_layout(); plt.savefig("outputs/actual_vs_pred.png", dpi=130); plt.close()

# ---------- 5. Feature Importance Analysis ----------
# Determine which features (columns) were most important for the best model
pipe = fitted[best]
names = pipe["pre"].get_feature_names_out()
est = pipe["model"]

# Extract importance scores (works for tree-based models)
imp = est.feature_importances_ if hasattr(est, "feature_importances_") else np.abs(est.coef_)
fi = pd.Series(imp, index=[n.split("__")[1] for n in names]).sort_values(ascending=False)
fi.to_csv("outputs/feature_importance.csv", header=["importance"])

# Plot the top features
plt.figure(figsize=(8, 6)); fi.head(12)[::-1].plot.barh(color="teal")
plt.title(f"Top demand predictors ({best})"); plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=130); plt.close()

# ---------- 6. Final Analysis & Summary Generation ----------
# Calculate various business insights (promotion lift, category averages, safety stock)
promo = df.groupby("promotion")[TARGET].mean()
promo_lift = (promo[1] / promo[0] - 1) * 100
cat_avg = df.groupby("product_category")[TARGET].mean().sort_values(ascending=False)
season_avg = df.groupby("season")[TARGET].mean().sort_values(ascending=False)
hol = df.groupby("holiday")[TARGET].mean()

# Calculate safety stock based on the forecast error (95% service level)
overstock_gap = (y_te.values - preds[best])
safety = 1.65 * np.std(overstock_gap)

# Save these insights into a JSON file for the Streamlit app to use
summary = {
    "best_model": best, "metrics": res.iloc[0].to_dict(),
    "top_predictors": fi.head(5).round(4).to_dict(),
    "high_demand_categories": cat_avg.round(1).to_dict(),
    "seasonal_avg_demand": season_avg.round(1).to_dict(),
    "promotion_lift_pct": round(float(promo_lift), 1),
    "holiday_lift_pct": round(float((hol[1] / hol[0] - 1) * 100), 1),
    "safety_stock_units": round(float(safety), 1),
    "residual_std": round(float(np.std(overstock_gap)), 2),
}
json.dump(summary, open("outputs/summary.json", "w"), indent=2)

# Generate a final markdown report with answers to the case study questions
b = res.iloc[0]
report = f"""# Final Analysis – Product Demand Prediction

## Answers to the case-study questions
1. **Can product demand be predicted?** Yes. The best model ({best}) reaches R² = {b.R2:.3f}, MAE = {b.MAE:.2f} units, RMSE = {b.RMSE:.2f} units on unseen data.
2. **Which factors influence demand?** Top predictors: {', '.join(fi.head(5).index)}. See `feature_importance.png`.
3. **Which algorithm gives the best prediction?** {best} (highest R², lowest error). Full table in `model_comparison.csv`.
4. **How do promotions affect demand?** Promoted products sell on average {promo_lift:.1f}% more than non-promoted ones; holidays add ~{(hol[1]/hol[0]-1)*100:.1f}%.
5. **Can the model reduce overstocking?** Yes. Forecasts replace guesswork; ordering predicted demand + a safety stock of ~{safety:.0f} units (95% service level) avoids both stock-outs and excess inventory.
6. **Can predicted demand support inventory management?** Yes – used for reorder quantity = forecast + safety stock − current stock (implemented in the Streamlit app).

## High-demand products (avg units)
{cat_avg.round(1).to_string()}

## Seasonal demand patterns (avg units)
{season_avg.round(1).to_string()}

## Inventory planning implications
- Stock up on top categories before their peak season (see heatmap in `eda.png`).
- Time promotions on products with high price/discount sensitivity.
- Use the safety-stock buffer to cover forecast error; re-train monthly with new sales.
"""
open("outputs/final_analysis.md", "w").write(report)
print("\n" + report)
