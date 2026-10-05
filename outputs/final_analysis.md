# Final Analysis – Product Demand Prediction

## Answers to the case-study questions
1. **Can product demand be predicted?** Yes. The best model (XGBoost) reaches R² = 0.989, MAE = 7.95 units, RMSE = 11.84 units on unseen data.
2. **Which factors influence demand?** Top predictors: previous_month_sales, historical_demand, promotion, discount, product_price. See `feature_importance.png`.
3. **Which algorithm gives the best prediction?** XGBoost (highest R², lowest error). Full table in `model_comparison.csv`.
4. **How do promotions affect demand?** Promoted products sell on average 30.5% more than non-promoted ones; holidays add ~15.8%.
5. **Can the model reduce overstocking?** Yes. Forecasts replace guesswork; ordering predicted demand + a safety stock of ~20 units (95% service level) avoids both stock-outs and excess inventory.
6. **Can predicted demand support inventory management?** Yes – used for reorder quantity = forecast + safety stock − current stock (implemented in the Streamlit app).

## High-demand products (avg units)
product_category
Groceries         357.2
Clothing          148.2
Home & Kitchen    103.5
Toys               96.3
Sports             84.4
Electronics        67.6

## Seasonal demand patterns (avg units)
season
Winter    151.1
Summer    142.5
Spring    139.1
Autumn    138.5

## Inventory planning implications
- Stock up on top categories before their peak season (see heatmap in `eda.png`).
- Time promotions on products with high price/discount sensitivity.
- Use the safety-stock buffer to cover forecast error; re-train monthly with new sales.
