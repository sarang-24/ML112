# Case Study 156 – Product Demand Prediction Using Machine Learning
B.Tech CSE 2024-28 | Machine Learning | Semester V

## Structure
| File | Purpose |
|---|---|
| `generate_data.py` | Creates `data/product_demand.csv` (synthetic; swap with real data using same columns) |
| `train.py` | Trains & compares Linear, Decision Tree, Random Forest, Gradient Boosting, XGBoost; metrics MAE/MSE/RMSE/R²; saves best model, plots, final analysis |
| `app.py` | Streamlit app – enter product details → expected demand + reorder suggestion |
| `outputs/` | Comparison table, plots, `final_analysis.md` (answers to the 6 questions) |

## Run
```bash
pip install -r requirements.txt
python generate_data.py     # skip if you use your own CSV
python train.py
streamlit run app.py
```
Required CSV columns: product_category, historical_demand, product_price, discount, promotion, season, holiday, previous_month_sales, demand.
