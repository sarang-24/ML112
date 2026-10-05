"""Generate a realistic synthetic retail sales dataset.
Replace data/product_demand.csv with a real dataset (same columns) if you have one."""
import numpy as np # Used for fast numerical operations and random number generation
import pandas as pd # Used for creating and saving the dataset as a CSV file

# Initialize a random number generator with a fixed seed (42) so the generated data is reproducible
rng = np.random.default_rng(42)

# Set the total number of records (rows) to generate
N = 6000

# Define base demand and base price for different product categories
# Format: "Category Name": (Average Demand, Average Price)
categories = {
    "Electronics": (60, 450), "Clothing": (120, 40), "Groceries": (300, 6),
    "Home & Kitchen": (90, 55), "Toys": (80, 25), "Sports": (70, 70),
}

# Define seasonal multipliers for each category
# For example, Winter boosts Clothing (1.30x) but drops Sports (0.85x)
season_effect = {
    "Winter":  {"Clothing": 1.30, "Electronics": 1.15, "Groceries": 1.05, "Home & Kitchen": 1.00, "Toys": 1.20, "Sports": 0.85},
    "Spring":  {"Clothing": 1.00, "Electronics": 0.95, "Groceries": 1.00, "Home & Kitchen": 1.10, "Toys": 0.95, "Sports": 1.10},
    "Summer":  {"Clothing": 0.90, "Electronics": 0.90, "Groceries": 1.10, "Home & Kitchen": 0.95, "Toys": 1.00, "Sports": 1.35},
    "Autumn":  {"Clothing": 1.15, "Electronics": 1.05, "Groceries": 1.00, "Home & Kitchen": 1.05, "Toys": 1.00, "Sports": 0.95},
}

# Randomly assign a category and season to each of the N rows
cat = rng.choice(list(categories), N)
season = rng.choice(list(season_effect), N)

# Extract the base demand and base price for the assigned categories
base_d = np.array([categories[c][0] for c in cat])
base_p = np.array([categories[c][1] for c in cat])

# Generate random prices around the base price (between 70% and 140% of base price)
price = np.round(base_p * rng.uniform(0.7, 1.4, N), 2)

# Randomly assign discounts based on specified probabilities (e.g., 35% chance of 0 discount, 5% chance of 30%)
discount = rng.choice([0, 5, 10, 15, 20, 30], N, p=[.35, .15, .2, .15, .1, .05])

# Assign promotions: Higher chance of a promotion (60%) if there is a discount, else lower chance (10%)
promotion = (rng.random(N) < np.where(discount > 0, 0.6, 0.1)).astype(int)

# Randomly assign holidays (18% chance for any given record)
holiday = (rng.random(N) < 0.18).astype(int)

# Generate historical demand based on base demand with some random variation (60% to 150%)
hist = np.round(base_d * rng.uniform(0.6, 1.5, N)).astype(int)

# Generate previous month's sales based on historical demand (80% to 120%)
prev = np.round(hist * rng.uniform(0.8, 1.2, N)).astype(int)

# --- CALCULATE FINAL DEMAND TARGET VARIABLE ---

# 1. Apply season multipliers
s_mult = np.array([season_effect[s][c] for s, c in zip(season, cat)])
price_ratio = price / base_p

# 2. Base formula: weighted average of historical and previous sales, multiplied by the season effect
demand = (0.55 * hist + 0.35 * prev) * s_mult

# 3. Apply price elasticity (if price is higher than base, demand drops)
demand *= (1 - 0.6 * (price_ratio - 1))

# 4. Apply discount lift (higher discount increases demand)
demand *= (1 + discount / 100 * 1.4)

# 5. Apply promotion lift (active promotions boost demand by 18%)
demand *= np.where(promotion == 1, 1.18, 1.0)

# 6. Apply holiday lift (holidays boost demand by 15%)
demand *= np.where(holiday == 1, 1.15, 1.0)

# 7. Add random noise to simulate real-world unpredictability
demand += rng.normal(0, base_d * 0.06, N)

# Ensure demand doesn't drop below zero and is an integer
demand = np.clip(np.round(demand), 0, None).astype(int)

# Combine all generated arrays into a Pandas DataFrame
df = pd.DataFrame({
    "product_category": cat, "historical_demand": hist, "product_price": price,
    "discount": discount, "promotion": promotion, "season": season,
    "holiday": holiday, "previous_month_sales": prev, "demand": demand,
})

# Save the dataset to a CSV file in the 'data' directory
df.to_csv("data/product_demand.csv", index=False)
print("Saved data/product_demand.csv", df.shape)
