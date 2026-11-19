
import numpy as np
import csv


file_path = r"/FRESH_FLOW/datasets/freshflow_sales.csv"

with open(file_path, "r", newline="") as file:
    reader = csv.reader(file)

    headers = next(reader)
    rows = list(reader)

# Convert empty values into NaN
data = np.array([
    [float(value) if value.strip() != "" else np.nan
     for value in row]
    for row in rows
], dtype=float)

print("\n========== DATASET LOADED ==========")

print("Dataset loaded successfully")
print("Shape:", data.shape)
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])
print("Dimensions:", data.ndim)
print("Data Type:", data.dtype)

print("\nFirst 5 Records:")
print(data[:5])

print("\nColumn Names:")
print(headers)

print("\nColumn Indexes:")
print(list(enumerate(headers)))

print("\n========== MISSING VALUE ANALYSIS ==========")

for i, column in enumerate(headers):

    missing_count = np.sum(np.isnan(data[:, i]))

    print(column, ":", missing_count)

print("\n========== UNIQUE VALUE ANALYSIS ==========")

for i, column in enumerate(headers):

    unique_values = np.unique(data[:, i])

    print(column, ":", len(unique_values))

print("\n========== STATISTICAL ANALYSIS ==========")

for i, column in enumerate(headers):

    print("\nColumn:", column)

    print("Minimum:", np.nanmin(data[:, i]))
    print("Maximum:", np.nanmax(data[:, i]))
    print("Mean:", np.nanmean(data[:, i]))
    print("Median:", np.nanmedian(data[:, i]))
    print("Standard Deviation:", np.nanstd(data[:, i]))

print("\n========== DUPLICATE TRANSACTION CHECK ==========")

transaction_ids = data[:, 0]

# Ignore missing transaction IDs
valid_ids = transaction_ids[np.isfinite(transaction_ids)]

unique_ids, counts = np.unique(
    valid_ids,
    return_counts=True
)

duplicate_ids = unique_ids[counts > 1]

print("Duplicate Transaction IDs:", duplicate_ids)
print("Number of Duplicate IDs:", len(duplicate_ids))

print("\n========== OUTLET VALIDATION ==========")

outlet_ids = data[:, 2]

invalid_outlets = (
    (outlet_ids < 1) |
    (outlet_ids > 3) |
    (~np.isfinite(outlet_ids)) |
    (outlet_ids != np.floor(outlet_ids))
)

print("Invalid outlet records:", np.sum(invalid_outlets))

print("\n========== QUANTITY VALIDATION ==========")

quantity = data[:, 5]

invalid_quantity = (
    (~np.isfinite(quantity)) |
    (quantity <= 0)
)

print("Invalid quantity records:", np.sum(invalid_quantity))

print("\n========== UNIT PRICE VALIDATION ==========")

unit_price = data[:, 6]

invalid_price = (
    (~np.isfinite(unit_price)) |
    (unit_price <= 0)
)

print("Invalid unit prices:", np.sum(invalid_price))

print("\n========== UNIT COST VALIDATION ==========")

unit_cost = data[:, 7]

invalid_cost = (
    (~np.isfinite(unit_cost)) |
    (unit_cost < 0)
)

print("Invalid or missing unit costs:", np.sum(invalid_cost))

print("\n========== DISCOUNT VALIDATION ==========")

discount = data[:, 8]

invalid_discount = (
    (~np.isfinite(discount)) |
    (discount < 0) |
    (discount > 1)
)

print("Invalid discount records:", np.sum(invalid_discount))


print("\n========== CUSTOMER RATING VALIDATION ==========")

rating = data[:, 11]

invalid_rating = (
    ((rating < 1) | (rating > 5))
    & np.isfinite(rating)
)

missing_rating = ~np.isfinite(rating)

print("Invalid customer ratings:", np.sum(invalid_rating))
print("Missing customer ratings:", np.sum(missing_rating))

print("\n========== STOCK VALIDATION ==========")

stock = data[:, 9]

negative_stock = stock < 0
zero_stock = stock == 0

print("Negative stock records:", np.sum(negative_stock))
print("Zero stock records:", np.sum(zero_stock))
print("Missing stock records:", np.sum(~np.isfinite(stock)))

print("\n PRODUCT CATEGORY MAPPING ")

product_category_mapping = {
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 2,
    6: 2,
    7: 3,
    8: 4
}

product_code = data[:, 3]

valid_product_for_mapping = (
    np.isfinite(product_code) &
    (product_code == np.floor(product_code)) &
    np.isin(
        product_code,
        list(product_category_mapping.keys())
    )
)

expected_category = np.full(
    product_code.shape,
    np.nan
)

expected_category[valid_product_for_mapping] = np.array([
    product_category_mapping[int(value)]
    for value in product_code[valid_product_for_mapping]
])

print("Valid product codes:", np.sum(valid_product_for_mapping))

print(
    "Invalid product codes:",
    np.sum(~valid_product_for_mapping)
)

print("Expected Categories:")
print(expected_category)

print("\nFINAL DATASET SUMMARY")

print("Total Records:", data.shape[0])
print("Total Columns:", data.shape[1])

print("Duplicate Transaction IDs:", len(duplicate_ids))
print("Invalid Outlets:", np.sum(invalid_outlets))
print("Invalid Quantities:", np.sum(invalid_quantity))
print("Invalid Prices:", np.sum(invalid_price))
print("Invalid Costs:", np.sum(invalid_cost))
print("Invalid Discounts:", np.sum(invalid_discount))
print("Invalid Ratings:", np.sum(invalid_rating))
print("Invalid Product Codes:", np.sum(~valid_product_for_mapping))

print("\n ANALYSIS COMPLETED")

expected_headers = [
    "transaction_id",
    "day_index",
    "outlet_id",
    "product_code",
    "category_code",
    "quantity",
    "unit_price",
    "unit_cost",
    "discount_rate",
    "stock_before_units",
    "payment_code",
    "customer_rating"
]

print("\n DATASET PROFILE ")

print("Loaded Shape:", data.shape)
print("Expected Rows: 5000")
print("Actual Rows:", data.shape[0])

print("Expected Columns: 12")
print("Actual Columns:", data.shape[1])

print("Numeric Data Type:", data.dtype)

print("\nColumn Verification:")

if headers == expected_headers:
    print("All column names match.")
else:
    print("Column names do not match.")
    print("Actual Headers:", headers)

if data.shape[1] != 12:
    raise ValueError("Dataset must contain exactly 12 columns.")

print("\nMissing Values:")

for i, name in enumerate(expected_headers):
    print(name, ":", np.isnan(data[:, i]).sum())

(
    transaction_id,
    day,
    outlet,
    product,
    category,
    quantity,
    unit_price,
    unit_cost,
    discount,
    stock_before,
    payment,
    rating
) = data.T

print("\n NAMED ARRAYS CREATED ")

print("Transaction ID:", transaction_id[:5])
print("Day:", day[:5])
print("Outlet:", outlet[:5])
print("Product:", product[:5])
print("Category:", category[:5])
print("Quantity:", quantity[:5])
print("Unit Price:", unit_price[:5])
print("Unit Cost:", unit_cost[:5])
print("Discount:", discount[:5])
print("Stock:", stock_before[:5])
print("Payment:", payment[:5])
print("Rating:", rating[:5])



def is_integer_value(values):
    return np.isfinite(values) & (values == np.floor(values))

# Identifier and code checks
bad_transaction_id = ~is_integer_value(transaction_id) | (transaction_id <= 0)
bad_day = ~is_integer_value(day) | (day < 1) | (day > 30)
bad_outlet = ~is_integer_value(outlet) | ~np.isin(outlet, [1, 2, 3])
bad_product = ~is_integer_value(product) | ~np.isin(product, np.arange(1, 9))
bad_category = ~is_integer_value(category) | ~np.isin(category, [1, 2, 3, 4])
bad_payment = ~is_integer_value(payment) | ~np.isin(payment, [1, 2, 3, 4])

# Numeric business checks
bad_quantity = ~is_integer_value(quantity) | (quantity <= 0)
bad_price = ~np.isfinite(unit_price) | (unit_price <= 0)
bad_cost = ~np.isfinite(unit_cost) | (unit_cost < 0)  # zero is allowed
bad_discount = ~np.isfinite(discount) | (discount < 0) | (discount > 1)
bad_stock = ~np.isfinite(stock_before) | (stock_before < 0)
bad_rating = np.isfinite(rating) & ((rating < 1) | (rating > 5))
missing_rating = np.isnan(rating)

# Product-to-category mapping from the assignment:
# products 1-4 -> category 1; 5-6 -> category 2; 7 -> category 3; 8 -> category 4
product_to_category = np.array([1, 1, 1, 1, 2, 2, 3, 4])
category_mismatch = np.zeros(data.shape[0], dtype=bool)
valid_product_for_mapping = ~bad_product
category_mismatch[valid_product_for_mapping] = (
    category[valid_product_for_mapping]
    != product_to_category[product[valid_product_for_mapping].astype(int) - 1]
)

# Duplicate IDs: report all rows participating in a repeated ID.
finite_ids = transaction_id[np.isfinite(transaction_id)]
unique_ids, id_counts = np.unique(finite_ids, return_counts=True)
duplicate_id_values = unique_ids[id_counts > 1]
duplicate_id_rows = np.isin(transaction_id, duplicate_id_values)

issue_masks = {
    "bad_transaction_id": bad_transaction_id,
    "duplicate_transaction_id": duplicate_id_rows,
    "bad_day": bad_day,
    "bad_outlet": bad_outlet,
    "bad_product": bad_product,
    "bad_category": bad_category,
    "category_mismatch": category_mismatch,
    "bad_payment": bad_payment,
    "bad_quantity": bad_quantity,
    "bad_price_or_missing": bad_price,
    "bad_cost_or_missing": bad_cost,
    "bad_discount": bad_discount,
    "bad_stock": bad_stock,
    "bad_rating": bad_rating
}

print("QUALITY AUDIT")
for issue, mask in issue_masks.items():
    print(f"{issue}: {int(mask.sum())}")

print("Missing customer ratings (allowed but report):", int(missing_rating.sum()))
print("Unique duplicate transaction IDs:", len(duplicate_id_values))
print("Rows involved in duplicate IDs:", int(duplicate_id_rows.sum()))

core_valid = (
    ~bad_transaction_id & ~duplicate_id_rows &
    ~bad_day & ~bad_outlet & ~bad_product & ~bad_category &
    ~category_mismatch & ~bad_payment &
    ~bad_quantity & ~bad_price & ~bad_cost &
    ~bad_discount & ~bad_stock & ~bad_rating
)

clean_data = data[core_valid].copy()
print("Original transaction rows:", len(data))
print("Rows retained for initial analysis:", len(clean_data))
print("Rows excluded from initial analysis:", int((~core_valid).sum()))
print("Raw data unchanged:", data.shape)

q = clean_data[:, 5]
price = clean_data[:, 6]
cost = clean_data[:, 7]
disc = clean_data[:, 8]

gross_sales = q * price
discount_amount = gross_sales * disc
net_revenue = gross_sales - discount_amount
cogs = q * cost
gross_profit = net_revenue - cogs

gross_margin_pct = np.divide(
    gross_profit * 100,
    net_revenue,
    out=np.full(net_revenue.shape, np.nan, dtype=float),
    where=net_revenue != 0
)

average_selling_price = np.divide(
    net_revenue,
    q,
    out=np.full(net_revenue.shape, np.nan, dtype=float),
    where=q > 0
)

print("Gross sales total: ₹", np.sum(gross_sales))
print("Discount amount total: ₹", np.sum(discount_amount))
print("Net revenue total: ₹", np.sum(net_revenue))
print("Cost of goods sold: ₹", np.sum(cogs))
print("Gross profit: ₹", np.sum(gross_profit))

total_revenue = np.sum(net_revenue)
total_profit = np.sum(gross_profit)
overall_margin_pct = (100 * total_profit / total_revenue) if total_revenue != 0 else np.nan
print("Overall gross margin (%):", overall_margin_pct)

def describe(values, label):
    x = values[np.isfinite(values)]
    print(f"\\n{label} — n={len(x)}")
    if len(x) == 0:
        print("No valid observations")
        return
    print("Mean:", np.mean(x))
    print("Median:", np.median(x))
    print("Minimum:", np.min(x))
    print("Maximum:", np.max(x))
    print("Standard deviation (population):", np.std(x))
    print("25th percentile:", np.percentile(x, 25))
    print("75th percentile:", np.percentile(x, 75))

describe(net_revenue, "Net revenue per transaction")
describe(gross_profit, "Gross profit per transaction")
describe(q, "Quantity per transaction")

outlet_values = clean_data[:, 2].astype(int)
product_values = clean_data[:, 3].astype(int)

print("OUTLET COMPARISON")
outlet_summary = []
for code_value in [1, 2, 3]:
    mask = outlet_values == code_value
    outlet_summary.append([
        code_value, int(mask.sum()), np.sum(net_revenue[mask]),
        np.sum(gross_profit[mask]), np.sum(q[mask])
    ])
    print(outlet_summary[-1])

outlet_summary = np.array(outlet_summary, dtype=float)
print("Outlet ranking by revenue:", outlet_summary[np.argsort(outlet_summary[:, 2])[::-1], 0].astype(int))
print("Outlet ranking by gross profit:", outlet_summary[np.argsort(outlet_summary[:, 3])[::-1], 0].astype(int))

print("\\nPRODUCT COMPARISON")
product_summary = []
for code_value in range(1, 9):
    mask = product_values == code_value
    product_summary.append([
        code_value, int(mask.sum()), np.sum(net_revenue[mask]),
        np.sum(gross_profit[mask]), np.sum(q[mask])
    ])
    print(product_summary[-1])

product_summary = np.array(product_summary, dtype=float)
print("Product ranking by revenue:", product_summary[np.argsort(product_summary[:, 2])[::-1], 0].astype(int))
print("Product ranking by gross profit:", product_summary[np.argsort(product_summary[:, 3])[::-1], 0].astype(int))
print("Product ranking by units:", product_summary[np.argsort(product_summary[:, 4])[::-1], 0].astype(int))

clean_days = clean_data[:, 1].astype(int)
daily_revenue = np.array([
    np.sum(net_revenue[clean_days == day_no])
    for day_no in range(1, 31)
], dtype=float)

daily_change = np.diff(daily_revenue)
daily_pct_change = np.divide(
    daily_change * 100,
    daily_revenue[:-1],
    out=np.full(daily_change.shape, np.nan),
    where=daily_revenue[:-1] != 0
)
rolling_7_day = np.convolve(daily_revenue, np.ones(7) / 7, mode="valid")

print("Day | Revenue | Change | Change %")
for i in range(30):
    change = "—" if i == 0 else f"{daily_change[i-1]:.2f}"
    pct = "—" if i == 0 or not np.isfinite(daily_pct_change[i-1]) else f"{daily_pct_change[i-1]:.2f}%"
    print(i + 1, daily_revenue[i], change, pct)

print("7-day moving average values:", rolling_7_day)

def iqr_flags(values):
    finite = np.isfinite(values)
    x = values[finite]
    flags = np.zeros(values.shape, dtype=bool)
    if len(x) < 4:
        return flags, np.nan, np.nan
    q1, q3 = np.percentile(x, [25, 75])
    spread = q3 - q1
    low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
    flags[finite] = (x < low) | (x > high)
    return flags, low, high

revenue_outliers, revenue_low, revenue_high = iqr_flags(net_revenue)
price_outliers, price_low, price_high = iqr_flags(price)
discount_outliers, discount_low, discount_high = iqr_flags(disc)

# Z-score flags; guard against zero standard deviation.
rev_std = np.std(net_revenue)
if rev_std > 0:
    revenue_z = (net_revenue - np.mean(net_revenue)) / rev_std
    revenue_z_flags = np.abs(revenue_z) > 3
else:
    revenue_z_flags = np.zeros(net_revenue.shape, dtype=bool)

anomaly_mask = revenue_outliers | price_outliers | discount_outliers | revenue_z_flags
anomaly_rows = clean_data[anomaly_mask]
print("Revenue IQR bounds:", revenue_low, revenue_high)
print("Price IQR bounds:", price_low, price_high)
print("Discount IQR bounds:", discount_low, discount_high)
print("Anomaly rows flagged:", len(anomaly_rows))
print("Flagged transaction IDs:", anomaly_rows[:, 0])

def safe_corr(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return np.nan
    aa, bb = a[valid], b[valid]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return np.nan
    return np.corrcoef(aa, bb)[0, 1]

print("Correlation: discount and quantity:", safe_corr(disc, q))
print("Correlation: discount and gross margin:", safe_corr(disc, gross_margin_pct))

print("\\nDISCOUNT SCENARIOS")
for scenario_discount in [0.10, 0.20]:
    scenario_revenue = gross_sales * (1 - scenario_discount)
    scenario_profit = scenario_revenue - cogs
    revenue_delta = np.sum(scenario_revenue) - np.sum(net_revenue)
    profit_delta = np.sum(scenario_profit) - np.sum(gross_profit)
    print(f"Discount {scenario_discount:.0%}:")
    print("  Scenario revenue:", np.sum(scenario_revenue))
    print("  Scenario gross profit:", np.sum(scenario_profit))
    print("  Revenue change vs actual:", revenue_delta)
    print("  Gross profit change vs actual:", profit_delta)

def calculate_transaction_metrics(clean_array):
    quantities = clean_array[:, 5]
    prices = clean_array[:, 6]
    costs = clean_array[:, 7]
    discounts = clean_array[:, 8]

    gross = quantities * prices
    discount_value = gross * discounts
    revenue = gross - discount_value
    cost_of_goods = quantities * costs
    profit = revenue - cost_of_goods
    return {
        "gross_sales": gross,
        "discount_amount": discount_value,
        "net_revenue": revenue,
        "cogs": cost_of_goods,
        "gross_profit": profit
    }

pipeline_metrics = calculate_transaction_metrics(clean_data)
print("Pipeline rows processed:", len(pipeline_metrics["net_revenue"]))
print("Pipeline total net revenue:", np.sum(pipeline_metrics["net_revenue"]))
#%% md
## Phase 11 — Validation tests


# Hand-calculated example from the assignment learning guide:
# quantity=5, price=100, cost=60, discount=15%
test_quantity = np.array([5.0])
test_price = np.array([100.0])
test_cost = np.array([60.0])
test_discount = np.array([0.15])

test_gross = test_quantity * test_price
test_revenue = test_gross * (1 - test_discount)
test_profit = test_revenue - test_quantity * test_cost

np.testing.assert_allclose(test_gross, [500.0])
np.testing.assert_allclose(test_revenue, [425.0])
np.testing.assert_allclose(test_profit, [125.0])

# Zero denominator must not crash or produce infinity.
safe_division = np.divide(
    np.array([10.0, 20.0]),
    np.array([2.0, 0.0]),
    out=np.full(2, np.nan),
    where=np.array([2.0, 0.0]) != 0
)
assert safe_division[0] == 5.0 and np.isnan(safe_division[1])

print("All validation tests passed.")

import os
os.makedirs("outputs", exist_ok=True)

# Save daily trend
daily_output = np.column_stack([
    np.arange(1, 31),
    daily_revenue,
    np.r_[np.nan, daily_change],
    np.r_[np.nan, daily_pct_change]
])
np.savetxt(
    "outputs/daily_trends.csv",
    daily_output,
    delimiter=",",
    header="day_index,net_revenue,daily_change,daily_percentage_change",
    comments="",
    fmt="%.6f"
)

# Save outlet and product comparison
np.savetxt(
    "outputs/outlet_comparison.csv",
    outlet_summary,
    delimiter=",",
    header="outlet_id,transaction_count,net_revenue,gross_profit,units_sold",
    comments="",
    fmt="%.6f"
)
np.savetxt(
    "outputs/product_comparison.csv",
    product_summary,
    delimiter=",",
    header="product_code,transaction_count,net_revenue,gross_profit,units_sold",
    comments="",
    fmt="%.6f"
)

# Save anomaly transaction IDs and core values for review
if len(anomaly_rows):
    np.savetxt(
        "outputs/anomaly_register.csv",
        anomaly_rows,
        delimiter=",",
        header=",".join(expected_headers),
        comments="",
        fmt="%.6f"
    )

# Save audit counts; issue categories may overlap.
with open("outputs/data_quality_report.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["issue", "flagged_rows"])
    for issue, mask in issue_masks.items():
        writer.writerow([issue, int(mask.sum())])
    writer.writerow(["missing_customer_rating_allowed", int(missing_rating.sum())])

print("Exported analysis files to the outputs folder.")

