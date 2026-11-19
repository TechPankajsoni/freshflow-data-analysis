"""
FreshFlow Sales Analysis - Final Report Generator
-------------------------------------------------
Run:
    python freshflow_final_report.py freshflow_sales.csv

Dependencies:
    pip install numpy

The script reads the supplied CSV, performs the checks and calculations described
in the FreshFlow analysis notebook, and creates:
    outputs/FreshFlow_Final_Report.html
    outputs/data_quality_report.csv
    outputs/outlet_comparison.csv
    outputs/product_comparison.csv
    outputs/daily_trends.csv
    outputs/anomaly_register.csv
"""

import csv
import html
import os
import sys
import numpy as np

EXPECTED_HEADERS = [
    "transaction_id", "day_index", "outlet_id", "product_code",
    "category_code", "quantity", "unit_price", "unit_cost",
    "discount_rate", "stock_before_units", "payment_code",
    "customer_rating"
]

PRODUCT_TO_CATEGORY = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 4}


def load_dataset(file_path):
    with open(file_path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        headers = next(reader)

    if headers != EXPECTED_HEADERS:
        raise ValueError(
            "CSV headers do not match the 12-column FreshFlow specification.\n"
            f"Expected: {EXPECTED_HEADERS}\nFound: {headers}"
        )

    data = np.genfromtxt(
        file_path, delimiter=",", skip_header=1,
        filling_values=np.nan, ndmin=2, encoding="utf-8-sig"
    )

    if data.shape[1] != 12:
        raise ValueError(f"Expected 12 columns, but found {data.shape[1]}.")

    return data


def integer_values(values):
    return np.isfinite(values) & (values == np.floor(values))


def safe_corr(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return np.nan
    x, y = a[valid], b[valid]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


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
    return flags, float(low), float(high)


def money(value):
    if not np.isfinite(value):
        return "N/A"
    return f"₹{value:,.2f}"


def number(value, digits=2):
    if not np.isfinite(value):
        return "N/A"
    return f"{value:,.{digits}f}"


def html_table(headers, rows):
    head = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(
            f"<td>{html.escape(str(x))}</td>" for x in row
        ) + "</tr>"
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "freshflow_sales.csv"
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Could not find '{csv_path}'. Place freshflow_sales.csv in this "
            "folder or pass its full path as an argument."
        )

    data = load_dataset(csv_path)
    rows, cols = data.shape

    # Assign named columns.
    (transaction_id, day, outlet, product, category, quantity,
     unit_price, unit_cost, discount, stock_before, payment, rating) = data.T

    # Data-quality audit. Issue counts can overlap.
    bad_transaction_id = ~integer_values(transaction_id) | (transaction_id <= 0)
    bad_day = ~integer_values(day) | (day < 1) | (day > 30)
    bad_outlet = ~integer_values(outlet) | ~np.isin(outlet, [1, 2, 3])
    bad_product = ~integer_values(product) | ~np.isin(product, np.arange(1, 9))
    bad_category = ~integer_values(category) | ~np.isin(category, [1, 2, 3, 4])
    bad_payment = ~integer_values(payment) | ~np.isin(payment, [1, 2, 3, 4])
    bad_quantity = ~integer_values(quantity) | (quantity <= 0)
    bad_price = ~np.isfinite(unit_price) | (unit_price <= 0)
    bad_cost = ~np.isfinite(unit_cost) | (unit_cost < 0)
    bad_discount = ~np.isfinite(discount) | (discount < 0) | (discount > 1)
    bad_stock = ~np.isfinite(stock_before) | (stock_before < 0)
    bad_rating = np.isfinite(rating) & ((rating < 1) | (rating > 5))
    missing_rating = np.isnan(rating)

    valid_product = ~bad_product
    expected_category = np.full(rows, np.nan)
    for code, cat in PRODUCT_TO_CATEGORY.items():
        expected_category[product == code] = cat
    category_mismatch = (
        valid_product & np.isfinite(category) &
        (category != expected_category)
    )

    finite_ids = transaction_id[np.isfinite(transaction_id)]
    unique_ids, id_counts = np.unique(finite_ids, return_counts=True)
    duplicate_values = unique_ids[id_counts > 1]
    duplicate_rows = np.isin(transaction_id, duplicate_values)

    issue_masks = {
        "Invalid transaction ID": bad_transaction_id,
        "Rows involved in duplicate transaction IDs": duplicate_rows,
        "Invalid day index": bad_day,
        "Invalid outlet code": bad_outlet,
        "Invalid product code": bad_product,
        "Invalid category code": bad_category,
        "Product-category mismatch": category_mismatch,
        "Invalid payment code": bad_payment,
        "Invalid quantity": bad_quantity,
        "Missing or invalid unit price": bad_price,
        "Missing or invalid unit cost": bad_cost,
        "Invalid discount rate": bad_discount,
        "Missing or negative stock value": bad_stock,
        "Invalid customer rating": bad_rating,
    }

    # Conservative analytical view: preserve raw data and exclude invalid rows.
    core_valid = (
        ~bad_transaction_id & ~duplicate_rows & ~bad_day &
        ~bad_outlet & ~bad_product & ~bad_category &
        ~category_mismatch & ~bad_payment & ~bad_quantity &
        ~bad_price & ~bad_cost & ~bad_discount & ~bad_stock &
        ~bad_rating
    )
    clean = data[core_valid].copy()
    if len(clean) == 0:
        raise ValueError("No rows remain after the quality checks; inspect the CSV.")

    q = clean[:, 5]
    price = clean[:, 6]
    cost = clean[:, 7]
    disc = clean[:, 8]
    clean_days = clean[:, 1].astype(int)
    clean_outlets = clean[:, 2].astype(int)
    clean_products = clean[:, 3].astype(int)

    gross_sales = q * price
    discount_amount = gross_sales * disc
    net_revenue = gross_sales - discount_amount
    cogs = q * cost
    gross_profit = net_revenue - cogs
    overall_revenue = float(np.sum(net_revenue))
    overall_profit = float(np.sum(gross_profit))
    overall_margin = 100 * overall_profit / overall_revenue if overall_revenue else np.nan

    # Outlet and product summaries.
    outlet_summary = []
    for code in [1, 2, 3]:
        mask = clean_outlets == code
        outlet_summary.append([
            code, int(mask.sum()), float(np.sum(net_revenue[mask])),
            float(np.sum(gross_profit[mask])), float(np.sum(q[mask]))
        ])
    product_summary = []
    for code in range(1, 9):
        mask = clean_products == code
        product_summary.append([
            code, int(mask.sum()), float(np.sum(net_revenue[mask])),
            float(np.sum(gross_profit[mask])), float(np.sum(q[mask]))
        ])

    outlet_array = np.asarray(outlet_summary, dtype=float)
    product_array = np.asarray(product_summary, dtype=float)
    top_outlet_revenue = int(outlet_array[np.argmax(outlet_array[:, 2]), 0])
    top_outlet_profit = int(outlet_array[np.argmax(outlet_array[:, 3]), 0])
    top_product_revenue = int(product_array[np.argmax(product_array[:, 2]), 0])
    top_product_profit = int(product_array[np.argmax(product_array[:, 3]), 0])
    top_product_units = int(product_array[np.argmax(product_array[:, 4]), 0])

    # Daily revenue, changes and moving average.
    daily_revenue = np.array([
        np.sum(net_revenue[clean_days == d]) for d in range(1, 31)
    ], dtype=float)
    daily_change = np.diff(daily_revenue)
    daily_pct_change = np.divide(
        daily_change * 100, daily_revenue[:-1],
        out=np.full(daily_change.shape, np.nan),
        where=daily_revenue[:-1] != 0
    )
    rolling_7 = np.convolve(daily_revenue, np.ones(7) / 7, mode="valid")
    best_day = int(np.argmax(daily_revenue) + 1)
    lowest_day = int(np.argmin(daily_revenue) + 1)

    # Anomaly detection: unusual values are flagged for review, not deleted.
    rev_flags, rev_low, rev_high = iqr_flags(net_revenue)
    price_flags, price_low, price_high = iqr_flags(price)
    disc_flags, disc_low, disc_high = iqr_flags(disc)
    rev_std = np.std(net_revenue)
    if rev_std > 0:
        z_flags = np.abs((net_revenue - np.mean(net_revenue)) / rev_std) > 3
    else:
        z_flags = np.zeros(len(net_revenue), dtype=bool)
    anomaly_mask = rev_flags | price_flags | disc_flags | z_flags
    anomaly_data = clean[anomaly_mask]

    # Static discount scenarios: hold quantity and unit cost constant.
    scenarios = []
    for rate in [0.10, 0.20]:
        scenario_revenue = gross_sales * (1 - rate)
        scenario_profit = scenario_revenue - cogs
        scenarios.append([
            f"{rate:.0%}", float(np.sum(scenario_revenue)),
            float(np.sum(scenario_profit)),
            float(np.sum(scenario_revenue) - overall_revenue),
            float(np.sum(scenario_profit) - overall_profit)
        ])

    corr_discount_quantity = safe_corr(disc, q)
    corr_discount_margin = safe_corr(
        disc, np.divide(
            gross_profit * 100, net_revenue,
            out=np.full(len(net_revenue), np.nan),
            where=net_revenue != 0
        )
    )

    # Export machine-readable evidence.
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/data_quality_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["issue", "flagged_rows"])
        for label, mask in issue_masks.items():
            writer.writerow([label, int(mask.sum())])
        writer.writerow(["Missing customer ratings (allowed)", int(missing_rating.sum())])

    np.savetxt(
        "outputs/outlet_comparison.csv", outlet_array, delimiter=",",
        header="outlet_id,transaction_count,net_revenue,gross_profit,units_sold",
        comments="", fmt="%.6f"
    )
    np.savetxt(
        "outputs/product_comparison.csv", product_array, delimiter=",",
        header="product_code,transaction_count,net_revenue,gross_profit,units_sold",
        comments="", fmt="%.6f"
    )
    daily_export = np.column_stack([
        np.arange(1, 31), daily_revenue,
        np.r_[np.nan, daily_change], np.r_[np.nan, daily_pct_change]
    ])
    np.savetxt(
        "outputs/daily_trends.csv", daily_export, delimiter=",",
        header="day_index,net_revenue,daily_change,daily_percentage_change",
        comments="", fmt="%.6f"
    )
    with open("outputs/anomaly_register.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(EXPECTED_HEADERS)
        writer.writerows(anomaly_data.tolist())

    # Build the written report.
    quality_rows = [[name, int(mask.sum())] for name, mask in issue_masks.items()]
    quality_rows.append(["Missing customer ratings (allowed)", int(missing_rating.sum())])
    outlet_rows = [[int(r[0]), int(r[1]), money(r[2]), money(r[3]), number(r[4])]
                   for r in outlet_summary]
    product_rows = [[int(r[0]), int(r[1]), money(r[2]), money(r[3]), number(r[4])]
                    for r in product_summary]
    daily_rows = []
    for i, rev in enumerate(daily_revenue):
        daily_rows.append([
            i + 1, money(rev),
            "—" if i == 0 else money(daily_change[i - 1]),
            "—" if i == 0 else (number(daily_pct_change[i - 1]) + "%"
                                if np.isfinite(daily_pct_change[i - 1]) else "N/A")
        ])
    scenario_rows = [[r[0], money(r[1]), money(r[2]), money(r[3]), money(r[4])]
                     for r in scenarios]

    # Basic, evidence-linked recommendations.
    recommendations = [
        f"Review the operating practices of Outlet {top_outlet_revenue}, which recorded "
        "the highest net revenue in the retained dataset. Compare staffing, product mix "
        "and operating processes with the other outlets.",
        f"Prioritise a margin and cost review for Product {top_product_profit}, the "
        "highest gross-profit product. Check whether its pricing, availability and "
        "promotion strategy can be maintained.",
        f"Investigate Day {best_day} and Day {lowest_day}, the highest- and lowest-revenue "
        "days respectively. Compare transaction counts, product mix and discount levels "
        "before changing staffing or replenishment plans."
    ]

    report = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FreshFlow Sales Analysis - Final Report</title>
<style>
body{{font-family:Arial,sans-serif;line-height:1.55;color:#243047;background:#f4f7fb;margin:0}}
main{{max-width:1050px;margin:30px auto;background:white;padding:38px;border-radius:12px}}
header{{background:#172554;color:white;padding:32px;border-radius:10px;margin-bottom:28px}}
h1{{margin:0 0 8px;font-size:30px}} h2{{color:#1e3a8a;border-bottom:2px solid #dbeafe;padding-bottom:7px;margin-top:30px}}
h3{{color:#334155}} .muted{{color:#64748b}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.card{{background:#eff6ff;padding:16px;border-radius:8px}} .metric{{font-size:22px;font-weight:bold;color:#1e40af}}
.table-wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;margin:12px 0 20px;font-size:14px}}
th,td{{border:1px solid #dbe2ea;padding:9px;text-align:left}} th{{background:#eaf1fb}}
.note{{background:#fff7ed;border-left:4px solid #f59e0b;padding:12px 15px}}
footer{{margin-top:35px;color:#64748b;font-size:12px;border-top:1px solid #e2e8f0;padding-top:14px}}
@media print{{body{{background:white}}main{{margin:0;padding:15px;box-shadow:none}}}}
</style>
</head>
<body><main>
<header>
<p>PYTHON DATA ANALYSIS PROJECT</p>
<h1>FreshFlow Sales Analysis</h1>
<p>Final Project Report | 30-Day Transaction Analysis</p>
<p class="muted" style="color:#cbd5e1">Generated automatically from the supplied FreshFlow sales CSV</p>
</header>

<h2>1. Executive Summary</h2>
<p>This report presents a numerical analysis of FreshFlow sales transactions using Python and NumPy.
It reviews data quality, financial performance, outlet and product performance, daily revenue,
statistical anomalies and fixed-discount scenarios. The raw input data is preserved; a separate
analytical dataset is created after applying the documented validation rules.</p>
<div class="cards">
<div class="card"><div class="muted">Input transactions</div><div class="metric">{rows:,}</div></div>
<div class="card"><div class="muted">Rows retained</div><div class="metric">{len(clean):,}</div></div>
<div class="card"><div class="muted">Net revenue</div><div class="metric">{money(overall_revenue)}</div></div>
<div class="card"><div class="muted">Gross profit</div><div class="metric">{money(overall_profit)}</div></div>
<div class="card"><div class="muted">Gross margin</div><div class="metric">{number(overall_margin)}%</div></div>
</div>

<h2>2. Introduction</h2>
<p>FreshFlow sales analysis is designed to understand transaction-level business performance.
The analysis uses the 12 numeric fields specified in the project dataset: transaction ID, day,
outlet, product, category, quantity, unit price, unit cost, discount rate, stock before sale,
payment code and customer rating.</p>

<h2>3. Objectives</h2>
<ul>
<li>Load and inspect the sales dataset using Python.</li>
<li>Identify missing, invalid, duplicated and inconsistent records.</li>
<li>Calculate sales, discounts, revenue, cost of goods sold and gross profit.</li>
<li>Compare outlets and products using revenue, profit and units sold.</li>
<li>Study daily revenue changes and detect unusual transaction values.</li>
<li>Compare fixed 10% and 20% discount scenarios with the observed results.</li>
</ul>

<h2>4. Dataset Overview</h2>
<p><b>Rows:</b> {rows:,} &nbsp; <b>Columns:</b> {cols} &nbsp;
<b>Data type:</b> numeric array &nbsp; <b>Observation period:</b> days 1–30 (as defined by the project).</p>
<p><b>Missing values by column:</b></p>
{html_table(["Column", "Missing values"], [[name, int(np.isnan(data[:, i]).sum())] for i, name in enumerate(EXPECTED_HEADERS)])}

<h2>5. Data Quality Audit</h2>
<p>Each validation rule is counted separately. Issue counts may overlap because one transaction can violate more than one rule.</p>
{html_table(["Validation check", "Flagged rows"], quality_rows)}
<p><b>Unique duplicate transaction IDs:</b> {len(duplicate_values)}.
<b>Rows involved in duplicate IDs:</b> {int(duplicate_rows.sum())}.
<b>Rows excluded from the initial analytical view:</b> {int((~core_valid).sum())}.</p>
<p>Rows with missing customer ratings are reported but are not excluded solely for that reason.
Duplicate IDs are excluded from the initial analysis pending manual review. The original array remains unchanged.</p>

<h2>6. Financial Performance</h2>
<p>Gross sales = quantity × unit price. Discount amount = gross sales × discount rate.
Net revenue = gross sales − discount amount. COGS = quantity × unit cost.
Gross profit = net revenue − COGS.</p>
{html_table(["Financial measure", "Result"], [
["Gross sales", money(np.sum(gross_sales))],
["Discount amount", money(np.sum(discount_amount))],
["Net revenue", money(overall_revenue)],
["Cost of goods sold (COGS)", money(np.sum(cogs))],
["Gross profit", money(overall_profit)],
["Overall gross margin", number(overall_margin) + "%"]
])}
<p>Gross margin is calculated as gross profit divided by net revenue, multiplied by 100.</p>

<h2>7. Outlet Performance</h2>
{html_table(["Outlet", "Transactions", "Net revenue", "Gross profit", "Units sold"], outlet_rows)}
<p>Highest net revenue: <b>Outlet {top_outlet_revenue}</b>.
Highest gross profit: <b>Outlet {top_outlet_profit}</b>.</p>

<h2>8. Product Performance</h2>
{html_table(["Product", "Transactions", "Net revenue", "Gross profit", "Units sold"], product_rows)}
<p>Highest net revenue: <b>Product {top_product_revenue}</b>.
Highest gross profit: <b>Product {top_product_profit}</b>.
Highest units sold: <b>Product {top_product_units}</b>.</p>
<p>Sales volume and profitability are different measures; the highest-volume product is not necessarily the highest-profit product.</p>

<h2>9. Daily Sales Trend</h2>
<p>Highest revenue day: <b>Day {best_day}</b> ({money(daily_revenue[best_day-1])}).
Lowest revenue day: <b>Day {lowest_day}</b> ({money(daily_revenue[lowest_day-1])}).
A seven-day moving average was calculated to smooth short-term fluctuations.</p>
{html_table(["Day", "Net revenue", "Change from previous day", "Percentage change"], daily_rows)}

<h2>10. Anomaly Detection</h2>
<p>Transactions are flagged when net revenue, unit price or discount rate falls outside the IQR
limits, or when transaction net revenue has an absolute z-score above 3.</p>
{html_table(["Measure", "Lower IQR limit", "Upper IQR limit"], [
["Net revenue", number(rev_low), number(rev_high)],
["Unit price", number(price_low), number(price_high)],
["Discount rate", number(disc_low), number(disc_high)]
])}
<p><b>Transactions flagged for review:</b> {int(anomaly_mask.sum())}.
These records are not automatically deleted. Statistical unusualness does not prove an error.</p>
<p>Transaction IDs flagged: {html.escape(", ".join(str(int(x)) for x in anomaly_data[:, 0] if np.isfinite(x)) or "None")}.</p>

<h2>11. Discount and Relationship Analysis</h2>
<p>Correlation between discount rate and quantity: <b>{number(corr_discount_quantity, 4)}</b>.
Correlation between discount rate and gross margin: <b>{number(corr_discount_margin, 4)}</b>.
Correlation describes association and does not establish causation.</p>

<h2>12. Discount Scenarios</h2>
<p>The scenarios apply one fixed discount rate to the same quantities and costs. They do not estimate how customer demand would change.</p>
{html_table(["Scenario", "Scenario revenue", "Scenario gross profit", "Revenue change vs actual", "Profit change vs actual"], scenario_rows)}

<h2>13. Business Recommendations</h2>
<ol>{''.join(f'<li>{html.escape(item)}</li>' for item in recommendations)}</ol>
<p>Management should assign an owner to each action, monitor net revenue and gross profit, and review results in the next reporting cycle.</p>

<h2>14. Conclusion</h2>
<p>The analysis provides a structured view of FreshFlow's recorded sales performance and the quality
of the available transaction data. The financial, outlet, product and daily summaries can support
further business review. Decisions should be based on both revenue and gross profit rather than
sales volume alone.</p>

<h2>15. Limitations</h2>
<ul>
<li>The analysis covers only the supplied 30-day transaction window and cannot establish annual seasonality.</li>
<li>Gross profit is not net profit because rent, payroll and other operating expenses are not included.</li>
<li>Discount scenarios hold quantity and unit cost constant and do not estimate customer demand response.</li>
<li>Correlation does not prove a causal relationship.</li>
<li>Transaction data alone cannot establish inventory waste or stock-flow reconciliation. A daily inventory companion file is required.</li>
<li>Flagged outliers require human investigation and should not be treated as confirmed errors.</li>
</ul>
<div class="note"><b>Important:</b> This report is based on the CSV supplied when the script is run.
Review flagged data-quality issues and anomaly transactions before using the results for operational decisions.</div>

<footer>FreshFlow Sales Analysis | Prepared using Python and NumPy | Generated report</footer>
</main></body></html>
"""

    report_path = "outputs/FreshFlow_Final_Report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\nFreshFlow final report generated successfully.")
    print("Input rows:", rows)
    print("Retained rows:", len(clean))
    print("Net revenue:", money(overall_revenue))
    print("Gross profit:", money(overall_profit))
    print("Gross margin:", number(overall_margin), "%")
    print("Anomaly rows:", int(anomaly_mask.sum()))
    print("Report:", os.path.abspath(report_path))
    print("Supporting CSV files: outputs/")


if __name__ == "__main__":
    main()
