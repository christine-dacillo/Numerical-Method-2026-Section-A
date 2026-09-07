"""
Linear Regression Analysis
Relationship between total population aged 5-24 (x) and population aged 5-24
currently attending school (y), for the 17 Philippine regions.

Model:      y = a0 + a1*x
Formulas (least squares, done by hand -- NOT Excel trendline, NOT numpy.polyfit):
    a1 = [n*Sum(xy) - Sum(x)*Sum(y)] / [n*Sum(x^2) - (Sum(x))^2]
    a0 = ybar - a1*xbar

Requires: openpyxl, matplotlib
"""

import openpyxl
import matplotlib.pyplot as plt

# =========================================================================
# CONFIG -- EDIT THESE TO MATCH YOUR EXCEL FILE
# =========================================================================
FILE_PATH = r"C:\Users\Camille\Downloads\Statistical Tables_0.xlsx"

# The exact name of the worksheet (tab) that holds Table 1.
SHEET_NAME = "Table 1"

# The table is not laid out as one continuous block -- each region's total
# row is followed by several age-group breakdown rows, and every few
# regions there's a "Notes / Source" block from the original report.
# So instead of a start/end range, we list the EXACT row number of each
# of the 17 regions' TOTAL line (found by inspecting the sheet).
REGION_ROWS = [
    23,   # NATIONAL CAPITAL REGION
    34,   # CORDILLERA ADMINISTRATIVE REGION
    45,   # REGION I - ILOCOS REGION
    56,   # REGION II - CAGAYAN VALLEY
    86,   # REGION III - CENTRAL LUZON
    97,   # REGION IV-A-CALABARZON
    108,  # MIMAROPA REGION
    119,  # REGION V - BICOL REGION
    130,  # REGION VI - WESTERN VISAYAS
    160,  # REGION VII - CENTRAL VISAYAS
    171,  # REGION VIII - EASTERN VISAYAS
    182,  # REGION IX - ZAMBOANGA PENINSULA
    193,  # REGION X - NORTHERN MINDANAO
    204,  # REGION XI - DAVAO REGION
    233,  # REGION XII - SOCCSKSARGEN
    244,  # REGION XIII - CARAGA
    255,  # BANGSAMORO AUTONOMOUS REGION (IN MUSLIM MINDANAO)
]

# Column letters (as shown in Excel) for region name, x, and y.
COL_REGION = "A"   # region name
COL_X = "C"        # Total population aged 5-24 (thousands)
COL_Y = "D"        # Population aged 5-24 currently attending school (thousands)

# An x-value NOT among your 17 regions, used for the prediction demo.
NEW_X_FOR_PREDICTION = 1000  # thousands

# =========================================================================
# STEP 1: READ DATA FROM EXCEL
# =========================================================================
workbook = openpyxl.load_workbook(FILE_PATH, data_only=True)
sheet = workbook[SHEET_NAME]

regions = []
x_values = []
y_values = []

for row in REGION_ROWS:
    region_name = sheet[f"{COL_REGION}{row}"].value
    x_val = sheet[f"{COL_X}{row}"].value
    y_val = sheet[f"{COL_Y}{row}"].value

    # Skip any row that turned out blank just in case
    if x_val is None or y_val is None:
        continue

    regions.append(region_name)
    x_values.append(float(x_val))
    y_values.append(float(y_val))

n = len(x_values)
print(f"Number of regions read from file: {n}")
if n != 17:
    print("WARNING: expected 17 regions -- check REGION_ROWS in CONFIG.")

# =========================================================================
# STEP 2: CALCULATE THE SUMS NEEDED FOR THE LEAST-SQUARES FORMULAS
# =========================================================================
sum_x = sum(x_values)
sum_y = sum(y_values)
sum_x2 = sum(x ** 2 for x in x_values)
sum_xy = sum(x_values[i] * y_values[i] for i in range(n))

x_bar = sum_x / n
y_bar = sum_y / n

print("\n--- Summary Statistics ---")
print(f"n       = {n}")
print(f"Sum(x)  = {sum_x:.4f}")
print(f"Sum(y)  = {sum_y:.4f}")
print(f"Sum(x^2)= {sum_x2:.4f}")
print(f"Sum(xy) = {sum_xy:.4f}")
print(f"x_bar   = {x_bar:.4f}")
print(f"y_bar   = {y_bar:.4f}")

# =========================================================================
# STEP 3: CALCULATE SLOPE (a1) AND INTERCEPT (a0) USING THE GIVEN FORMULAS
# =========================================================================
a1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
a0 = y_bar - a1 * x_bar

print("\n--- Regression Coefficients ---")
print(f"slope (a1)     = {a1:.6f}")
print(f"intercept (a0) = {a0:.6f}")
print(f"\nRegression equation: y = {a0:.4f} + {a1:.4f}x")

# =========================================================================
# STEP 4: PREDICTED VALUES AND RESIDUALS
# =========================================================================
y_predicted = [a0 + a1 * x for x in x_values]
residuals = [y_values[i] - y_predicted[i] for i in range(n)]

print("\n--- Region-by-Region Results ---")
print(f"{'Region':<20}{'x':>10}{'y (actual)':>12}{'y (pred)':>12}{'residual':>12}")
for i in range(n):
    print(f"{str(regions[i]):<20}{x_values[i]:>10.2f}{y_values[i]:>12.2f}"
          f"{y_predicted[i]:>12.2f}{residuals[i]:>12.2f}")

# =========================================================================
# STEP 5: ERROR / GOODNESS-OF-FIT MEASURES
# =========================================================================
# Sr (also called SSE) = sum of squared residuals = "unexplained" variation
Sr = sum(r ** 2 for r in residuals)

# St = total sum of squares = "total" variation in y around its mean
St = sum((y - y_bar) ** 2 for y in y_values)

# r-squared = fraction of variation in y explained by the regression line
r_squared = (St - Sr) / St

# Standard error of the estimate (syx) -- typical size of a residual
# Divide by (n - 2) because 2 parameters (a0, a1) were estimated
syx = (Sr / (n - 2)) ** 0.5

print("\n--- Goodness of Fit ---")
print(f"Sr (SSE)              = {Sr:.4f}")
print(f"St (total SS)          = {St:.4f}")
print(f"r^2                    = {r_squared:.6f}")
print(f"standard error (Sy/x)  = {syx:.4f} (thousand persons)")

# =========================================================================
# STEP 6: PREDICTION FOR A NEW x-VALUE (NOT ONE OF THE 17 ORIGINAL POINTS)
# =========================================================================
y_new_prediction = a0 + a1 * NEW_X_FOR_PREDICTION

print("\n--- Prediction for a New x-value ---")
print(f"Using x = {NEW_X_FOR_PREDICTION} thousand (5-24 y/o population)")
print(f"y = {a0:.4f} + {a1:.4f}({NEW_X_FOR_PREDICTION})")
print(f"y = {y_new_prediction:.4f} thousand persons attending school")

# =========================================================================
# STEP 7: GRAPH 1 -- SCATTER PLOT WITH REGRESSION LINE
# =========================================================================
plt.figure(figsize=(8, 6))
plt.scatter(x_values, y_values, color="blue", label="Actual regional data")

# Draw the regression line across the range of x
x_line = [min(x_values), max(x_values)]
y_line = [a0 + a1 * x for x in x_line]
plt.plot(x_line, y_line, color="red",
         label=f"Regression line: y = {a0:.2f} + {a1:.4f}x")

plt.title("Population Aged 5-24 vs. Population Attending School\n(17 Philippine Regions)")
plt.xlabel("Total population aged 5-24 years (thousands)")
plt.ylabel("Population aged 5-24 attending school (thousands)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("scatter_with_regression_line.png", dpi=150)
plt.show()

# =========================================================================
# STEP 8: GRAPH 2 -- RESIDUAL PLOT
# =========================================================================
plt.figure(figsize=(8, 6))
plt.scatter(x_values, residuals, color="green")
plt.axhline(y=0, color="black", linestyle="--")  # reference line at 0

plt.title("Residual Plot")
plt.xlabel("Total population aged 5-24 years (thousands)")
plt.ylabel("Residual = actual y - predicted y (thousands)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("residual_plot.png", dpi=150)
plt.show()