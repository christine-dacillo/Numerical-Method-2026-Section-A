"""
Series1 - Laboratory Exercise
============================================================
Exercise 1 : Convergence of  (1 + 1/n)^n  to  e   (compound interest)

Scenario
--------
We put $1 at 100 % annual interest and ask what it grows to after one
year when the interest is credited (compounded) n times per year:

        A(n) = (1 + 1/n)^n

"How often"  n (times per year)
------------------------------
    yearly ...........   1
    twice a year .....   2
    quarterly ........   4
    monthly ..........  12
    daily ............ 365
    hourly ...........   8 760
    every minute .....     525 600
    every second .....  31 536 000
    every millisecond  .  31 536 000 000
    every microsecond  .. 31 536 000 000 000
    every nanosecond  ... 31 536 000 000 000 000

As n grows, A(n) tends to e = 2.718281828...
Excellent approximation of the error:   e - A(n) ~ e / (2n)

IMPORTANT (numerical stability)
-------------------------------
The naive Python expression  (1 + 1/n)**n  breaks for the very large n
(for n >= 1e16,  1 + 1/n  rounds to 1.0  and the power returns 1.0).
We therefore use the mathematically identical, numerically stable form

        A(n) = exp( n * ln(1 + 1/n) )

implemented with  math.log1p  which keeps 1/n accurate for tiny values.
Both values are printed side by side so the difference is visible.

Output figure :  exercise1_histogram.png
"""
from __future__ import annotations

import math
import sys

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["figure.dpi"] = 130

# Windows console uses cp1252 by default; make all prints UTF-8 safe
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ----------------------------------------------------------------------
# 1. Data : every compounding frequency from "yearly" to "nanosecond"
# ----------------------------------------------------------------------
PERIODS = [
    ("yearly", 1),
    ("twice a year", 2),
    ("quarterly", 4),
    ("monthly", 12),
    ("daily", 365),
    ("hourly", 8_760),
    ("every minute", 525_600),
    ("every second", 31_536_000),
    ("every ms", 31_536_000_000),
    ("every \u03bcs", 31_536_000_000_000),
    ("every ns", 31_536_000_000_000_000),
    # the last one ~= number of nanoseconds in one year
]


def naive_a(n: float) -> float:
    """(1 + 1/n)**n computed exactly as written -- loses precision."""
    return (1.0 + 1.0 / n) ** n


def stable_a(n: float) -> float:
    """Numerically stable version:  exp( n * ln(1 + 1/n) )."""
    return math.exp(n * math.log1p(1.0 / n))


# ----------------------------------------------------------------------
# 2. Build the table
# ----------------------------------------------------------------------
E = math.e

rows: list[tuple[str, float, float, float, float]] = []
print("=" * 92)
print("EXERCISE 1  |  A(n) = (1 + 1/n)^n   converges to  e = %.6f" % E)
print("=" * 92)
hdr = f"{'How often':<15}{'n':>18}{'A(n) naive':>15}{'A(n) stable':>15}{'error |e-A|':>15}"
print(hdr)
print("-" * 92)
for name, n in PERIODS:
    an_naive = naive_a(n)
    an_stable = stable_a(n)
    err = abs(E - an_stable)
    rows.append((name, n, an_naive, an_stable, err))
    print(f"{name:<15}{n:>18,.0f}{an_naive:>15.6f}{an_stable:>15.6f}{err:>15.2e}")
print("-" * 92)
print("e = 2.718282  |  theoretical error e/(2n)  shrinks with n")
print("At n = 3.1536e16 (nanosecond) the naive value collapses to 1.0")
print("because 1 + 1/n is not representable in a float; the stable")
print("version returns e to full double precision.")
print("=" * 92)

# ----------------------------------------------------------------------
# 3. Figure 1 : histogram of A(n) per compounding period
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                               gridspec_kw={"width_ratios": [1.55, 1]})

labels = [r[0] for r in rows]
values = [r[3] for r in rows]          # stable values
errors = [r[4] for r in rows]          # |e - A(n)|

# (a) bar chart  -- the "histogram"
bar_colors = plt.cm.viridis([0.15 + 0.7 * i / max(len(values) - 1, 1)
                             for i in range(len(values))])
bars = ax1.bar(range(len(values)), values, color=bar_colors, edgecolor="black")
ax1.axhline(E, color="crimson", linestyle="--", lw=1.6,
            label=f"e = {E:.6f}")
for i, (v, e_serr) in enumerate(zip(values, errors)):
    ax1.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)

ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax1.set_ylabel("A(n) = (1 + 1/n)$^n$")
ax1.set_title("Exercise 1: convergence of (1 + 1/n)$^n$ to e")
ax1.set_ylim(1.85, 2.85)
ax1.legend(loc="lower right")
ax1.grid(axis="y", alpha=0.35)
ax1.set_axisbelow(True)

# (b) error vs n on a loglog scale with the theoretical e/(2n) line
ns = [float(r[1]) for r in rows]
ax2.loglog(ns, errors, "o-", color="navy", lw=1.6,
           label="numerical error |e - A(n)|")
th = [E / (2.0 * n) for n in ns]
ax2.loglog(ns, th, "--", color="crimson", lw=1.4,
           label="theory e/(2n)")
ax2.set_xlabel("compounding periods n")
ax2.set_ylabel("error")
ax2.set_title("Error shrinks like e/(2n)")
ax2.legend(fontsize=8)
ax2.grid(which="both", alpha=0.35)

fig.suptitle("Series1 - Exercise 1  |  (1 + 1/n)$^n$  ->  e  =  2.718282",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("exercise1_histogram.png", bbox_inches="tight")
print("\nSaved figure ->  exercise1_histogram.png")

if "--show" in sys.argv:
    plt.show()