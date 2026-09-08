"""
Series1 - Laboratory Exercise
============================================================
Exercise 2 :  (a^h - 1)/h  settles at  ln(a)      (h -> 0)

We study the difference quotient of the exponential function

        q(a, h) = (a^h - 1) / h

for three different bases and shrinking step sizes h:

        a = 2               a = e = 2.71828...      a = 3
        h = 0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7

This quantity is the slope of the secant of a^x between x = 0 and x = h.
As h -> 0 the secant becomes the tangent and the quotient tends to the
derivative of a^x at 0, which is exactly ln(a):

        lim_{h->0} (a^h - 1)/h  =  ln(a)

    a = 2 :  0.6931        a = e : 1.0000        a = 3 : 1.0986

Tolerance criterion : the lab asks to continue shrinking h until the
computed quotient agrees with ln(a) within 1e-6 (tolerance x 10^-6).

Output figure :  exercise2_histogram.png
"""
from __future__ import annotations

import math
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["figure.dpi"] = 130

# Windows console uses cp1252 by default; make all prints UTF-8 safe
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

E = math.e

BASES = [(2.0, "a = 2   (ln a = 0.6931)", 2.0),
         (E,   "a = e   (ln a = 1.0000)", E),
         (3.0, "a = 3   (ln a = 1.0986)", 3.0)]

H_STEPS = [0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7]
TOL = 1e-6


def quotient(a: float, h: float) -> float:
    """The difference quotient (a^h - 1)/h."""
    return (a ** h - 1.0) / h


# ----------------------------------------------------------------------
# 1. Build the table
# ----------------------------------------------------------------------
print("=" * 100)
print("EXERCISE 2  |  q(a,h) = (a^h - 1)/h  ->  ln(a)   as  h -> 0")
print("=" * 100)
print(f"{'h':>10}{'(2^h-1)/h':>16}{'(e^h-1)/h':>16}{'(3^h-1)/h':>16}"
      f"{'| - ln2|':>13}{'| - ln3|':>13}")
print("-" * 100)

table: dict[float, list[float]] = {a: [] for a, _, _ in BASES}
tol_report: dict[float, float | None] = {a: None for a, _, _ in BASES}

for h in H_STEPS:
    vals = [quotient(a, h) for a, _, _ in BASES]
    for a, v in zip([b[0] for b in BASES], vals):
        table[a].append(v)
        if tol_report[a] is None and abs(v - math.log(a)) < TOL:
            tol_report[a] = h
    print(f"{h:>10.4g}"
          f"{vals[0]:>16.4f}{vals[1]:>16.4f}{vals[2]:>16.4f}"
          f"{abs(vals[0]-math.log(2.0)):>13.2e}{abs(vals[2]-math.log(3.0)):>13.2e}")

print("-" * 100)
print("limits ln(a):   0.6931        1.0000        1.0986")
print("tolerance 1e-6 reached for h =", {f"a={a:.0f}" if abs(a-2.0) < 1e-9 or abs(a-3.0) < 1e-9
                                         else f"a=e": tl for a, tl in tol_report.items()})
print("=" * 100)

# ----------------------------------------------------------------------
# 2. Figure 2 : histogram of the quotient for every h and base
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12.5, 5.6))

xpos = np.arange(len(H_STEPS))
width = 0.26
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for j, (a, name, lna) in enumerate(BASES):
    ys = table[a]
    ax.bar(xpos + (j - 1) * width, ys, width, label=name, color=colors[j],
           edgecolor="black", alpha=0.92)
    ax.axhline(lna, linestyle="--", color=colors[j], lw=1.4)

ax.set_xticks(xpos)
ax.set_xticklabels([f"h = {h:g}" for h in H_STEPS], rotation=45, ha="right")
ax.set_ylabel("(a$^h$ - 1)/h")
ax.set_ylim(0.55, 1.3)
ax.set_title("Exercise 2: bars of (a$^h$-1)/h per h; "
             "dashed lines are the limits ln(a)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.35)
ax.set_axisbelow(True)

fig.suptitle("Series1 - Exercise 2  |  (a$^h$ - 1)/h  ->  ln(a)   "
             "(0.6931, 1.0000, 1.0986)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("exercise2_histogram.png", bbox_inches="tight")
print("\nSaved figure ->  exercise2_histogram.png")

if "--show" in sys.argv:
    plt.show()