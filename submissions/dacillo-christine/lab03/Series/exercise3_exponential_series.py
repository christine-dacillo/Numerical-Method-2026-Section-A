"""
Series1 - Laboratory Exercise
============================================================
Exercise 3 :  e^x = sum_{n=0}^{infinity} x^n / n!      (x = 1)

The exponential function has the famous power-series expansion

        e^x = 1 + x + x^2/2! + x^3/3! + ...  =  sum_{n=0..inf} x^n / n!

For x = 1 this becomes the classic series for e:

        e = sum_{n=0..inf} 1 / n!

We implement the series iteratively with n running from 0 up to
N = 10,000 terms (the exercise's "up to 10,000"), each new term being
obtained from the previous one by

        term(n) = term(n-1) * x / n          (term(0) = 1)

so that no huge factorials are ever computed.

The table below shows the partial sums

        S_N = sum_{n=0}^{N} 1/n!

at N = 10, 100, 1 000, 10 000 together with |e - S_N|; their limit is
e = 2.7182818281828459 (double-precision value).

Output figure :  exercise3_histogram.png
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

X = 1.0                      # e^1 = e
N_MAX = 10_000               # summation up to 10,000 terms
E = math.e

CHECKPOINTS = [10, 100, 1_000, 10_000]


def partial_sums(x: float, n_max: int):
    """Yield (n, S_n, error) for n=0..n_max using term(n)=term(n-1)*x/n."""
    s = 0.0
    term = 1.0                 # x^0 / 0!
    small_steps = []
    for n in range(0, n_max + 1):
        s += term
        small_steps.append((n, s, abs(E - s)))
        term *= x / (n + 1)
    return small_steps


# ----------------------------------------------------------------------
# 1. Run the summation and print the table
# ----------------------------------------------------------------------
steps = partial_sums(X, N_MAX)
by_n = {n: (s, err) for n, s, err in steps}

print("=" * 96)
print("EXERCISE 3  |  e^x = sum x^n/n!  with  x = 1   (e = %.6f)" % E)
print("=" * 96)
print(f"{'N (terms used)':>18}{'S_N':>22}{'|e - S_N|':>16}{'correct digits':>18}")
print("-" * 96)
for n in CHECKPOINTS:
    s, err = by_n[n]
    digits = max(0, -int(math.floor(math.log10(err + 1e-300)))) if err > 0 else 17
    print(f"{n:>18,}{s:>22.10f}{err:>16.2e}{digits:>18}")
print("-" * 96)
print(f"final value  S_{N_MAX:,} = {by_n[N_MAX][0]:.15f}")
print("Terms beyond n ~ 20 are smaller than 1/21! ~ 5e-20 so the sum has")
print("already saturated at the double-precision value of e.")
print("=" * 96)

# ----------------------------------------------------------------------
# 2. Figure 3 : histogram of the partial sums + accuracy analysis
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

# (a) histogram of the partial sums at the checkpoints
s_vals = [by_n[n][0] for n in CHECKPOINTS]
ax1.bar(range(len(CHECKPOINTS)), s_vals, width=0.62,
        color="mediumpurple", edgecolor="black")
ax1.axhline(E, color="crimson", linestyle="--", lw=1.6,
            label=f"e = {E:.6f}")
for i, v in enumerate(s_vals):
    ax1.text(i, v + 0.012, f"{v:.6f}", ha="center", fontsize=9)
ax1.set_xticks(range(len(CHECKPOINTS)))
ax1.set_xticklabels([f"N = {n:,}" for n in CHECKPOINTS])
ax1.set_ylim(2.68, 2.74)
ax1.set_ylabel("S_N  =  sum_{n=0..N} 1/n!")
ax1.set_title("Exercise 3: partial sums reach e")
ax1.legend(loc="lower right")
ax1.grid(axis="y", alpha=0.35)
ax1.set_axisbelow(True)

# (b) correct digits vs N on a log-log scale with accuracy bands
ns = [n for n, _, _ in steps]
errs = [max(e, 1e-18) for _, _, e in steps]
digits = [-math.log10(e) for e in errs]

ax2.loglog(ns, digits, "-", color="darkgreen", lw=1.8,
           label="correct decimal digits of e")
ax2.axhspan(2, 3, color="orange", alpha=0.15)
ax2.axhspan(6, 7, color="yellowgreen", alpha=0.15)
ax2.axhspan(15, 17, color="lightgreen", alpha=0.15)
ax2.text(12, 2.55, "rough\n(>1e-2)", fontsize=7, color="darkorange")
ax2.text(60, 6.45, "engineering\n(1e-6..1e-2)", fontsize=7, color="darkgreen")
ax2.text(1400, 15.5, "high precision / machine precision (2e-16)",
         fontsize=7, color="green")
ax2.set_xlabel("number of terms N in the summation (log scale)")
ax2.set_ylabel("correct decimal digits  (log scale)")
ax2.set_title("How many correct digits each N buys")
ax2.legend(fontsize=8, loc="lower right")
ax2.grid(which="both", alpha=0.35)

fig.suptitle("Series1 - Exercise 3  |  e = sum 1/n!   up to N = 10,000",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("exercise3_histogram.png", bbox_inches="tight")
print("\nSaved figure ->  exercise3_histogram.png")

if "--show" in sys.argv:
    plt.show()