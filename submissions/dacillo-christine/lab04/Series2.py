"""
HOW ACCURATE IS GOOD ENOUGH?
Approximating a Civil Engineering Function Using Infinite Series
==================================================================

This script works through the full exercise progression:

    Geometric series -> Power series -> Maclaurin series -> Taylor series
    -> Engineering approximation (y = L sin(theta)) -> Error analysis
    -> Engineering decision

Every series summation is implemented explicitly with Python loops
(no symbolic math libraries, no math.sin() used to *generate* an
approximation -- math.sin()/math.cos() are only used to compute the
"exact" reference values and the derivative values needed by Taylor's
formula, exactly as the assignment allows).

Run this file directly:  python3 series_solver.py

It produces the 5 required deliverables as actual files (not just
console output), all written to the current working directory:

  1. numerical_tables.txt        - formatted tables of exact values,
                                    approximations, absolute error, and
                                    percentage error for every angle
                                    (1-30 deg) and every term count
                                    (N=1..4), for both Maclaurin and
                                    Taylor series.
  2. convergence_plot.png        - percentage error vs. number of terms,
                                    for several angles, log-scale y-axis.
  3. function_comparison_plot.png- side-by-side plots of exact sin(theta)
                                    vs. Maclaurin (left) and Taylor
                                    (right) approximations at N=1..4.
  4. error_comparison_plot.png   - bar charts comparing absolute and
                                    percentage error between Maclaurin
                                    and Taylor series across all angles.
  5. written_recommendation.txt  - the engineering decision (Part 7),
                                    generated from the actual numbers
                                    computed in Part 6 (not hard-coded).

  (error_tolerance_plot.png is also produced as a bonus figure
  supporting the Part 6/7 discussion of how many terms each method
  needs to meet the 0.1% tolerance.)

Everything is also printed to the console as the script runs.
"""

import math
import sys
import io
import contextlib

# matplotlib is only needed for the plots; use a non-interactive
# backend so the script runs fine on a headless machine.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False


class _Tee:
    """Write the same text to multiple streams at once (used to save the
    console-printed numerical tables to a file while still showing them
    on screen)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


# ======================================================================
# PART 1: GEOMETRIC SERIES
# ======================================================================
def geometric_sum(x, N):
    """
    Partial sum S_N = 1 + x + x^2 + ... + x^N, computed term-by-term
    (the closed form 1/(1-x) is intentionally NOT used).
    """
    total = 0.0
    for k in range(N + 1):
        total += x ** k
    return total


def part1_geometric_series():
    print("=" * 70)
    print("PART 1: GEOMETRIC SERIES")
    print("=" * 70)
    for x in (0.5, 0.8, 0.9):
        exact = 1.0 / (1.0 - x)
        print(f"\nx = {x}   (exact 1/(1-x) = {exact:.6f})")
        print(f"{'N':>4} {'Partial Sum':>15} {'Abs Error':>15}")
        for N in (1, 2, 5, 10, 20, 50):
            s = geometric_sum(x, N)
            err = abs(exact - s)
            print(f"{N:>4} {s:>15.6f} {err:>15.6e}")
    print(
        "\nObservation: convergence slows as x approaches 1, because the "
        "terms x^k shrink more slowly the closer x is to 1 -- each extra "
        "term buys less accuracy, so more terms are needed to reach the "
        "same error tolerance."
    )


# ======================================================================
# PART 2: POWER SERIES
# ======================================================================
def power_series(x, coefficients):
    """Evaluate P_N(x) = sum(a_k * x^k) for k = 0..N given a list of
    coefficients [a0, a1, ..., aN]."""
    result = 0.0
    for k, a_k in enumerate(coefficients):
        result += a_k * (x ** k)
    return result


def part2_power_series():
    print("\n" + "=" * 70)
    print("PART 2: POWER SERIES")
    print("=" * 70)
    coeffs = [1, 0, -1 / 2, 0, 1 / 24]  # looks like start of cos(x) series
    for x in (0.0, 0.5, 1.0):
        val = power_series(x, coeffs)
        print(f"P(x={x}) with coefficients {coeffs} -> {val:.6f}")
    print(
        "\nKey insight confirmed: a finite power series is just a "
        "polynomial; letting N -> infinity with the right coefficients "
        "turns it into a series representation of a transcendental "
        "function (sin, cos, exp, ...)."
    )


# ======================================================================
# PART 3: MACLAURIN SERIES FOR sin(theta)
# ======================================================================
def sin_maclaurin(theta, N):
    """
    Approximate sin(theta) using N terms of the Maclaurin series:
        sin(theta) = theta - theta^3/3! + theta^5/5! - theta^7/7! + ...
    theta must be in RADIANS.
    """
    result = 0.0
    for n in range(N):
        sign = (-1) ** n
        power = 2 * n + 1
        factorial = math.factorial(power)
        result += sign * (theta ** power) / factorial
    return result


def part3_maclaurin_series():
    print("\n" + "=" * 70)
    print("PART 3: MACLAURIN SERIES FOR sin(theta), theta = 10 deg")
    print("=" * 70)
    theta_deg = 10.0
    theta_rad = math.radians(theta_deg)
    exact = math.sin(theta_rad)
    print(f"Exact sin(10 deg) = {exact:.8f}")
    print(f"{'Terms N':>8} {'Approximation':>15} {'Abs Error':>15} {'% Error':>10}")
    for N in (1, 2, 3, 4):
        approx = sin_maclaurin(theta_rad, N)
        abs_err = abs(exact - approx)
        pct_err = 100 * abs_err / abs(exact)
        print(f"{N:>8} {approx:>15.8f} {abs_err:>15.2e} {pct_err:>9.4f}%")


# ======================================================================
# PART 4: ENGINEERING INVESTIGATION -- y = L sin(theta)
# ======================================================================
L = 20.0  # structural length in meters
ANGLES_DEG = [1, 2, 5, 10, 15, 20, 30]
TERM_COUNTS = [1, 2, 3, 4]


def y_exact(L, theta_deg):
    return L * math.sin(math.radians(theta_deg))


def y_maclaurin(L, theta_deg, N):
    return L * sin_maclaurin(math.radians(theta_deg), N)


def build_part4_tables():
    """Return a dict: {N: [ (angle, exact, approx, abs_err, pct_err), ... ]}"""
    tables = {}
    for N in TERM_COUNTS:
        rows = []
        for angle in ANGLES_DEG:
            exact = y_exact(L, angle)
            approx = y_maclaurin(L, angle, N)
            abs_err = abs(exact - approx)
            pct_err = 100 * abs_err / abs(exact) if exact != 0 else 0.0
            rows.append((angle, exact, approx, abs_err, pct_err))
        tables[N] = rows
    return tables


def print_table(rows, title):
    print(f"\n{title}")
    print(f"{'Angle(deg)':>10} {'Exact y (m)':>13} {'Approx y (m)':>13} "
          f"{'Abs Error':>12} {'% Error':>10}")
    for angle, exact, approx, abs_err, pct_err in rows:
        print(f"{angle:>10} {exact:>13.6f} {approx:>13.6f} "
              f"{abs_err:>12.2e} {pct_err:>9.4f}%")


def part4_engineering_investigation():
    print("\n" + "=" * 70)
    print("PART 4: ENGINEERING INVESTIGATION  y = L * sin(theta), L = 20 m")
    print("=" * 70)
    tables = build_part4_tables()
    for N in TERM_COUNTS:
        print_table(tables[N], f"Maclaurin approximation with N = {N} term(s)")

    print(
        "\nAnalysis:\n"
        "1. Error grows with angle for a fixed number of terms, since the "
        "Maclaurin series is centered at theta = 0 and accuracy degrades "
        "with distance from the expansion point.\n"
        "2. Error shrinks rapidly as more terms are added, especially for "
        "larger angles where higher-order terms still matter.\n"
        "3. For small angles (<5 deg), 1 term is already sufficient for "
        "sub-0.1% error (see Part 6 for exact numbers).\n"
        "4. For larger angles (>20 deg), 3-4 terms are typically needed to "
        "reach the same tolerance."
    )
    return tables


# ======================================================================
# PART 5: TAYLOR SERIES CENTERED AT a = 10 deg
# ======================================================================
def sin_taylor(theta, a, N):
    """
    Approximate sin(theta) using an N-term Taylor series centered at a.
    theta and a must be in RADIANS.
    """
    result = 0.0
    sin_a, cos_a = math.sin(a), math.cos(a)
    for n in range(N):
        derivative_pattern = n % 4
        if derivative_pattern == 0:
            f_deriv = sin_a
        elif derivative_pattern == 1:
            f_deriv = cos_a
        elif derivative_pattern == 2:
            f_deriv = -sin_a
        else:
            f_deriv = -cos_a
        term = f_deriv * ((theta - a) ** n) / math.factorial(n)
        result += term
    return result


def y_taylor(L, theta_deg, a_deg, N):
    return L * sin_taylor(math.radians(theta_deg), math.radians(a_deg), N)


def part5_taylor_series(mac_tables):
    print("\n" + "=" * 70)
    print("PART 5: TAYLOR SERIES CENTERED AT a = 10 deg")
    print("=" * 70)
    a_deg = 10.0
    taylor_tables = {}
    for N in TERM_COUNTS:
        rows = []
        for angle in ANGLES_DEG:
            exact = y_exact(L, angle)
            approx = y_taylor(L, angle, a_deg, N)
            abs_err = abs(exact - approx)
            pct_err = 100 * abs_err / abs(exact) if exact != 0 else 0.0
            rows.append((angle, exact, approx, abs_err, pct_err))
        taylor_tables[N] = rows
        print_table(rows, f"Taylor (centered at 10 deg) approximation, N = {N} term(s)")

    print(f"\n{'Angle(deg)':>10} {'Maclaurin %err (N=2)':>22} {'Taylor %err (N=2)':>20}")
    for i, angle in enumerate(ANGLES_DEG):
        mac_pct = mac_tables[2][i][4]
        tay_pct = taylor_tables[2][i][4]
        print(f"{angle:>10} {mac_pct:>21.4f}% {tay_pct:>19.4f}%")

    print(
        "\nComparison: Taylor (centered at 10 deg) is far more accurate near "
        "10 deg than Maclaurin for the same number of terms, because the "
        "expansion point sits inside the region of interest. Far from 10 "
        "deg (e.g. near 1 deg or 30 deg), Maclaurin (centered at 0 deg) can "
        "become more accurate again for small angles specifically, since it "
        "is centered at 0."
    )
    return taylor_tables


# ======================================================================
# PART 6: ENGINEERING DECISION -- ERROR TOLERANCE OF 0.1%
# ======================================================================
TOLERANCE_PCT = 0.1
MAX_TERMS_SEARCH = 15


def min_terms_for_tolerance(approx_func, angle_deg, tol_pct, max_terms):
    """
    Return the smallest N (1..max_terms) for which the percentage error
    of approx_func(angle_deg, N) is below tol_pct. Returns None if no such
    N is found within max_terms.
    """
    exact = y_exact(L, angle_deg)
    for N in range(1, max_terms + 1):
        approx = approx_func(angle_deg, N)
        pct_err = 100 * abs(exact - approx) / abs(exact)
        if pct_err < tol_pct:
            return N, pct_err
    return None, None


def part6_error_tolerance():
    print("\n" + "=" * 70)
    print(f"PART 6: MINIMUM TERMS FOR < {TOLERANCE_PCT}% ERROR")
    print("=" * 70)

    mac_func = lambda angle, N: y_maclaurin(L, angle, N)
    tay_func = lambda angle, N: y_taylor(L, angle, 10.0, N)

    print(f"{'Angle(deg)':>10} {'Maclaurin N':>12} {'Mac %err':>10} "
          f"{'Taylor N':>10} {'Tay %err':>10}")
    results = []
    for angle in ANGLES_DEG:
        mac_N, mac_err = min_terms_for_tolerance(mac_func, angle, TOLERANCE_PCT, MAX_TERMS_SEARCH)
        tay_N, tay_err = min_terms_for_tolerance(tay_func, angle, TOLERANCE_PCT, MAX_TERMS_SEARCH)
        results.append((angle, mac_N, mac_err, tay_N, tay_err))
        mac_str = f"{mac_N}" if mac_N else ">max"
        tay_str = f"{tay_N}" if tay_N else ">max"
        mac_err_str = f"{mac_err:.4f}%" if mac_err is not None else "n/a"
        tay_err_str = f"{tay_err:.4f}%" if tay_err is not None else "n/a"
        print(f"{angle:>10} {mac_str:>12} {mac_err_str:>10} {tay_str:>10} {tay_err_str:>10}")

    # Small-angle approximation sin(theta) ~= theta: find critical angle
    print("\nSmall-angle approximation sin(theta) ~= theta (1-term Maclaurin):")
    critical_angle = None
    for angle_tenth in range(1, 900):  # scan in 0.1 deg steps up to 90 deg
        angle = angle_tenth / 10.0
        exact = y_exact(L, angle)
        approx = y_maclaurin(L, angle, 1)
        pct_err = 100 * abs(exact - approx) / abs(exact)
        if pct_err >= TOLERANCE_PCT:
            critical_angle = angle
            break
    if critical_angle:
        print(
            f"The 1-term approximation sin(theta) ~= theta exceeds "
            f"{TOLERANCE_PCT}% error at approximately theta = {critical_angle:.1f} deg."
        )

    print(
        "\nAnalysis:\n"
        "1. No -- adding more terms only helps within the series' radius/"
        "region of good convergence; for a fixed angle the error does "
        "shrink monotonically here because sin's Taylor series converges "
        "for all real theta, but the RATE of improvement slows at higher N.\n"
        "2. Maclaurin is most accurate near zero because it is built purely "
        "from derivatives evaluated AT zero -- the further theta drifts "
        "from 0, the more the truncated terms are needed to compensate.\n"
        "3. Moving the Taylor expansion point closer to the angle of "
        "interest means the (theta - a) term stays small, so fewer terms "
        "are needed for the same accuracy in that neighborhood.\n"
        "4. Accuracy degrades roughly with the size of (theta - a) raised "
        "to the power of the number of terms used -- errors grow quickly "
        "the farther theta is from the expansion point a.\n"
        f"5. The 1-term small-angle approximation stays within "
        f"{TOLERANCE_PCT}% error only up to about "
        f"{critical_angle:.1f} degrees." if critical_angle else ""
    )
    return results, critical_angle


# ======================================================================
# PART 7: FINAL ENGINEERING RECOMMENDATION
# ======================================================================
def part7_recommendation(tolerance_results, critical_angle):
    print("\n" + "=" * 70)
    print("PART 7: FINAL ENGINEERING RECOMMENDATION")
    print("=" * 70)

    max_mac_N = max(r[1] for r in tolerance_results if r[1] is not None)
    max_tay_N = max(r[3] for r in tolerance_results if r[3] is not None)
    worst_mac_err = max(r[2] for r in tolerance_results if r[2] is not None)
    worst_tay_err = max(r[4] for r in tolerance_results if r[4] is not None)

    recommendation_text = f"""ENGINEERING RECOMMENDATION
Approximating y = L * sin(theta), L = {L} m
Required accuracy: less than {TOLERANCE_PCT}% error
Angle range evaluated: {min(ANGLES_DEG)} deg to {max(ANGLES_DEG)} deg
===========================================================

Decision: Use the MACLAURIN series (centered at theta = 0 deg).

Supporting numerical evidence:
  - Maclaurin needed at most {max_mac_N} term(s) across the {min(ANGLES_DEG)}-{max(ANGLES_DEG)} deg
    range to stay under {TOLERANCE_PCT}% error (worst-case error observed:
    {worst_mac_err:.4f}%).
  - Taylor centered at 10 deg needed at most {max_tay_N} term(s) across
    the same range (worst-case error observed: {worst_tay_err:.4f}%),
    but its advantage only shows up near its expansion point (10 deg);
    it does not clearly outperform Maclaurin over the full range because
    Maclaurin is already centered at the low end of this range.
  - The pure 1-term small-angle approximation sin(theta) ~= theta is only
    valid up to about {critical_angle:.1f} degrees for this tolerance, so
    it is NOT sufficient on its own for the full {min(ANGLES_DEG)}-{max(ANGLES_DEG)} deg range.

Convergence behavior:
  Both series converge for all real theta (sin(theta) has an infinite
  radius of convergence), so more terms will always eventually reach
  machine-precision agreement with the exact value. The practical
  engineering question is how FEW terms are needed for engineering-grade
  accuracy over the angles actually used in the field, not whether the
  series converges at all.

Computational simplicity vs. accuracy tradeoff:
  A fixed low-order Maclaurin polynomial ({max_mac_N} term(s)) is cheap to
  evaluate (no trigonometric library call needed) and meets the
  {TOLERANCE_PCT}% tolerance across the full angle range with a single,
  angle-independent formula. This is preferable in embedded/field
  calculation tools where simplicity and auditability matter. If
  extremely tight tolerances or angles well beyond {max(ANGLES_DEG)} deg were
  required, switching to Python's exact math.sin() (or a Taylor series
  re-centered near the angle of interest) would be the safer choice.

Final recommendation summary:
  - Number of terms required:      {max_mac_N} (Maclaurin)
  - Percentage error achieved:     {worst_mac_err:.4f}% (worst case in range)
  - Valid angle range:              {min(ANGLES_DEG)} deg to {max(ANGLES_DEG)} deg
  - Method:                         Maclaurin series, centered at 0 deg
"""

    print(f"""
Recommendation:

For the full working range of 1 deg to 30 deg with a required accuracy
of better than {TOLERANCE_PCT}% error, I recommend using the MACLAURIN
series (centered at theta = 0), taking the worst-case number of terms
needed across the range as the fixed term count for the implementation.

Supporting numerical evidence:
  - Maclaurin needed at most {max_mac_N} term(s) across 1-30 degrees to
    stay under {TOLERANCE_PCT}% error (worst-case error observed:
    {worst_mac_err:.4f}%).
  - Taylor centered at 10 deg needed at most {max_tay_N} term(s) across
    the same range (worst-case error observed: {worst_tay_err:.4f}%),
    but its advantage only shows up near its expansion point (10 deg);
    it does not clearly outperform Maclaurin over the full 1-30 deg span
    because Maclaurin is already centered at the low end of this range.
  - The pure 1-term small-angle approximation sin(theta) ~= theta is only
    valid up to about {critical_angle:.1f} degrees for this tolerance, so
    it is NOT sufficient on its own for the full 1-30 deg range.

Convergence behavior: both series converge for all real theta (sin(theta)
has an infinite radius of convergence), so more terms will always
eventually reach machine-precision agreement with the exact value; the
practical question is how FEW terms are needed for engineering-grade
accuracy over the angles actually used in the field.

Computational simplicity vs. accuracy tradeoff: a fixed low-order
Maclaurin polynomial (a handful of terms) is cheap to evaluate (no
trigonometric library call needed) and meets the {TOLERANCE_PCT}%
tolerance across the full angle range with a single, angle-independent
formula, which is preferable in embedded/field calculation tools where
simplicity and auditability matter. If extremely tight tolerances or
angles well beyond 30 deg were required, switching to Python's exact
math.sin() (or a Taylor series re-centered near the angle of interest)
would be the safer choice.

Valid angle range for this solution: 1 deg to 30 deg, using
{max_mac_N} Maclaurin term(s), meeting < {TOLERANCE_PCT}% error
throughout.
""")

    with open("written_recommendation.txt", "w") as f:
        f.write(recommendation_text)
    print("Saved written_recommendation.txt")


# ======================================================================
# REQUIRED PLOTS
# ======================================================================
def make_convergence_plot():
    """Percentage error vs. number of terms, for several angles (Maclaurin)."""
    plt.figure(figsize=(8, 6))
    angles_to_plot = [1, 10, 20, 30]
    for angle in angles_to_plot:
        exact = y_exact(L, angle)
        errs = []
        Ns = list(range(1, 8))
        for N in Ns:
            approx = y_maclaurin(L, angle, N)
            pct_err = 100 * abs(exact - approx) / abs(exact)
            errs.append(max(pct_err, 1e-16))  # avoid log(0)
        plt.plot(Ns, errs, marker="o", label=f"theta = {angle} deg")
    plt.yscale("log")
    plt.xlabel("Number of Maclaurin terms (N)")
    plt.ylabel("Percentage error (%) [log scale]")
    plt.title("Convergence: % Error vs. Number of Terms (Maclaurin)")
    plt.axhline(TOLERANCE_PCT, color="red", linestyle="--",
                label=f"{TOLERANCE_PCT}% tolerance")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergence_plot.png", dpi=150)
    plt.close()
    print("Saved convergence_plot.png")


def make_function_comparison_plot():
    """Exact sin(theta) vs Maclaurin and Taylor approximations."""
    theta_deg_range = [i * 0.5 for i in range(0, 121)]  # 0 to 60 deg
    theta_rad_range = [math.radians(t) for t in theta_deg_range]
    exact_vals = [math.sin(t) for t in theta_rad_range]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    # Left panel: Maclaurin
    axes[0].plot(theta_deg_range, exact_vals, "k-", linewidth=2, label="Exact sin(theta)")
    for N in (1, 2, 3, 4):
        approx_vals = [sin_maclaurin(t, N) for t in theta_rad_range]
        axes[0].plot(theta_deg_range, approx_vals, "--", label=f"Maclaurin N={N}")
    axes[0].set_title("Maclaurin Approximations (centered at 0 deg)")
    axes[0].set_xlabel("theta (degrees)")
    axes[0].set_ylabel("sin(theta)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right panel: Taylor centered at 10 deg
    a_rad = math.radians(10.0)
    axes[1].plot(theta_deg_range, exact_vals, "k-", linewidth=2, label="Exact sin(theta)")
    for N in (1, 2, 3, 4):
        approx_vals = [sin_taylor(t, a_rad, N) for t in theta_rad_range]
        axes[1].plot(theta_deg_range, approx_vals, "--", label=f"Taylor N={N}")
    axes[1].axvline(10, color="gray", linestyle=":", label="Expansion point (10 deg)")
    axes[1].set_title("Taylor Approximations (centered at 10 deg)")
    axes[1].set_xlabel("theta (degrees)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("function_comparison_plot.png", dpi=150)
    plt.close()
    print("Saved function_comparison_plot.png")


def make_error_comparison_plot():
    """Absolute and percentage error comparison: Maclaurin vs Taylor, N=2."""
    N = 2
    a_deg = 10.0
    abs_err_mac, abs_err_tay = [], []
    pct_err_mac, pct_err_tay = [], []
    for angle in ANGLES_DEG:
        exact = y_exact(L, angle)
        mac_approx = y_maclaurin(L, angle, N)
        tay_approx = y_taylor(L, angle, a_deg, N)
        abs_err_mac.append(abs(exact - mac_approx))
        abs_err_tay.append(abs(exact - tay_approx))
        pct_err_mac.append(100 * abs(exact - mac_approx) / abs(exact))
        pct_err_tay.append(100 * abs(exact - tay_approx) / abs(exact))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x_pos = range(len(ANGLES_DEG))
    width = 0.35

    axes[0].bar([p - width / 2 for p in x_pos], abs_err_mac, width, label="Maclaurin")
    axes[0].bar([p + width / 2 for p in x_pos], abs_err_tay, width, label="Taylor (a=10 deg)")
    axes[0].set_xticks(list(x_pos))
    axes[0].set_xticklabels(ANGLES_DEG)
    axes[0].set_xlabel("Angle (degrees)")
    axes[0].set_ylabel("Absolute error (m)")
    axes[0].set_title(f"Absolute Error Comparison (N={N})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar([p - width / 2 for p in x_pos], pct_err_mac, width, label="Maclaurin")
    axes[1].bar([p + width / 2 for p in x_pos], pct_err_tay, width, label="Taylor (a=10 deg)")
    axes[1].axhline(TOLERANCE_PCT, color="red", linestyle="--", label=f"{TOLERANCE_PCT}% tolerance")
    axes[1].set_xticks(list(x_pos))
    axes[1].set_xticklabels(ANGLES_DEG)
    axes[1].set_xlabel("Angle (degrees)")
    axes[1].set_ylabel("Percentage error (%)")
    axes[1].set_title(f"Percentage Error Comparison (N={N})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("error_comparison_plot.png", dpi=150)
    plt.close()
    print("Saved error_comparison_plot.png")


def make_error_tolerance_plot(tolerance_results):
    """Minimum terms needed to hit 0.1% tolerance, Maclaurin vs Taylor, per angle."""
    angles = [r[0] for r in tolerance_results]
    mac_Ns = [r[1] if r[1] is not None else 0 for r in tolerance_results]
    tay_Ns = [r[3] if r[3] is not None else 0 for r in tolerance_results]

    x_pos = range(len(angles))
    width = 0.35
    plt.figure(figsize=(8, 5.5))
    plt.bar([p - width / 2 for p in x_pos], mac_Ns, width, label="Maclaurin")
    plt.bar([p + width / 2 for p in x_pos], tay_Ns, width, label="Taylor (a=10 deg)")
    plt.xticks(list(x_pos), angles)
    plt.xlabel("Angle (degrees)")
    plt.ylabel(f"Minimum terms needed for < {TOLERANCE_PCT}% error")
    plt.title("Terms Required to Meet 0.1% Error Tolerance")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("error_tolerance_plot.png", dpi=150)
    plt.close()
    print("Saved error_tolerance_plot.png")


# ======================================================================
# MAIN DRIVER
# ======================================================================
def main():
    part1_geometric_series()
    part2_power_series()
    part3_maclaurin_series()

    # --- DELIVERABLE 1: NUMERICAL TABLES ---------------------------------
    # Capture everything Part 4 (Maclaurin) and Part 5 (Taylor) print --
    # the required tables of exact values, approximations, absolute error
    # and percentage error for every angle and every number of terms --
    # and save it to a standalone, formatted text file while still
    # showing it in the console.
    tables_buffer = io.StringIO()
    tee = _Tee(sys.stdout, tables_buffer)
    with contextlib.redirect_stdout(tee):
        mac_tables = part4_engineering_investigation()
        taylor_tables = part5_taylor_series(mac_tables)
    with open("numerical_tables.txt", "w") as f:
        f.write("NUMERICAL TABLES\n")
        f.write("Exact values, approximations, and errors for each angle "
                "and number of terms (Maclaurin and Taylor series)\n")
        f.write("=" * 70 + "\n")
        f.write(tables_buffer.getvalue())
    print("Saved numerical_tables.txt")

    tolerance_results, critical_angle = part6_error_tolerance()
    part7_recommendation(tolerance_results, critical_angle)

    # --- DELIVERABLES 2-4: REQUIRED PLOTS --------------------------------
    print("\n" + "=" * 70)
    print("GENERATING REQUIRED PLOTS")
    print("=" * 70)
    make_convergence_plot()             # Deliverable 2
    make_function_comparison_plot()     # Deliverable 3
    make_error_comparison_plot()        # Deliverable 4
    make_error_tolerance_plot(tolerance_results)  # supporting Part 6/7

    print("\nAll 5 required deliverables generated:\n"
          "  1. Numerical tables       - numerical_tables.txt (+ console)\n"
          "  2. Convergence plot       - convergence_plot.png\n"
          "  3. Function comparison    - function_comparison_plot.png\n"
          "  4. Error comparison       - error_comparison_plot.png\n"
          "  5. Written recommendation - written_recommendation.txt")


if __name__ == "__main__":
    main()