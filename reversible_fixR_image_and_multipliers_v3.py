
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt


# ============================================================
# PARAMETERS: CHANGE ONLY THIS BLOCK
# ============================================================

# Poincare map P = shift by one forcing period T = 2*pi/3.
# We build P^K(FixR), where FixR = {x=0} for R(x,y)=(-x,y).
K = 3
P1 = 1.0
P2 = 2.44

# Window on FixR = {x=0, y in [Y_MIN, Y_MAX]}
Y_MIN = -20.0
Y_MAX = 20.0

# Phase of the section t = T0 mod T
T0 = 0.0

# Start grid and adaptive refinement
N0 = 61
MAX_DEPTH = 12
MIN_PRE_STEP = 1e-4
MAX_CHORD = 0.08
MAX_DEV = 0.015
MAX_ANGLE = 0.20  # radians

# Root search
ROOT_TOL = 1e-11
HIT_TOL = 1e-9
ROOT_MERGE_TOL = 1e-7

# Period detection
PERIOD_TOL = 1e-7

# Integrator tolerances
RTOL = 1e-9
ATOL = 1e-11

# Plot / output
RESAMPLE_DS = 0.01
SHOW_PLOTS = True
SAVE_PLOTS = False
OUTPUT_DIR = "."


# ============================================================
# MODEL
# x' = y
# y' = -x - x^3 + p1*x*y + p2*sin(3t)
# Involution R(x,y)=(-x,y), hence FixR = {x=0}.
# ============================================================

T = 2.0 * math.pi / 3.0


@dataclass
class SymmetricPoint:
    y0: float
    z0: np.ndarray
    zk: np.ndarray
    minimal_period: int
    multipliers: np.ndarray
    point_type: str


# Small cache for repeated calls to P^K(0,y0)
_map_cache: Dict[Tuple[int, float, float, float, float], np.ndarray] = {}


def rhs(t: float, z: np.ndarray, p1: float, p2: float) -> np.ndarray:
    x, y = z
    return np.array([
        y,
        -x - x**3 + p1 * x * y + p2 * math.sin(3.0 * t),
    ], dtype=float)


def rhs_with_variational(t: float, Z: np.ndarray, p1: float, p2: float) -> np.ndarray:
    x, y = Z[0], Z[1]
    A = np.array([
        [0.0, 1.0],
        [-1.0 - 3.0 * x * x + p1 * y, p1 * x]
    ], dtype=float)
    Phi = Z[2:].reshape(2, 2)
    dPhi = A @ Phi

    return np.array([
        y,
        -x - x**3 + p1 * x * y + p2 * math.sin(3.0 * t),
        dPhi[0, 0], dPhi[0, 1], dPhi[1, 0], dPhi[1, 1]
    ], dtype=float)


def flow(z0: np.ndarray, p1: float, p2: float, periods: int,
         t0: float = T0, with_jacobian: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    tau = periods * T

    if not with_jacobian:
        sol = solve_ivp(
            lambda t, z: rhs(t, z, p1, p2),
            (t0, t0 + tau),
            z0,
            method="DOP853",
            rtol=RTOL,
            atol=ATOL
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y[:, -1].copy(), None

    Z0 = np.array([z0[0], z0[1], 1.0, 0.0, 0.0, 1.0], dtype=float)
    sol = solve_ivp(
        lambda t, Z: rhs_with_variational(t, Z, p1, p2),
        (t0, t0 + tau),
        Z0,
        method="DOP853",
        rtol=RTOL,
        atol=ATOL
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    Zf = sol.y[:, -1]
    zf = Zf[:2].copy()
    Phi = Zf[2:].reshape(2, 2).copy()
    return zf, Phi


def flow_with_checkpoints(z0: np.ndarray, p1: float, p2: float, checkpoints: List[int],
                          t0: float = T0):
    """
    Single integration up to max(checkpoints)*T with t_eval at all checkpoint times.
    Returns dictionaries: states[d], jacobians[d].
    """
    checkpoints = sorted(set(checkpoints))
    max_period = max(checkpoints)
    t_eval = [t0 + d * T for d in checkpoints]

    Z0 = np.array([z0[0], z0[1], 1.0, 0.0, 0.0, 1.0], dtype=float)
    sol = solve_ivp(
        lambda t, Z: rhs_with_variational(t, Z, p1, p2),
        (t0, t0 + max_period * T),
        Z0,
        t_eval=t_eval,
        method="DOP853",
        rtol=RTOL,
        atol=ATOL
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    states = {}
    jacobians = {}
    for idx, d in enumerate(checkpoints):
        Zf = sol.y[:, idx]
        states[d] = Zf[:2].copy()
        jacobians[d] = Zf[2:].reshape(2, 2).copy()

    return states, jacobians


def poincare_k_on_fixR(y0: float, k: int, p1: float, p2: float) -> np.ndarray:
    key = (k, round(float(y0), 14), round(float(p1), 14), round(float(p2), 14), round(float(T0), 14))
    if key in _map_cache:
        return _map_cache[key].copy()

    z0 = np.array([0.0, y0], dtype=float)
    zf, _ = flow(z0, p1, p2, periods=k, with_jacobian=False)
    _map_cache[key] = zf.copy()
    return zf


def x_after_k(y0: float, k: int, p1: float, p2: float) -> float:
    return poincare_k_on_fixR(y0, k, p1, p2)[0]


def divisors(n: int) -> List[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def point_segment_distance(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    AB = B - A
    denom = np.dot(AB, AB)
    if denom < 1e-30:
        return float(np.linalg.norm(P - A))
    s = np.dot(P - A, AB) / denom
    s = max(0.0, min(1.0, s))
    Q = A + s * AB
    return float(np.linalg.norm(P - Q))


def turning_angle(A: np.ndarray, M: np.ndarray, B: np.ndarray) -> float:
    u = M - A
    v = B - M
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-30 or nv < 1e-30:
        return 0.0
    c = np.dot(u, v) / (nu * nv)
    c = max(-1.0, min(1.0, c))
    return float(math.acos(c))


def refine_interval(a: float, Fa: np.ndarray, b: float, Fb: np.ndarray,
                    depth: int, k: int, p1: float, p2: float) -> Tuple[np.ndarray, np.ndarray]:
    m = 0.5 * (a + b)
    Fm = poincare_k_on_fixR(m, k, p1, p2)

    chord = float(np.linalg.norm(Fb - Fa))
    dev = point_segment_distance(Fm, Fa, Fb)
    ang = turning_angle(Fa, Fm, Fb)

    need_refine = (
        depth < MAX_DEPTH
        and abs(b - a) > MIN_PRE_STEP
        and (chord > MAX_CHORD or dev > MAX_DEV or ang > MAX_ANGLE)
    )

    if not need_refine:
        return np.array([a, b]), np.vstack([Fa, Fb])

    preL, imgL = refine_interval(a, Fa, m, Fm, depth + 1, k, p1, p2)
    preR, imgR = refine_interval(m, Fm, b, Fb, depth + 1, k, p1, p2)

    pre = np.concatenate([preL, preR[1:]])
    img = np.vstack([imgL, imgR[1:]])
    return pre, img


def build_image(k: int, p1: float, p2: float, ymin: float, ymax: float) -> Tuple[np.ndarray, np.ndarray]:
    y_grid = np.linspace(ymin, ymax, N0)
    F_grid = np.array([poincare_k_on_fixR(y0, k, p1, p2) for y0 in y_grid])

    pre_all = [y_grid[0]]
    img_all = [F_grid[0]]

    for i in range(len(y_grid) - 1):
        pre_i, img_i = refine_interval(y_grid[i], F_grid[i], y_grid[i + 1], F_grid[i + 1],
                                       0, k, p1, p2)
        pre_all.extend(pre_i[1:])
        img_all.extend(img_i[1:])

    pre = np.array(pre_all, dtype=float)
    img = np.array(img_all, dtype=float)
    return pre, img


def resample_curve(curve: np.ndarray, pre: np.ndarray, ds: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(curve) < 2 or ds <= 0.0:
        return curve.copy(), pre.copy()

    seg = np.diff(curve, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])
    if s[-1] <= 0:
        return curve.copy(), pre.copy()

    s_new = np.arange(0.0, s[-1], ds)
    if len(s_new) == 0 or s_new[-1] < s[-1]:
        s_new = np.append(s_new, s[-1])

    x_new = np.interp(s_new, s, curve[:, 0])
    y_new = np.interp(s_new, s, curve[:, 1])
    pre_new = np.interp(s_new, s, pre)
    return np.column_stack([x_new, y_new]), pre_new


def find_symmetric_roots(pre: np.ndarray, img: np.ndarray, k: int, p1: float, p2: float) -> List[float]:
    roots = []

    for j in range(len(pre) - 1):
        x1 = img[j, 0]
        x2 = img[j + 1, 0]

        if abs(x1) < HIT_TOL:
            roots.append(pre[j])

        if x1 * x2 < 0.0:
            a, b = pre[j], pre[j + 1]
            y0 = brentq(lambda yy: x_after_k(yy, k, p1, p2), a, b, xtol=ROOT_TOL, rtol=ROOT_TOL)
            roots.append(y0)

    roots.sort()
    uniq = []
    for r in roots:
        if not uniq or abs(r - uniq[-1]) > ROOT_MERGE_TOL:
            uniq.append(r)
    return uniq


def classify_type(eigs: np.ndarray) -> str:
    eigs = np.asarray(eigs)
    mod = np.abs(eigs)
    tr = float(np.real_if_close(np.sum(eigs)))

    if np.allclose(np.imag(eigs), 0.0, atol=1e-8):
        if np.all(mod < 1.0 - 1e-7):
            return "sink"
        if np.all(mod > 1.0 + 1e-7):
            return "source"
        if (mod[0] < 1.0 - 1e-7 and mod[1] > 1.0 + 1e-7) or (mod[1] < 1.0 - 1e-7 and mod[0] > 1.0 + 1e-7):
            return "saddle"
        return "parabolic / nearly neutral"

    if abs(tr) < 2.0 + 1e-6 and np.all(np.abs(mod - 1.0) < 1e-5):
        return "elliptic"
    return "complex non-hyperbolic / focus-like"


def minimal_period_and_multipliers(y0: float, k: int, p1: float, p2: float):
    z0 = np.array([0.0, y0], dtype=float)
    full_period = 2 * k
    divs = divisors(full_period)

    states, jacobians = flow_with_checkpoints(z0, p1, p2, divs)

    min_period = None
    Phi_min = None
    for d in divs:
        if np.linalg.norm(states[d] - z0) < PERIOD_TOL:
            min_period = d
            Phi_min = jacobians[d]
            break

    if min_period is None:
        min_period = full_period
        Phi_min = jacobians[full_period]

    eigs = np.linalg.eigvals(Phi_min)
    typ = classify_type(eigs)

    return min_period, eigs, typ


def collect_symmetric_points(k: int, p1: float, p2: float, ymin: float, ymax: float):
    pre, img = build_image(k, p1, p2, ymin, ymax)
    roots = find_symmetric_roots(pre, img, k, p1, p2)

    points = []
    for y0 in roots:
        z0 = np.array([0.0, y0], dtype=float)
        zk = poincare_k_on_fixR(y0, k, p1, p2)
        m, eigs, typ = minimal_period_and_multipliers(y0, k, p1, p2)
        points.append(SymmetricPoint(
            y0=float(y0),
            z0=z0,
            zk=zk,
            minimal_period=m,
            multipliers=np.asarray(eigs),
            point_type=typ,
        ))
    return pre, img, points


def fmt_num(x: float) -> str:
    xr = float(np.real_if_close(x))
    if abs(xr) < 0.000005:
        xr = 0.0
    return f"{xr:.5f}"


def fmt_complex(z: complex) -> str:
    a = float(np.real(z))
    b = float(np.imag(z))
    if abs(a) < 0.000005:
        a = 0.0
    if abs(b) < 0.000005:
        b = 0.0
    if b == 0.0:
        return f"{a:.5f}"
    sign = "+" if b >= 0 else "-"
    return f"{a:.5f}{sign}{abs(b):.5f}i"


def save_report(points: List[SymmetricPoint], filename: str) -> None:
    lines = []
    lines.append(f"K = {K}, p1 = {fmt_num(P1)}, p2 = {fmt_num(P2)}")
    lines.append(f"FixR = {{x=0}}, y-window = [{fmt_num(Y_MIN)}, {fmt_num(Y_MAX)}]")
    lines.append("")
    lines.append(f"Number of symmetric points found: {len(points)}")
    lines.append("")

    for i, pt in enumerate(points, 1):
        eig1, eig2 = pt.multipliers[0], pt.multipliers[1]
        lines.append(f"[{i}] y0 = {fmt_num(pt.y0)}")
        lines.append(f"    z0 (point on FixR) = ({fmt_num(pt.z0[0])}, {fmt_num(pt.z0[1])})")
        lines.append(f"    P^{K}(z0)          = ({fmt_num(pt.zk[0])}, {fmt_num(pt.zk[1])})")
        lines.append(f"    minimal period w.r.t. P = {pt.minimal_period}")
        lines.append(f"    multipliers of DP^{pt.minimal_period}: {fmt_complex(eig1)}, {fmt_complex(eig2)}")
        lines.append(f"    type = {pt.point_type}")
        lines.append("")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_results(pre: np.ndarray, img: np.ndarray, points: List[SymmetricPoint],
                 plot_file_image: str) -> None:
    img_plot, _ = resample_curve(img, pre, RESAMPLE_DS)

    plt.figure(figsize=(7.5, 6.0))
    plt.plot(img_plot[:, 0], img_plot[:, 1], linewidth=1.5, label=rf"$P^{{{K}}}(\mathrm{{Fix}}R)$")
    plt.axvline(0.0, linestyle="--", linewidth=1.0, label=rf"$\mathrm{{Fix}}R=\{{x=0\}}$")
    if points:
        ys = [pt.zk[1] for pt in points]
        plt.plot(np.zeros(len(ys)), ys, "o", markersize=6, label="symmetric points")
    plt.xlabel("x after K periods")
    plt.ylabel("y after K periods")
    plt.title(f"Image of FixR under P^{K}   (p1={fmt_num(P1)}, p2={fmt_num(P2)})")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(plot_file_image, dpi=180, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


def print_summary(points: List[SymmetricPoint]) -> None:
    print("=" * 80)
    print(f"Image of FixR under P^{K},   p1={fmt_num(P1)}, p2={fmt_num(P2)}")
    print(f"Window in y: [{fmt_num(Y_MIN)}, {fmt_num(Y_MAX)}]")
    print(f"Symmetric points found: {len(points)}")
    print("=" * 80)

    for i, pt in enumerate(points, 1):
        print(f"[{i}] y0 = {fmt_num(pt.y0)}")
        print(f"    z0 (point on FixR) = ({fmt_num(pt.z0[0])}, {fmt_num(pt.z0[1])})")
        print(f"    P^{K}(z0)          = ({fmt_num(pt.zk[0])}, {fmt_num(pt.zk[1])})")
        print(f"    minimal period = {pt.minimal_period}")
        print(f"    multipliers of DP^{pt.minimal_period} = {fmt_complex(pt.multipliers[0])}, {fmt_complex(pt.multipliers[1])}")
        print(f"    type = {pt.point_type}")
        print("")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pre, img, points = collect_symmetric_points(K, P1, P2, Y_MIN, Y_MAX)
    print_summary(points)

    report_file = os.path.join(OUTPUT_DIR, "symmetric_points_report.txt")
    plot_file_image = os.path.join(OUTPUT_DIR, "image_of_fixR.png")

    save_report(points, report_file)
    plot_results(pre, img, points, plot_file_image)

    print(f"Saved report: {report_file}")
    if SAVE_PLOTS:
        print(f"Saved plot:   {plot_file_image}")


if __name__ == "__main__":
    main()
