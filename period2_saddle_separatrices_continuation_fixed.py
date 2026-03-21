
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
#  Duffing:
#      x'' + x + x^3 = p1*x*x' + p2*sin(3t)
#
#  Poincare map P for minimal forcing period T = 2*pi/3.
#
#  This script builds separatrices of a PERIOD-2 saddle orbit of P.
#
#  IMPORTANT:
#    - Local invariant manifolds are built for P^2.
#    - Then their image under P is added, so the final picture is
#      the full separatrix picture of the 2-cycle for P.
#
#  Colors:
#    unstable -> black
#    stable   -> red
#
#  This version uses PARAMETER CONTINUATION by default from the known
#  branch point:
#      p2 = 0.736, z = (1.036, -3.375)
# ============================================================

# ===================== USER PARAMETERS ======================

p1 = 1.0
p2 = 1.0                 # target parameter

t0 = 0.0
T = 2.0 * np.pi / 3.0

# --- branch continuation for the target period-2 saddle -----
USE_CONTINUATION = True

# Known point on the branch of the symmetric period-2 saddle:
p2_start = 0.736
z_start  = np.array([0.0, 1.036688204], float)

# Alternative right-end anchor if needed:
# p2_start = 5.588
# z_start  = np.array([-1.295, -5.33], float)

continuation_steps = 60   # increase if continuation becomes unstable

# Manual Newton guess is only used if USE_CONTINUATION = False
x_guess = 0.0
y_guess = 1.036688204

# --- separatrix drawing -------------------------------------
K = 200
smin = 2e-7
smax = 6e-5
N0 = 500
rounds = 6
segmax = 0.025
max_points = 60000

# map accuracy
nsteps = 600
rtol = 1e-10
atol = 1e-12

# stable branches:
#   "backward" -> true backward iteration for P^2
#   "reflect"  -> reflect unstable picture by G(x,y)=(-x,y)
STABLE_MODE = "reflect"

# plotting
xlim = (-4.5, 4.5)
ylim = (-8.0, 4.0)
jump_cut = 1.0
show_second_point = True

savefig = False
outfile = f"period2_saddle_separatrices_cont_p2_{p2:.5f}.png"

# ============================================================


def rhs(t, z, p2):
    x, y = z
    return np.array([y, -x - x**3 + p1 * x * y + p2 * np.sin(3.0 * t)], float)


def integrate_map(z0, p2, with_variational=False, rtol=1e-10, atol=1e-12):
    if not with_variational:
        sol = solve_ivp(
            lambda t, z: rhs(t, z, p2),
            (t0, t0 + T),
            np.asarray(z0, float),
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )
        return sol.y[:, -1]

    x0, y0 = map(float, z0)
    Y0 = np.array([x0, y0, 1.0, 0.0, 0.0, 1.0], float)

    def f(t, Y):
        x, y, J11, J12, J21, J22 = Y
        dx = y
        dy = -x - x**3 + p1 * x * y + p2 * np.sin(3.0 * t)
        a21 = -1.0 - 3.0 * x * x + p1 * y
        a22 = p1 * x
        dJ11 = J21
        dJ12 = J22
        dJ21 = a21 * J11 + a22 * J21
        dJ22 = a21 * J12 + a22 * J22
        return np.array([dx, dy, dJ11, dJ12, dJ21, dJ22], float)

    sol = solve_ivp(
        f, (t0, t0 + T), Y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    xf, yf, J11, J12, J21, J22 = sol.y[:, -1]
    DP = np.array([[J11, J12], [J21, J22]], float)
    return np.array([xf, yf], float), DP


def integrate_map2(z0, p2, rtol=1e-10, atol=1e-12):
    z1, DP0 = integrate_map(z0, p2, with_variational=True, rtol=rtol, atol=atol)
    z2, DP1 = integrate_map(z1, p2, with_variational=True, rtol=rtol, atol=atol)
    DP2 = DP1 @ DP0
    return z2, DP2, z1


def period2_newton(z_init, p2, maxit=16, tol=1e-12):
    z = np.asarray(z_init, float).copy()

    for _ in range(maxit):
        z2, DP2, z1 = integrate_map2(z, p2, rtol=rtol, atol=atol)
        F = z2 - z
        nF = float(np.linalg.norm(F))
        if nF < tol:
            return z, z1, DP2, nF

        A = DP2 - np.eye(2)
        dz = np.linalg.lstsq(A, -F, rcond=None)[0]

        alpha = 1.0
        improved = False
        for _ in range(10):
            z_try = z + alpha * dz
            z_mid = integrate_map(
                z_try, p2, with_variational=False, rtol=2 * rtol, atol=2 * atol
            )
            z2_try = integrate_map(
                z_mid, p2, with_variational=False, rtol=2 * rtol, atol=2 * atol
            )
            F_try = z2_try - z_try
            if np.linalg.norm(F_try) < (1.0 - 1e-4 * alpha) * nF:
                z = z_try
                improved = True
                break
            alpha *= 0.5

        if not improved:
            z = z + dz

    z2, DP2, z1 = integrate_map2(z, p2, rtol=rtol, atol=atol)
    return z, z1, DP2, float(np.linalg.norm(z2 - z))


def continue_period2_branch(p2_start, z_start, p2_target, steps=60):
    if steps < 2:
        steps = 2
    path = np.linspace(p2_start, p2_target, steps)
    z = np.asarray(z_start, float).copy()
    last_z1 = None
    last_DP2 = None
    last_resid = None
    for p in path:
        z, z1, DP2, resid = period2_newton(z, p)
        last_z1 = z1
        last_DP2 = DP2
        last_resid = resid
    return z, last_z1, last_DP2, last_resid


def find_period2_saddle(p2):
    if USE_CONTINUATION:
        z0, z1, DP2, resid = continue_period2_branch(
            p2_start, z_start, p2, steps=continuation_steps
        )
    else:
        z0, z1, DP2, resid = period2_newton([x_guess, y_guess], p2)

    z1 = integrate_map(z0, p2, with_variational=False, rtol=rtol, atol=atol)
    per1_err = float(np.linalg.norm(z1 - z0))

    eigvals, eigvecs = np.linalg.eig(DP2)
    iu = int(np.argmax(np.abs(eigvals)))
    is_ = int(np.argmin(np.abs(eigvals)))

    vu = np.real(eigvecs[:, iu])
    vs = np.real(eigvecs[:, is_])
    vu /= np.linalg.norm(vu)
    vs /= np.linalg.norm(vs)

    if vu[1] < 0:
        vu = -vu
    if vs[1] < 0:
        vs = -vs

    return z0, z1, DP2, eigvals, vu, vs, resid, per1_err


def rk4_map_batch(X, Y, p2, nsteps=600, direction=+1):
    X = np.asarray(X, float).copy()
    Y = np.asarray(Y, float).copy()
    dt = direction * T / nsteps
    half = 0.5 * dt
    t = t0

    for _ in range(nsteps):
        a1 = -X - X**3 + p1 * X * Y + p2 * np.sin(3.0 * t)
        k1x, k1y = Y, a1

        x2 = X + half * k1x
        y2 = Y + half * k1y
        th = t + half
        a2 = -x2 - x2**3 + p1 * x2 * y2 + p2 * np.sin(3.0 * th)
        k2x, k2y = y2, a2

        x3 = X + half * k2x
        y3 = Y + half * k2y
        a3 = -x3 - x3**3 + p1 * x3 * y3 + p2 * np.sin(3.0 * th)
        k3x, k3y = y3, a3

        x4 = X + dt * k3x
        y4 = Y + dt * k3y
        td = t + dt
        a4 = -x4 - x4**3 + p1 * x4 * y4 + p2 * np.sin(3.0 * td)
        k4x, k4y = y4, a4

        X += (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        Y += (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        t = td

    return X, Y


def iterates_for_s_P2(zs, vec, p2, s, K=8, direction=+1, nsteps=600):
    X = zs[0] + s * vec[0]
    Y = zs[1] + s * vec[1]
    out = []
    for _ in range(K):
        X, Y = rk4_map_batch(X, Y, p2, nsteps=nsteps, direction=direction)
        X, Y = rk4_map_batch(X, Y, p2, nsteps=nsteps, direction=direction)
        out.append((X.copy(), Y.copy()))
    return out


def adaptive_branch_P2(
    zs, vec, p2, direction, K=8,
    smin=2e-7, smax=6e-5, N0=500,
    segmax=0.025, rounds=6,
    max_points=60000, nsteps=600
):
    s = np.linspace(smin, smax, N0)
    its = None

    for _ in range(rounds):
        its = iterates_for_s_P2(
            zs, vec, p2, s, K=K, direction=direction, nsteps=nsteps
        )
        need = np.zeros(len(s) - 1, dtype=bool)

        for X, Y in its:
            seg = np.hypot(np.diff(X), np.diff(Y))
            need |= (seg > segmax)

        if not np.any(need):
            return s, its

        mids = 0.5 * (s[:-1] + s[1:])
        s_new = np.sort(np.unique(np.concatenate([s, mids[need]])))

        if len(s_new) > max_points:
            print(f"[warn] max_points reached: {len(s_new)}")
            return s, its

        s = s_new

    return s, its


def reflect_G(X, Y):
    return -X, Y


def plot_curve_segments(ax, X, Y, color="k", lw=1.0, jump_cut=1.0):
    jump = np.hypot(np.diff(X), np.diff(Y))
    cuts = np.where(jump > jump_cut)[0] + 1
    idx = np.concatenate([[0], cuts, [len(X)]])
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a >= 2:
            ax.plot(X[a:b], Y[a:b], color=color, lw=lw)


def add_P_images(ax, curve_list, p2, color="k", lw=1.0, jump_cut=1.0, nsteps=600):
    for X, Y in curve_list:
        X1, Y1 = rk4_map_batch(X, Y, p2, nsteps=nsteps, direction=+1)
        plot_curve_segments(ax, X1, Y1, color=color, lw=lw, jump_cut=jump_cut)


# ========================= RUN ==============================

z0, z1, DP2, eigvals, vu, vs, resid, per1_err = find_period2_saddle(p2)

lam_abs = np.sort(np.abs(eigvals))
lam_s, lam_u = lam_abs[0], lam_abs[-1]

print("Period-2 saddle of P (computed as fixed point of P^2):")
print(f"  z0 = {z0}")
print(f"  z1 = {z1}")
print(f"  ||P^2(z0)-z0|| = {resid:.3e}")
print(f"  ||P(z0)-z0||   = {per1_err:.8e}")
print(f"  |lambda_s(P^2)| = {lam_s:.8f}")
print(f"  |lambda_u(P^2)| = {lam_u:.8f}")
print(f"  vu = {vu}")
print(f"  vs = {vs}")

_, Wu_plus = adaptive_branch_P2(
    z0, +vu, p2, direction=+1,
    K=K, smin=smin, smax=smax, N0=N0, segmax=segmax,
    rounds=rounds, max_points=max_points, nsteps=nsteps
)
_, Wu_minus = adaptive_branch_P2(
    z0, -vu, p2, direction=+1,
    K=K, smin=smin, smax=smax, N0=N0, segmax=segmax,
    rounds=rounds, max_points=max_points, nsteps=nsteps
)

fig, ax = plt.subplots(figsize=(9.5, 7.2))

for X, Y in Wu_plus:
    plot_curve_segments(ax, X, Y, color="black", lw=1.15, jump_cut=jump_cut)
for X, Y in Wu_minus:
    plot_curve_segments(ax, X, Y, color="black", lw=1.15, jump_cut=jump_cut)

add_P_images(ax, Wu_plus,  p2, color="black", lw=1.15, jump_cut=jump_cut, nsteps=nsteps)
add_P_images(ax, Wu_minus, p2, color="black", lw=1.15, jump_cut=jump_cut, nsteps=nsteps)

if STABLE_MODE.lower() == "reflect":
    for X, Y in Wu_plus:
        Xr, Yr = reflect_G(X, Y)
        plot_curve_segments(ax, Xr, Yr, color="red", lw=1.0, jump_cut=jump_cut)
    for X, Y in Wu_minus:
        Xr, Yr = reflect_G(X, Y)
        plot_curve_segments(ax, Xr, Yr, color="red", lw=1.0, jump_cut=jump_cut)

    reflected_plus  = [reflect_G(X, Y) for X, Y in Wu_plus]
    reflected_minus = [reflect_G(X, Y) for X, Y in Wu_minus]
    add_P_images(ax, reflected_plus,  p2, color="red", lw=1.0, jump_cut=jump_cut, nsteps=nsteps)
    add_P_images(ax, reflected_minus, p2, color="red", lw=1.0, jump_cut=jump_cut, nsteps=nsteps)

elif STABLE_MODE.lower() == "backward":
    _, Ws_plus = adaptive_branch_P2(
        z0, +vs, p2, direction=-1,
        K=K, smin=smin, smax=smax, N0=N0, segmax=segmax,
        rounds=rounds, max_points=max_points, nsteps=nsteps
    )
    _, Ws_minus = adaptive_branch_P2(
        z0, -vs, p2, direction=-1,
        K=K, smin=smin, smax=smax, N0=N0, segmax=segmax,
        rounds=rounds, max_points=max_points, nsteps=nsteps
    )

    for X, Y in Ws_plus:
        plot_curve_segments(ax, X, Y, color="red", lw=1.0, jump_cut=jump_cut)
    for X, Y in Ws_minus:
        plot_curve_segments(ax, X, Y, color="red", lw=1.0, jump_cut=jump_cut)

    add_P_images(ax, Ws_plus,  p2, color="red", lw=1.0, jump_cut=jump_cut, nsteps=nsteps)
    add_P_images(ax, Ws_minus, p2, color="red", lw=1.0, jump_cut=jump_cut, nsteps=nsteps)
else:
    raise ValueError("STABLE_MODE must be 'backward' or 'reflect'")

ax.scatter([z0[0]], [z0[1]], s=28, color="black", zorder=5)
ax.scatter([z1[0]], [z1[1]], s=28, color="black", zorder=5)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.grid(True, alpha=0.25)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(
    f"Period-2 saddle separatrices by continuation: p1={p1}, p2={p2}\n"
    f"K={K}, smax={smax:g}, segmax={segmax:g}, stable_mode={STABLE_MODE}"
)

plt.tight_layout()

if savefig:
    plt.savefig(outfile, dpi=180, bbox_inches="tight")
    print(f"Saved to: {outfile}")

plt.show()
