
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
#  Duffing:
#      x'' + x + x^3 = p1*x*x' + p2*sin(3t)
#
#  Poincare map for minimal forcing period T = 2*pi/3.
#  Builds separatrices of the deep symmetric resonant saddle.
#
#  Colors:
#    unstable  -> black
#    stable    -> red
#
#  Edit parameters in the USER PARAMETERS block and run.
# ============================================================

# ===================== USER PARAMETERS ======================
p1 = 1.0
p2 = 3.0               # change this

t0 = 0.0               # section phase

# Newton initial guess for the deep symmetric resonant saddle
x_guess = 0.0
y_guess = -14.0        # try -12, -14, -16 if Newton lands on wrong branch

# how many iterates of the short seed segment to draw
K = 8

# seed segment near the saddle
smin = 2e-7
smax = 5e-5            # larger -> longer pieces

# adaptive refinement
N0 = 500
rounds = 6
segmax = 0.025
max_points = 60000

# accuracy of one map step
nsteps = 600
rtol = 1e-10
atol = 1e-12

# stable branches:
#   "reflect"  -> by symmetry G(x,y)=(-x,y)
#   "backward" -> true backward iteration
STABLE_MODE = "reflect"

# plotting window
xlim = (-4.5, 4.5)
ylim = (-16.5, 11.5)

# do not connect pieces across huge jumps
jump_cut = 1.0

# save figure
savefig = True
outfile = f"separatrices_p2_{p2:.5f}.png"
# ============================================================

T = 2.0 * np.pi / 3.0


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
        f,
        (t0, t0 + T),
        Y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    xf, yf, J11, J12, J21, J22 = sol.y[:, -1]
    DP = np.array([[J11, J12], [J21, J22]], float)
    return np.array([xf, yf], float), DP


def fixed_point_newton(z_init, p2, maxit=16, tol=1e-12):
    z = np.asarray(z_init, float).copy()
    for _ in range(maxit):
        zT, DP = integrate_map(z, p2, with_variational=True, rtol=rtol, atol=atol)
        F = zT - z
        nF = float(np.linalg.norm(F))
        if nF < tol:
            return z, DP, nF

        A = DP - np.eye(2)
        dz = np.linalg.lstsq(A, -F, rcond=None)[0]

        alpha = 1.0
        improved = False
        for _ in range(10):
            z_try = z + alpha * dz
            zT_try = integrate_map(
                z_try, p2, with_variational=False, rtol=2 * rtol, atol=2 * atol
            )
            F_try = zT_try - z_try
            if np.linalg.norm(F_try) < (1.0 - 1e-4 * alpha) * nF:
                z = z_try
                improved = True
                break
            alpha *= 0.5

        if not improved:
            z = z + dz

    zT, DP = integrate_map(z, p2, with_variational=True, rtol=rtol, atol=atol)
    return z, DP, float(np.linalg.norm(zT - z))


def find_deep_symmetric_saddle(p2, guess=(0.0, -14.0)):
    z, DP, resid = fixed_point_newton(guess, p2)
    eigvals, eigvecs = np.linalg.eig(DP)
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

    return z, DP, eigvals, vu, vs, resid


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


def iterates_for_s(zs, vec, p2, s, K=8, direction=+1, nsteps=600):
    X = zs[0] + s * vec[0]
    Y = zs[1] + s * vec[1]
    out = []
    for _ in range(K):
        X, Y = rk4_map_batch(X, Y, p2, nsteps=nsteps, direction=direction)
        out.append((X.copy(), Y.copy()))
    return out


def adaptive_branch(
    zs,
    vec,
    p2,
    direction,
    K=8,
    smin=2e-7,
    smax=5e-5,
    N0=500,
    segmax=0.025,
    rounds=6,
    max_points=60000,
    nsteps=600,
):
    s = np.linspace(smin, smax, N0)
    its = None

    for _ in range(rounds):
        its = iterates_for_s(zs, vec, p2, s, K=K, direction=direction, nsteps=nsteps)
        need = np.zeros(len(s) - 1, dtype=bool)

        for X, Y in its:
            seg = np.hypot(np.diff(X), np.diff(Y))
            need |= seg > segmax

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


# ========================= RUN ==============================

zs, DP, eigvals, vu, vs, resid = find_deep_symmetric_saddle(
    p2, guess=(x_guess, y_guess)
)

lam_abs = np.sort(np.abs(eigvals))
lam_s, lam_u = lam_abs[0], lam_abs[-1]

print("Deep symmetric resonant saddle:")
print(f"  z_s = {zs}")
print(f"  residual = {resid:.3e}")
print(f"  |lambda_s| = {lam_s:.8f}")
print(f"  |lambda_u| = {lam_u:.8f}")
print(f"  vu = {vu}")
print(f"  vs = {vs}")

# unstable branches
_, Wu_plus = adaptive_branch(
    zs,
    +vu,
    p2,
    direction=+1,
    K=K,
    smin=smin,
    smax=smax,
    N0=N0,
    segmax=segmax,
    rounds=rounds,
    max_points=max_points,
    nsteps=nsteps,
)

_, Wu_minus = adaptive_branch(
    zs,
    -vu,
    p2,
    direction=+1,
    K=K,
    smin=smin,
    smax=smax,
    N0=N0,
    segmax=segmax,
    rounds=rounds,
    max_points=max_points,
    nsteps=nsteps,
)

fig, ax = plt.subplots(figsize=(9.5, 7.2))

# unstable = black
for X, Y in Wu_plus:
    plot_curve_segments(ax, X, Y, color="black", lw=1.15, jump_cut=jump_cut)
for X, Y in Wu_minus:
    plot_curve_segments(ax, X, Y, color="black", lw=1.15, jump_cut=jump_cut)

# stable = red
if STABLE_MODE.lower() == "reflect":
    for X, Y in Wu_plus:
        Xr, Yr = reflect_G(X, Y)
        plot_curve_segments(ax, Xr, Yr, color="red", lw=1.0, jump_cut=jump_cut)
    for X, Y in Wu_minus:
        Xr, Yr = reflect_G(X, Y)
        plot_curve_segments(ax, Xr, Yr, color="red", lw=1.0, jump_cut=jump_cut)

elif STABLE_MODE.lower() == "backward":
    _, Ws_plus = adaptive_branch(
        zs,
        +vs,
        p2,
        direction=-1,
        K=K,
        smin=smin,
        smax=smax,
        N0=N0,
        segmax=segmax,
        rounds=rounds,
        max_points=max_points,
        nsteps=nsteps,
    )
    _, Ws_minus = adaptive_branch(
        zs,
        -vs,
        p2,
        direction=-1,
        K=K,
        smin=smin,
        smax=smax,
        N0=N0,
        segmax=segmax,
        rounds=rounds,
        max_points=max_points,
        nsteps=nsteps,
    )
    for X, Y in Ws_plus:
        plot_curve_segments(ax, X, Y, color="red", lw=1.0, jump_cut=jump_cut)
    for X, Y in Ws_minus:
        plot_curve_segments(ax, X, Y, color="red", lw=1.0, jump_cut=jump_cut)
else:
    raise ValueError("STABLE_MODE must be 'reflect' or 'backward'")

ax.scatter([zs[0]], [zs[1]], s=24, color="black", zorder=5)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.grid(True, alpha=0.25)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(
    f"Resonant saddle separatrices: p1={p1}, p2={p2}\n"
    f"K={K}, smax={smax:g}, segmax={segmax:g}, stable_mode={STABLE_MODE}"
)

plt.tight_layout()

if savefig:
    plt.savefig(outfile, dpi=180, bbox_inches="tight")
    print(f"Saved to: {outfile}")

plt.show()
