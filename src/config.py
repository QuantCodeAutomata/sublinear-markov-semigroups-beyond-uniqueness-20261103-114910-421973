"""
Configuration: all experiment parameters for the replication of
'Sublinear Markov Semigroups Beyond Uniqueness'.

All values are module-level UPPER_CASE variables, each with a comment giving
its meaning, unit, and source (paper section / equation, or 'not specified in
the paper').
"""

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42  # global RNG seed; not specified in the paper

# ---------------------------------------------------------------------------
# Example 4.1 parameters (paper Section 4.2, pp. 30-31)
# ---------------------------------------------------------------------------
ALPHA = 0.3          # diffusion exponent alpha; paper: alpha in (0, 1/2)
T0 = 1.0             # time horizon; paper: exists t0 > 0 with Q_0^0(X_{t0} != 0) = 1
                      # exact value not specified — we use 1.0

# sigma_alpha(x) = |x|^alpha / (1 + |x|^alpha)  [paper Eq. in Section 4.2]

# ---------------------------------------------------------------------------
# Monte Carlo simulation parameters (not specified in the paper)
# ---------------------------------------------------------------------------
N_PATHS = 8000       # number of Monte Carlo sample paths per control
DT = 0.005           # Euler-Maruyama time step (seconds); not specified in the paper

# Number of discrete control parameter values for controlled-diffusion model
N_THETA = 25         # grid points per control dimension; not specified in the paper

# ---------------------------------------------------------------------------
# State grid for semigroup evaluation and HJB solver
# ---------------------------------------------------------------------------
X_MIN = -3.0         # left boundary of spatial computational domain; not specified
X_MAX = 3.0          # right boundary; not specified
N_X = 61             # number of spatial grid points; not specified
N_X_COARSE = 21      # coarser grid for quick checks; not specified

# ---------------------------------------------------------------------------
# Time horizons for semigroup evaluation
# ---------------------------------------------------------------------------
T_EVAL_MIN = 0.1     # smallest t for semigroup evaluation; not specified
T_EVAL_MAX = 1.0     # largest t; not specified
N_T_EVAL = 5         # number of evaluation times; not specified

# ---------------------------------------------------------------------------
# Generator estimation (Section 2.2 / Definition 2.2 in the paper)
# Delta_h^T f(x) = (T_h f(x) - f(x)) / h
# ---------------------------------------------------------------------------
H_VALUES = [0.2, 0.1, 0.05, 0.02]  # decreasing sequence for lim sup; not specified

# ---------------------------------------------------------------------------
# Controlled-diffusion example (used in exp1, exp2, exp3, exp7)
# G(x, z, p, H) = sup_{theta in Theta(x)} [0.5 * a_theta * H + b_theta * p + c_theta * z]
# with c_theta <= 0 and a_theta >= 0  (paper Section 1.3, semigroup map definition)
# ---------------------------------------------------------------------------
CTRL_SIGMA_MIN = 0.2  # minimum diffusion coefficient; not specified in the paper
CTRL_SIGMA_MAX = 1.0  # maximum diffusion coefficient; not specified in the paper
CTRL_B_MIN = -0.5     # minimum drift; not specified in the paper
CTRL_B_MAX = 0.5      # maximum drift; not specified in the paper
CTRL_C_MAX = 0.3      # maximum killing rate (k = -c >= 0); not specified in the paper

# ---------------------------------------------------------------------------
# HJB PDE solver parameters (Section 5.2; monotone scheme not specified)
# ---------------------------------------------------------------------------
HJB_NX = 101         # spatial grid points for PDE solver; not specified
HJB_NT = 400         # temporal grid points for PDE solver; not specified
HJB_X_MIN = -3.0     # left boundary for truncated domain; not specified
HJB_X_MAX = 3.0      # right boundary for truncated domain; not specified
HJB_T_MAX = 1.0      # time horizon for PDE; not specified

# ---------------------------------------------------------------------------
# Supermartingale test parameters (Section 3.2, recovered structure R^T)
# ---------------------------------------------------------------------------
N_TEST_PAYOFFS = 5   # number of test payoff functions; not specified
N_TEST_MULT = 4      # number of cylinder multipliers F; not specified

# ---------------------------------------------------------------------------
# Full-run vs reduced-run selector (no reduction needed: pure math, no data)
# ---------------------------------------------------------------------------
FULL_RUN = False     # set to True (via --full flag) for larger N_PATHS / N_THETA
N_PATHS_FULL = 20000
N_THETA_FULL = 50

# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------
T_MAX = HJB_T_MAX        # default time horizon used across experiments; not specified
N_PDE_STEPS = HJB_NT     # time steps in PDE solver; not specified
