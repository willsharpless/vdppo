import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def set_ax_style(ax: plt.Axes):
    color_face = "white"
    color_grid = "0.9"
    color_spine = "0.4"

    ax.set_facecolor(color_face)
    ax.grid(color=color_grid, lw=0.5)
    ax.spines["bottom"].set_color(color_spine)
    ax.spines["left"].set_color(color_spine)


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return np.log(p) - np.log1p(-p)


def logistic_normal_mu(eta, sigma, z):
    """
    Monte Carlo estimate of mu = E[ sigmoid(eta + sigma * Z) ], Z ~ N(0,1)
    z: array of standard normal draws, shape (K,)
    """
    return logistic(eta + sigma * z).mean()


def get_ci(b_num, b_tot, B: int = 10_000, K: int = 10_000):
    out = logit_t_with_logistic_normal_mu_ci(
        successes=b_num,
        trials=b_tot,
        alpha=0.05,
        smooth="jeffreys",
        B=B,
        K=B,
        random_seed=42,
    )
    means = out["mu_hat"]
    ci_lo = out["mu_ci"][0]
    ci_hi = out["mu_ci"][1]
    return means, ci_lo, ci_hi


def logit_t_with_logistic_normal_mu_ci(
    successes,
    trials,
    alpha=0.05,
    smooth="jeffreys",
    B=20000,
    K=20000,
    random_seed=0,
):
    """
    "Suggested version":
      1) Smooth seed proportions to avoid 0/1 logits
      2) Compute t-interval on logit scale for eta (mean log-odds across seeds)
      3) Parametric bootstrap over (eta, sigma) to get a CI for
           mu = E_u[ sigmoid(eta + u) ], u ~ N(0, sigma^2)
         under a logistic-normal random effects assumption.

    Returns a dict with:
      - p_hat: raw proportions
      - p_tilde: smoothed proportions used for logit
      - x: logits of p_tilde
      - eta_hat, s_hat: sample mean/sd of logits
      - eta_ci: t-interval for eta (logit scale)
      - p_naive_ci: inverse-logit of eta_ci (interval for sigmoid(eta), NOT E[p])
      - mu_hat: plug-in estimate of E[p] via MC at (eta_hat, s_hat)
      - mu_ci: bootstrap CI for E[p]
      - mu_draws: bootstrap draws of mu (useful for plotting)
    """
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    if successes.shape != trials.shape:
        raise ValueError("successes and trials must have the same shape (one per seed).")
    m = successes.size
    if m < 2:
        raise ValueError("Need at least 2 seeds for a t-interval.")

    # Raw proportions
    p_hat = successes / trials

    # Smoothing to avoid 0/1 logits
    if smooth == "jeffreys":
        # Jeffreys prior smoothing: (y + 0.5) / (n + 1)
        p_tilde = (successes + 0.5) / (trials + 1.0)
    elif smooth == "laplace":
        # Laplace smoothing: (y + 1) / (n + 2)
        p_tilde = (successes + 1.0) / (trials + 2.0)
    elif smooth is None or smooth == "none":
        # Warning: will explode if any p_hat is 0 or 1
        p_tilde = p_hat.copy()
    else:
        raise ValueError("smooth must be 'jeffreys', 'laplace', or None/'none'.")

    # Logits
    x = logit(p_tilde)

    # Across-seed mean/sd on logit scale
    eta_hat = x.mean()
    s_hat = x.std(ddof=1)
    se_eta = s_hat / np.sqrt(m)

    # t-interval for eta
    tcrit = stats.t.ppf(1 - alpha / 2, df=m - 1)
    eta_lo = eta_hat - tcrit * se_eta
    eta_hi = eta_hat + tcrit * se_eta
    eta_ci = (eta_lo, eta_hi)

    # Naive bounded interval (for sigmoid(eta), not E[p])
    p_naive_ci = (logistic(eta_lo), logistic(eta_hi))

    rng = np.random.default_rng(random_seed)

    # Shared standard normal draws for stable MC
    z = rng.standard_normal(size=K)

    # Plug-in estimate of mu = E[sigmoid(eta + sigma Z)] using eta_hat, s_hat
    mu_hat = logistic_normal_mu(eta_hat, s_hat, z)

    # ---- Parametric bootstrap for mu under logistic-normal assumption ----
    # Sampling distribution for (eta, sigma) with unknown sigma:
    #   eta | sigma ~ Normal(eta_hat, sigma / sqrt(m))
    #   (m-1)*s_hat^2 / sigma^2 ~ ChiSquare(df=m-1)
    # => sigma = s_hat * sqrt((m-1)/chi2)
    df = m - 1
    chi2 = rng.chisquare(df=df, size=B)
    sigma_draws = s_hat * np.sqrt(df / chi2)

    eta_draws = rng.normal(loc=eta_hat, scale=sigma_draws / np.sqrt(m), size=B)

    # Compute mu for each bootstrap draw
    # (re-using same z makes this fast/stable and reduces MC noise in the interval)
    mu_draws = np.array([logistic_normal_mu(eta_draws[b], sigma_draws[b], z) for b in range(B)])

    mu_lo, mu_hi = np.quantile(mu_draws, [alpha / 2, 1 - alpha / 2])
    mu_ci = (mu_lo, mu_hi)

    return {
        "p_hat": p_hat,
        "p_tilde": p_tilde,
        "x": x,
        "eta_hat": eta_hat,
        "s_hat": s_hat,
        "eta_ci": eta_ci,
        "p_naive_ci": p_naive_ci,
        "mu_hat": mu_hat,
        "mu_ci": mu_ci,
        "mu_draws": mu_draws,
        "tcrit": tcrit,
        "df": df,
        "alpha": alpha,
        "smooth": smooth,
        "B": B,
        "K": K,
    }
