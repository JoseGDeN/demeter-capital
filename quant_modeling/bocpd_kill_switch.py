import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp, gammaln


class BOCPDRunner:
    """
    Robust, log-space implementation of Bayesian Online Changepoint
    Detection (BOCPD) using a Normal-Inverse-Gamma conjugate prior,
    resulting in a Student-t posterior predictive distribution.

    Designed to be fed one data point at a time.
    """

    def __init__(self, hazard_lambda=250.0, prune_threshold_log_prob=-10.0):
        """
        Args:
            hazard_lambda (float): Expected run length of a regime.
                                   Hazard rate = 1 / lambda.
            prune_threshold_log_prob (float): Log-probability threshold
                                              for pruning the state space.
        """
        self.hazard_lambda = hazard_lambda
        self.log_hazard = np.log(1.0 / self.hazard_lambda)
        self.log_neg_hazard = np.log(1.0 - (1.0 / self.hazard_lambda))
        self.prune_threshold = prune_threshold_log_prob

        # Priors (set later by _init_priors)
        self.prior_a0 = None  # Alpha (shape) Inv-Gamma
        self.prior_b0 = None  # Beta (scale) Inv-Gamma
        self.prior_k0 = None  # Kappa (confidence) Normal
        self.prior_m0 = None  # Mu (mean) Normal

        # Live state vectors (for all active run lengths)
        # Start with a single run length 0 with prob = 1.
        self.log_R_t = np.array([0.0])  # log P(r_t = i)
        self.alphas = np.array([])      # run-length specific alpha
        self.betas = np.array([])       # run-length specific beta
        self.kappas = np.array([])      # run-length specific kappa
        self.mus = np.array([])         # run-length specific mu

    def _init_priors(self, burn_in_data: np.ndarray):
        """
        Set priors from a burn-in segment.
        """
        if len(burn_in_data) < 2:
            self.prior_m0 = 0.0
            var_guess = 0.01
        else:
            self.prior_m0 = float(np.mean(burn_in_data))
            var_guess = float(np.var(burn_in_data))

        self.prior_k0 = 1.0   # low confidence on mean
        self.prior_a0 = 1.0   # low confidence on variance

        if var_guess == 0:
            var_guess = 0.0001
        self.prior_b0 = var_guess * self.prior_a0

        print(f"Priors initialized: mu={self.prior_m0:.4f}, var_guess={var_guess:.4f}")

        # Initialize state for run length 0
        self.alphas = np.array([self.prior_a0])
        self.betas = np.array([self.prior_b0])
        self.kappas = np.array([self.prior_k0])
        self.mus = np.array([self.prior_m0])

    def _log_student_t_predictive(self, data_point: float) -> np.ndarray:
        """
        Log-pdf of Student-t predictive for all active run lengths.
        """
        if len(self.alphas) == 0:
            return np.array([])

        df = 2.0 * self.alphas
        loc = self.mus

        eps = 1e-10
        scale_sq = self.betas * (self.kappas + 1.0) / ((self.alphas * self.kappas) + eps)
        scale_sq = np.maximum(scale_sq, eps)

        log_prob = (
            gammaln(df / 2.0 + 0.5)
            - gammaln(df / 2.0)
            - 0.5 * np.log(df * np.pi * scale_sq)
            - (df / 2.0 + 0.5) * np.log(1.0 + (data_point - loc) ** 2 / (df * scale_sq))
        )

        return log_prob

    def update(self, data_point: float):
        """
        Process one new return, update internal state.

        Returns
        -------
        changepoint_prob : float
            P(r_t = 0 | x_1:t)
        expected_run_length : float
            E[r_t | x_1:t]
        log_R_t_copy : np.ndarray
            Copy of current log run-length distribution.
        keep_indices : np.ndarray
            Indices (run lengths) corresponding to log_R_t.
        """
        # 1. predictive log-probabilities
        log_pred_probs = self._log_student_t_predictive(data_point)

        # 2. growth probabilities
        if len(self.log_R_t) == 0 or len(log_pred_probs) == 0:
            log_growth_probs = np.array([])
        else:
            log_growth_probs = self.log_R_t + log_pred_probs + self.log_neg_hazard

        # 3. changepoint probability (r_t = 0)
        if len(self.log_R_t) == 0 or len(log_pred_probs) == 0:
            log_cp_prob = np.log(1.0)
        else:
            log_cp_prob = logsumexp(self.log_R_t + log_pred_probs + self.log_hazard)

        # 4. new run-length distribution (shift growth, prepend cp at 0)
        new_log_R_t = np.insert(log_growth_probs, 0, log_cp_prob)

        # 5. normalize
        log_norm_const = logsumexp(new_log_R_t)
        self.log_R_t = new_log_R_t - log_norm_const

        # 6. update sufficient statistics (N-IG)
        if len(self.alphas) == 0:
            new_alphas = np.array([])
            new_betas = np.array([])
            new_kappas = np.array([])
            new_mus = np.array([])
        else:
            new_alphas = self.alphas + 0.5
            new_betas = self.betas + (
                self.kappas * (data_point - self.mus) ** 2
            ) / (2.0 * (self.kappas + 1.0))
            new_kappas = self.kappas + 1.0
            new_mus = (self.kappas * self.mus + data_point) / new_kappas

        # 7. prepend priors for new regime (r_t = 0)
        self.alphas = np.insert(new_alphas, 0, self.prior_a0)
        self.betas = np.insert(new_betas, 0, self.prior_b0)
        self.kappas = np.insert(new_kappas, 0, self.prior_k0)
        self.mus = np.insert(new_mus, 0, self.prior_m0)

        # 8. prune unlikely states
        keep_indices = np.where(self.log_R_t > self.prune_threshold)[0]
        if len(keep_indices) == 0:
            keep_indices = np.array([np.argmax(self.log_R_t)])

        self.log_R_t = self.log_R_t[keep_indices]
        self.alphas = self.alphas[keep_indices]
        self.betas = self.betas[keep_indices]
        self.kappas = self.kappas[keep_indices]
        self.mus = self.mus[keep_indices]

        # outputs
        changepoint_prob = float(np.exp(log_cp_prob))
        active_run_lengths = keep_indices

        if len(active_run_lengths) > 0 and len(self.log_R_t) > 0:
            expected_run_length = float(
                np.sum(np.exp(self.log_R_t) * active_run_lengths)
            )
        else:
            expected_run_length = 0.0

        return changepoint_prob, expected_run_length, self.log_R_t.copy(), keep_indices.copy()


def analyze_pnl_stream(cumulative_pnl: np.ndarray):
    """
    Main wrapper to run BOCPD on a cumulative P&L stream.
    All parameters are inferred heuristically.
    """
    if len(cumulative_pnl) < 2:
        print("Error: P&L stream is too short.")
        return

    # 1. PnL -> returns
    returns = np.diff(cumulative_pnl, prepend=cumulative_pnl[0])
    T = len(returns)

    if T < 50:
        print(f"Error: Data is too short (T={T}). Need at least 50 points.")
        return

    # Heuristics
    burn_in_period = max(30, int(T * 0.15))
    expected_run_length = max(burn_in_period + 10, int(T / 3.0))
    kill_threshold = 0.5
    l_min = max(15, int(expected_run_length * 0.25))
    m_consecutive = max(5, int(l_min * 0.3))

    print("--- BOCPD automatic analysis ---")
    print(f"Total ticks (T):       {T}")
    print(f"Burn-in period:        {burn_in_period}")
    print(f"Expected run length:   {expected_run_length}")
    print(f"Shock kill threshold:  {kill_threshold}")
    print(f"Erosion kill (l_min):  {l_min}")
    print(f"Erosion ticks (m):     {m_consecutive}")
    print("--------------------------------")

    # 2. model
    model = BOCPDRunner(hazard_lambda=float(expected_run_length))

    # 3. priors from burn-in
    burn_in_data = returns[:burn_in_period]
    model._init_priors(burn_in_data)

    # 4. storage
    cp_probs = np.zeros(T)
    exp_rls = np.zeros(T)
    kill_signals_shock = np.zeros(T, dtype=bool)
    kill_signals_erosion = np.zeros(T, dtype=bool)
    run_length_distributions = []

    print(f"Running BOCPD from t=0 to t={T-1}...")
    erosion_counter = 0

    for t in range(T):
        cp, exp_rl, log_R, indices = model.update(returns[t])

        if t < burn_in_period:
            # pendant le burn-in : pas de triggers
            cp_probs[t] = 0.0
            exp_rls[t] = t + 1
            run_length_distributions.append(
                (np.array([t]), np.array([0.0]))
            )
            continue

        cp_probs[t] = cp
        exp_rls[t] = exp_rl
        run_length_distributions.append((indices, log_R))

        # Trigger 1 : shock
        if cp > kill_threshold:
            kill_signals_shock[t] = True

        # Trigger 2 : erosion (après période de grâce)
        grace_period = burn_in_period + l_min
        if exp_rl < l_min and t > grace_period:
            erosion_counter += 1
        else:
            erosion_counter = 0

        if erosion_counter >= m_consecutive:
            for i in range(m_consecutive):
                kill_signals_erosion[t - i] = True

    print("Analysis complete.")
    plot_results(
        cumulative_pnl,
        returns,
        cp_probs,
        exp_rls,
        kill_signals_shock,
        kill_signals_erosion,
        run_length_distributions,
        T,
        burn_in_period,
        kill_threshold,
        l_min,
    )


def plot_results(
    cum_pnl,
    returns,
    cp_probs,
    exp_rls,
    kills_shock,
    kills_erosion,
    rl_dists,
    T,
    burn_in,
    kill_thresh,
    l_min,
):
    """Helper function to plot PnL, returns, run length distribution and triggers."""
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(
        4,
        1,
        figsize=(15, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 3, 2]},
    )

    # 1) PnL
    ax0.plot(cum_pnl, "k-", label="Cumulative P&L")
    ax0.set_title("BOCPD Analysis: Strategy P&L", fontsize=16)
    ax0.axvline(burn_in, color="gray", linestyle=":", label=f"Burn-in ({burn_in})")

    shock_points = np.where(kills_shock)[0]
    erosion_points = np.where(kills_erosion)[0]

    if len(shock_points) > 0:
        shock_starts = shock_points[np.diff(shock_points, prepend=-1) > 1]
        ax0.plot(
            shock_starts,
            cum_pnl[shock_starts],
            "rv",
            markersize=8,
            label="Kill signal (shock)",
        )
    if len(erosion_points) > 0:
        erosion_starts = erosion_points[np.diff(erosion_points, prepend=-1) > 1]
        ax0.plot(
            erosion_starts,
            cum_pnl[erosion_starts],
            "o",
            mfc="orange",
            mec="k",
            markersize=8,
            label="Kill signal (erosion)",
        )

    ax0.legend(loc="upper left")
    ax0.grid(True)

    # 2) Returns
    ax1.bar(range(T), returns, color="gray", width=1.0)
    ax1.set_title("Tick-by-tick returns (model input)")
    ax1.axvline(burn_in, color="gray", linestyle=":")
    ax1.set_ylabel("Return")
    ax1.grid(True)

    # 3) Run-length posterior
    ax2.set_title(r"Run length posterior $P(r_t \mid x_{1:t})$")
    ax2.set_ylabel(r"Run length $r_t$")
    ax2.grid(True)

    max_run_length = 0
    for t in range(burn_in, T):
        indices, log_R = rl_dists[t]
        if len(indices) == 0:
            continue
        probs = np.exp(log_R - log_R.max())
        colors = plt.cm.jet(probs)
        ax2.scatter(
            np.full_like(indices, t),
            indices,
            c=colors,
            s=2,
            marker="s",
            lw=0,
        )
        if len(indices) > 0:
            max_run_length = max(max_run_length, int(indices.max()))

    ax2.plot(exp_rls, "w-", lw=2, label="Expected run length")
    ax2.axvline(burn_in, color="gray", linestyle=":")
    ax2.legend(loc="upper left")
    ax2.set_ylim(0, max(max_run_length + 10, l_min + 10))

    # 4) Triggers
    ax3.plot(cp_probs, "r-", label="P(r_t = 0) (shock trigger)")
    ax3.axhline(kill_thresh, color="r", linestyle=":", label=f"Shock threshold {kill_thresh}")

    ax3.plot(exp_rls, "b-", label="Expected run length (erosion)")
    ax3.axhline(l_min, color="b", linestyle=":", label=f"Erosion threshold {l_min}")

    ax3.set_title("Kill triggers")
    ax3.set_xlabel("Time (ticks)")
    ax3.set_ylabel("Probability / run length")
    ax3.axvline(burn_in, color="gray", linestyle=":")

    max_y_cp = np.max(cp_probs)
    max_y_erl = np.max(exp_rls[burn_in:]) if burn_in < T else l_min
    max_y = max(max_y_cp, max_y_erl, l_min, kill_thresh)
    ax3.set_ylim(-0.05, max_y * 1.1 + 1)

    ax3.legend(loc="upper left")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Demo synthetic example
    np.random.seed(482)
    T_good = 150
    T_bad = 100

    # Regime 1 : bon Sharpe
    returns_good = np.random.normal(0.3, 0.5, T_good)
    # Regime 2 : alpha cassé + vol plus haute
    returns_bad = np.random.normal(-0.4, 1.0, T_bad)

    returns_stream = np.concatenate([returns_good, returns_bad])
    pnl_stream = np.cumsum(returns_stream)

    print("Starting BOCPD auto-analysis...")
    analyze_pnl_stream(pnl_stream)
    print("Analysis finished.")
