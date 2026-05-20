import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

def boltzmann_weights(returns, T):
    """
    Compute equilibrium weights using Boltzmann distribution:
    w_i = exp(mean_i / T) / sum_j exp(mean_j / T)
    """
    mean_returns = returns.mean(axis=0).values
    # Avoid overflow: subtract max
    max_mean = np.max(mean_returns)
    exp_vals = np.exp((mean_returns - max_mean) / T)
    weights = exp_vals / np.sum(exp_vals)
    return weights

def free_energy(weights, returns, T):
    """
    F = U - T S
    U = - Σ w_i * mean_i (negative of expected return)
    S = - Σ w_i log w_i
    """
    mean_returns = returns.mean(axis=0).values
    U = -np.sum(weights * mean_returns)   # internal energy (negative expected return)
    S = -np.sum(weights * np.log(weights + 1e-12))  # entropy
    return U - T * S

def anneal_weights(returns, T_init, T_final, steps, schedule='exponential', alpha=0.9):
    """
    Simulated annealing to find optimal weights that minimise free energy.
    At each temperature, we perform a local optimisation (or direct Boltzmann).
    For the Boltzmann distribution, the equilibrium weights are given analytically,
    but we can also simulate the cooling process.
    """
    temperatures = []
    if schedule == 'exponential':
        temperatures = T_init * (alpha ** np.arange(steps))
        temperatures = temperatures[temperatures >= T_final]
    else:  # linear
        temperatures = np.linspace(T_init, T_final, steps)
    # Start with equal weights
    n = returns.shape[1]
    weights = np.ones(n) / n
    history = []
    for T in temperatures:
        # At each temperature, compute Boltzmann weights (analytical solution)
        weights = boltzmann_weights(returns, T)
        # Compute free energy
        F = free_energy(weights, returns, T)
        history.append((T, weights.copy(), F))
    # Final weights at the lowest temperature
    final_weights = weights
    return final_weights, history

def compute_thermodynamic_portfolio(returns_df, window, T_init=100.0, T_final=0.01, steps=50, schedule='exponential', alpha=0.9):
    """
    For the last `window` days of returns, compute the optimal weights via annealing.
    Returns:
        weights: dict {ticker: weight}
        history: list of (T, weights, F) for each step
    """
    if len(returns_df) < window:
        return None, None
    ret_win = returns_df.iloc[-window:]
    final_weights, history = anneal_weights(ret_win, T_init, T_final, steps, schedule, alpha)
    tickers = ret_win.columns.tolist()
    weights_dict = {tickers[i]: final_weights[i] for i in range(len(tickers))}
    return weights_dict, history
