# Thermodynamic Computing Engine

Models portfolio optimisation as a physical annealing process. Minimises free energy \(F = U - TS\) where \(U\) = negative expected return, \(T\) = risk aversion, \(S\) = portfolio entropy. Equilibrium weights follow Boltzmann distribution: \(w_i \propto \exp(-\mu_i / T)\). Annealing from high to low temperature yields regime‑adaptive allocations. Outputs the final portfolio weights at low temperature.

- **Thermodynamic model:** free energy, entropy, Boltzmann distribution
- **Annealing schedule:** exponential or linear (configurable)
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe by portfolio weight

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
