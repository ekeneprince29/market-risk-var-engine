\# Market Risk VaR Engine



A full institutional‑grade Market Risk analytics engine built in Python.

Implements industry‑standard methodologies used across banks, trading desks, and treasury functions.



\## Features



\- Historical VaR (non‑parametric)

\- Parametric VaR (variance‑covariance)

\- Monte Carlo VaR (normal simulation)

\- Monte Carlo VaR with correlations (Cholesky)

\- Expected Shortfall (ES)

\- Portfolio VaR (multi‑asset)

\- Backtesting (Kupiec Proportion of Failures Test)

\- Streamlit dashboard for interactive analysis



\## Project structure



var-engine/

│

├── main.py

├── dashboard.py

├── README.md

│

├── data/

│   ├── prices.csv

│   └── portfolio\_prices.csv

│

└── src/

&#x20;   ├── historical\_var.py

&#x20;   ├── parametric\_var.py

&#x20;   ├── monte\_carlo\_var.py

&#x20;   ├── monte\_carlo\_correlated.py

&#x20;   ├── portfolio\_var.py

&#x20;   └── backtesting.py



\## How to run



python main.py



\## How to run the dashboard



streamlit run dashboard.py



\## Why this project matters



This engine replicates core risk calculations used in:



\- Market Risk

\- Treasury Risk

\- Trading Controls

\- FRTB IMA

\- Stress Testing

\- Liquidity Risk



It demonstrates:



\- Risk modeling

\- Statistical simulation

\- Portfolio theory

\- Backtesting \& model validation

\- Python engineering



