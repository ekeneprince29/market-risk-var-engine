**Live Demo**

https://market-risk-var-engine-b3gevdcildadadkebe8ker.streamlit.app/

\# Market Risk VaR Engine


A full institutional‑grade Market Risk analytics engine implementing VaR, ES, Monte Carlo, Stress Testing, Liquidity Horizon (FRTB), Backtesting, and interactive scenario analysis. Includes a Streamlit dashboard with risk visualization (charts) and custom stress scenarios.


Market Risk analytics built in Python.

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
\- Single-asset and portfolio price/return charts
\- Custom stress scenario module (asset-level shocks with portfolio impact)




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
<img width="1210" height="767" alt="image" src="https://github.com/user-attachments/assets/c149e926-def7-4974-a53f-bde5308150ab" />
<img width="1032" height="592" alt="image" src="https://github.com/user-attachments/assets/ba7a8ab3-a822-4d13-99b1-d54e549cd264" />
<img width="887" height="401" alt="image" src="https://github.com/user-attachments/assets/68209d6a-0da0-4013-8ca1-0a5e86c220ca" />




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

\- Scenario analysis & risk visualization (interactive charts and stress shocks)

\- Backtesting \& model validation

\- Python engineering



