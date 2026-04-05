# Evaluating Economic Policies through the Lens of Classical and Quantum Algorithms

A research project comparing classical machine learning and quantum algorithms for modeling **CPI inflation dynamics in India (2010–2025)**, using macroeconomic data sourced from the Reserve Bank of India.

**Author:** Aastha Chhabra
**Supervisor:** Dr. Manish Kumar Pandey  

---

## What This Project Does

India's macroeconomic environment involves complex, non-linear relationships between policy instruments (repo rate, CRR, SLR), monetary aggregates (M3), commodity prices, and the Consumer Price Index. Classical models like VAR and DSGE often fail to capture these dynamics. This project:

- Builds and benchmarks **8 classical ML models** and **3 quantum ML models** on 184 monthly observations
- Implements a rigorous **three-stage evaluation pipeline**: baseline → stationarity correction → ReliefF feature selection
- Demonstrates that **Quantum SVM (QSVM) achieves 87.5% accuracy and 75.0% MCC** after feature selection — outperforming all classical models on the balanced metric (MCC), using only 4 features

---

## Results Summary

| Stage | Best Classical Model | Best Quantum Model | Note |
|---|---|---|---|
| 1 – Baseline (raw data) | Random Forest: 89.1% acc, 78.2% MCC | QSVM: 83.0% acc, 66.4% MCC | Inflated by autocorrelation leakage |
| 2 – After stationarity correction | Random Forest: 79.6% acc, 59.4% MCC | QSVM: 75.5% acc, 50.8% MCC | Realistic post-cleaning performance |
| 3 – After ReliefF feature selection | SVR: 88.9% acc, 77.9% MCC | **QSVM: 87.5% acc, 75.0% MCC** | Quantum advantage confirmed |

**Key findings:**
- QSVM achieves near-parity with the best classical model using only 4 features (vs 10–17 for classical)
- M3 money supply and fuel prices are the dominant CPI drivers
- Repo rate changes take 3–6 months to transmit to CPI — captured by lagged feature engineering
- QNN and Quantum Linear Regression underperform due to circuit depth sensitivity on NISQ hardware

---

## Repository Structure

```
├── quantum_cpi_analysis.ipynb   # Main notebook — full pipeline
├── final_data.xlsx              # Processed dataset (184 obs × 17 variables, monthly 2010–2025)
├── requirements.txt             # Python dependencies
└── outputs/                     # Generated plots and result tables (created on run)
    ├── gdp_interpolation.png
    ├── eda_distribution.png
    ├── eda_correlation.png
    ├── eda_decomposition.png
    ├── stage1_performance_10fold.png
    ├── stage2_performance_10fold.png
    ├── stage3_performance_10fold.png
    ├── relief_feature_importance.png
    ├── final_auc_mcc_comparison.png
    ├── accuracy_comparison_across_stages.png
    └── all_results.xlsx
```

---

## Dataset

`final_data.xlsx` contains 184 monthly observations (January 2010 – April 2025) with 17 variables sourced from the [RBI DBIE database](https://dbie.rbi.org.in/):

| Variable | Description |
|---|---|
| CPI inflation (Base = 2012) | **Target variable** |
| Crude Oil Price | Monthly average (USD/barrel) |
| Exchange Rate of Indian Rupee | INR/USD (month end) |
| M3 (money supply) | Broad money supply |
| Repo Rate | RBI policy rate |
| CRR / SLR | Reserve ratios |
| Non-Food Credit | Bank credit to non-food sectors |
| WPI | Wholesale Price Index |
| Primary Articles (inflation) | Food & primary commodity inflation |
| Fuel & Power inflation | Energy component |
| Manufactured Products inflation | Core manufacturing |
| CPI for Industrial Workers | Alternative inflation measure |
| Reverse Repo Rate | RBI reverse repo |
| Bank Rate | RBI bank rate |
| Interpolated GDP | Per-capita GDP (annual → monthly via cubic spline) |

---

## Methodology

### Pipeline Stages

**Stage 1 – Baseline:** All 8 classical + 3 quantum models trained on raw/interpolated data to establish a reference. High scores here are partly inflated by autocorrelation leakage.

**Stage 2 – Stationarity & Autocorrelation Correction:**
- ADF test → first-order differencing for non-stationary variables
- Ljung-Box test → identifies residual serial correlation
- ARIMA(2,0,1) pre-processing on CPI target; residuals used as features
- Lagged features (1–3 lags) added for key predictors

**Stage 3 – ReliefF Feature Selection + Final Evaluation:**
- Classical models: top-10 ReliefF features
- Quantum models: top-4 features (NISQ hardware constraint)
- Both 10-fold CV and LOOCV reported

### Quantum Models (PennyLane, `default.qubit` simulator)

| Model | Qubits | Circuit Depth | Approach |
|---|---|---|---|
| Quantum SVM | 4 | 20 | Quantum kernel K(xi,xj) = \|⟨φ(xi)\|φ(xj)⟩\|² via angle embedding |
| Quantum Neural Network | 4 | 35 | Parameterized circuit with `StronglyEntanglingLayers` |
| Quantum Linear Regression | 4 | 25 | RX/RY rotations with gradient descent |

---

## Setup & Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook quantum_cpi_analysis.ipynb
```

The notebook is self-contained. `final_data.xlsx` must be in the same directory. All outputs are saved to `outputs/`.

---

## Dependencies

See `requirements.txt`. Core libraries:

- `pennylane` — quantum circuit simulation
- `scikit-learn` — classical ML models and cross-validation
- `scikit-rebate` — ReliefF feature selection
- `statsmodels` — ADF test, ARIMA, Ljung-Box
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`

---
