# Input data

The empirical pipeline expects three cleaned input files beside `thesis_mujtaba_final.py`:

| File | Frequency | Purpose |
|---|---:|---|
| `monthly_signals.csv` | Monthly | Corporate-bond outcomes, credit variables, equity-implied signals and macro-financial controls |
| `daily_signals.csv` | Daily | Daily variables used to construct monthly measures and timing adjustments |
| `french_daily.csv` | Daily | Fama–French market-factor observations used in return construction and validation |

The first column of each file must contain dates that pandas can parse. Monthly observations should be month-end dated. The script expects the following columns:

### `monthly_signals.csv`

| Column | Interpretation | Unit or format |
|---|---|---|
| `rf_m` | Monthly risk-free return | Percent |
| `mktrf_m` | Monthly market excess return | Percent |
| `tri_hy`, `tri_ig` | High-yield and investment-grade total-return indices | Index levels |
| `dgs10` | Ten-year Treasury yield | Percent |
| `oas_hy`, `oas_ig` | High-yield and investment-grade option-adjusted spreads | Percentage points |
| `ebp` | Excess bond premium | Percentage points |
| `epu_3comp` | Three-component economic-policy-uncertainty index | Positive index level |
| `term_spread` | Term spread | Percentage points |
| `move` | Treasury-market implied-volatility index | Index level |
| `usrec` | US recession indicator | `0` or `1` |
| `baa`, `aaa` | Moody's Baa and Aaa corporate yields | Percent |
| `vixcls` | VIX closing level | Index level |
| `skew` | CBOE SKEW index | Index level |

### `daily_signals.csv`

| Column | Interpretation | Unit or format |
|---|---|---|
| `nfci` | Chicago Fed National Financial Conditions Index | Index level |

### `french_daily.csv`

| Column | Interpretation | Unit or format |
|---|---|---|
| `mktrf_d` | Daily market excess return | Percent |
| `rf_d` | Daily risk-free return | Percent |

## Availability

The research dataset includes fields derived from licensed Bloomberg data. The underlying observations are therefore excluded from this repository and covered by `.gitignore`.

Users with appropriate data access can reconstruct the inputs from their authorised sources. Before public release, this directory may be extended with:

- A full data dictionary
- Source and transformation notes for each field
- Public download instructions where permitted
- A small synthetic dataset for testing the pipeline without licensed information

Do not commit locally obtained Bloomberg, Refinitiv, Capital IQ or other licensed data to the repository.
