# Input data

The empirical pipeline expects three cleaned input files beside `thesis_mujtaba_final.py`:

| File | Frequency | Purpose |
|---|---:|---|
| `monthly_signals.csv` | Monthly | Corporate-bond outcomes, credit variables, equity-implied signals and macro-financial controls |
| `daily_signals.csv` | Daily | Daily variables used to construct monthly measures and timing adjustments |
| `french_daily.csv` | Daily | Fama–French market-factor observations used in return construction and validation |

## Availability

The research dataset includes fields derived from licensed Bloomberg data. The underlying observations are therefore excluded from this repository and covered by `.gitignore`.

Users with appropriate data access can reconstruct the inputs from their authorised sources. Before public release, this directory may be extended with:

- A full data dictionary
- Source and transformation notes for each field
- Public download instructions where permitted
- A small synthetic dataset for testing the pipeline without licensed information

Do not commit locally obtained Bloomberg, Refinitiv, Capital IQ or other licensed data to the repository.

