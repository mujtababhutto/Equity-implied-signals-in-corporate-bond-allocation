"""
Equity-Implied Signals in Corporate Bond Allocation

Inputs:
    monthly_signals.csv
    daily_signals.csv
    french_daily.csv

Outputs:
    results/*.csv
    results/figures/*.{pdf,png}

Entry point:
    thesis_mujtaba_final.py
"""

# ==============================================================================
# 0. Configuration
# ==============================================================================

from pathlib import Path

if "__file__" not in globals():
    raise SystemExit("Project directory could not be resolved because __file__ is unavailable.")

PROJECT_DIR = Path(__file__).resolve().parent
PATH_MONTHLY = PROJECT_DIR / "monthly_signals.csv"
PATH_DAILY = PROJECT_DIR / "daily_signals.csv"
PATH_FRENCH = PROJECT_DIR / "french_daily.csv"
OUTPUT_DIR = PROJECT_DIR / "results"
FIGURE_SUBFOLDER = "figures"
PRINT_TABLES = False

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "arch": "arch",
    "matplotlib": "matplotlib",
}

SAMPLE_START = "1994-02-28"
SAMPLE_END = "2026-04-30"
OOS_START = "2010-01-31"

HAC_LAGS = 5
HAC_SENSITIVITY = (0, 5, 12)
BOOTSTRAP_REPS = 5000
BLOCK_LENGTH = 6
MIN_TRAIN_OOS = 60
SEED_MAIN = 20260729
SEED_TACTICAL = 20260730
SEED_BOOTSTRAP = 12345
SEED_SPA = 777
HOLM_FAMILY = "separate"

Z_BURN_IN = 36
DURATION_BURN_IN = 60
VOL_WINDOW = 36
NFCI_LAG_DAYS = 8
W_IG, W_HY = 0.80, 0.20

CREDIT_SPREAD_DV = ["oas_diff", "d3_oas_hy"]
CREDIT_LEG_DV = ["oas_hy", "d3_oas_hy"]
CREDIT_ALT = ["ebp", "d3_oas_hy"]
CREDIT_MOODYS = ["baa_aaa", "d3_baa_aaa"]
EQUITY_BLOCK = ["ln_vix", "skew", "vrp"]
LAGGED_EQUITY = ["ret_eq_l"]

EXPECTED_SIGNS = {
    "oas_diff": +1, "oas_hy": +1, "d3_oas_hy": -1, "ebp": +1,
    "baa_aaa": +1, "d3_baa_aaa": -1,
    "ln_vix": +1, "skew": +1, "vrp": +1,
    "term_spread": +1, "nfci": -1, "ln_epu": -1, "move": +1,
    "ret_eq_l": 0,
}

CRISIS_WINDOWS = [("2008-01-31", "2009-06-30"), ("2020-02-29", "2020-06-30")]

GAMMA_WEIGHT = 5.0
GAMMAS_CE = (3.0, 5.0, 10.0)

W_HY_STRATEGIC = 0.20
W_HY_ANCHOR_ALT = 0.50
BAND_Q = 0.20
TARGET_SIGMA_W = 0.10
CAL_START = "2000-01-31"
CAL_END = "2009-12-31"

INFO_SETS = {
    "I1 persistence": [],
    "I2 credit": ["oas_diff", "d3_oas_hy"],
    "I3 equity": ["ln_vix", "skew", "vrp"],
    "I4 encompassing": ["oas_diff", "d3_oas_hy", "ln_vix", "skew", "vrp"],
}
COMBINATION_SIGNALS = ["oas_diff", "d3_oas_hy", "ln_vix", "skew", "vrp"]

SUBPERIODS = {
    "2010-2014": ("2010-01-31", "2014-12-31"),
    "2015-2019": ("2015-01-31", "2019-12-31"),
    "2020-2026": ("2020-01-31", "2026-04-30"),
}


# ==============================================================================
# 1. Imports and environment
# ==============================================================================

import contextlib
import importlib
import io
import warnings


def _ensure_packages() -> None:
    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append((module_name, pip_name))
    if missing:
        names = ", ".join(m for m, _ in missing)
        packages = " ".join(p for _, p in missing)
        raise SystemExit(
            f"Missing required packages: {names}\n"
            f"Install dependencies with: python -m pip install -r requirements.txt\n"
            f"Equivalent package names: {packages}"
        )


_ensure_packages()

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib as mpl
# Non-interactive Matplotlib backend for PDF and PNG figure output.
mpl.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from arch.unitroot import PhillipsPerron
from arch.bootstrap import SPA, StationaryBootstrap

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)

OUT = Path(OUTPUT_DIR)
FIG = OUT / FIGURE_SUBFOLDER
OOS_START_TS = pd.Timestamp(OOS_START)
CAL_START_TS = pd.Timestamp(CAL_START)
CAL_END_TS = pd.Timestamp(CAL_END)

rng = np.random.default_rng(SEED_MAIN)


def head(title: str) -> None:
    if PRINT_TABLES:
        print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def run_stage(label: str, func, *args, **kwargs):
    print(label, flush=True)
    if PRINT_TABLES:
        return func(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(OUT / f"{name}.csv")
    return df


def validate_environment() -> None:
    missing = [p for p in (PATH_MONTHLY, PATH_DAILY, PATH_FRENCH) if not Path(p).is_file()]
    if missing:
        raise SystemExit(
            "Cannot find these input files:\n  "
            + "\n  ".join(str(p) for p in missing)
            + "\n\nThe required input files should be located beside thesis_mujtaba_final.py."
        )
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    if PRINT_TABLES:
        print(f"Input files found. Output folder: {OUT.resolve()}")


# ==============================================================================
# SECTION 2. DATA INGESTION
# ==============================================================================

def load_raw():
    monthly = pd.read_csv(PATH_MONTHLY, index_col=0, parse_dates=True)
    daily = pd.read_csv(PATH_DAILY, index_col=0, parse_dates=True)
    french = pd.read_csv(PATH_FRENCH, index_col=0, parse_dates=True)
    print(f"  monthly : {monthly.shape[0]} rows x {monthly.shape[1]} cols, "
          f"{monthly.index.min().date()} to {monthly.index.max().date()}")
    print(f"  daily   : {daily.shape[0]} rows x {daily.shape[1]} cols")
    print(f"  french  : {french.shape[0]} rows x {french.shape[1]} cols")
    return monthly, daily, french


# ==============================================================================
# SECTION 3. VARIABLE CONSTRUCTION
# ==============================================================================

def expanding_z(s: pd.Series, min_periods: int = Z_BURN_IN, clip: float = 3.0) -> pd.Series:
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std()
    return ((s - mu) / sd).clip(-clip, clip)


def build_variables(m: pd.DataFrame, daily: pd.DataFrame, french: pd.DataFrame) -> pd.DataFrame:
    p = pd.DataFrame(index=m.index)

    p["rf"] = m["rf_m"] / 100
    p["ret_hy"] = m["tri_hy"] / m["tri_hy"].shift(1) - 1
    p["ret_ig"] = m["tri_ig"] / m["tri_ig"].shift(1) - 1
    p["y_tot"] = p["ret_hy"] - p["ret_ig"]                       # DV1, relative return
    p["dv2_hy"], p["dv2_ig"] = p["ret_hy"], p["ret_ig"]          # DV2, the two legs
    p["dv3_credit"] = W_IG * p["ret_ig"] + W_HY * p["ret_hy"] - p["rf"]   # DV3, credit exposure
    p["d_y10"] = m["dgs10"].diff() / 100

    dur_hy = pd.Series(index=p.index, dtype=float)
    dur_ig = pd.Series(index=p.index, dtype=float)
    for i in range(DURATION_BURN_IN, len(p)):
        w = p.iloc[:i][["ret_hy", "ret_ig", "d_y10"]].dropna()
        if len(w) < DURATION_BURN_IN:
            continue
        X = sm.add_constant(w["d_y10"])
        dur_hy.iloc[i] = -sm.OLS(w["ret_hy"], X).fit().params["d_y10"]
        dur_ig.iloc[i] = -sm.OLS(w["ret_ig"], X).fit().params["d_y10"]
    p["xr_hy"] = p["ret_hy"] + dur_hy * p["d_y10"]
    p["xr_ig"] = p["ret_ig"] + dur_ig * p["d_y10"]
    p["y_xr"] = p["xr_hy"] - p["xr_ig"]                          # DV used by robustness R7

    s = p["y_tot"].dropna()
    ma2 = ARIMA(s.values, order=(0, 0, 2), trend="c").fit()
    theta1, theta2 = np.asarray(ma2.params)[1:3]
    p["dv1_unsm"] = pd.Series(s.mean() + ma2.resid * (1 + theta1 + theta2), index=s.index)
    p.attrs["ma2"] = (theta1, theta2)

    p["oas_hy"], p["oas_ig"] = m["oas_hy"], m["oas_ig"]
    p["oas_diff"] = m["oas_hy"] - m["oas_ig"]
    p["d3_oas_hy"] = m["oas_hy"].diff(3)
    p["ebp"] = m["ebp"].shift(1)                  # published during t+1, so entered lagged
    p["ln_epu"] = np.log(m["epu_3comp"]).shift(1)
    p["term_spread"] = m["term_spread"]
    p["move"], p["usrec"] = m["move"], m["usrec"]

    p["baa_aaa"] = m["baa"] - m["aaa"]
    p["d3_baa_aaa"] = p["baa_aaa"].diff(3)

    nf = daily["nfci"].dropna()
    p["nfci"] = pd.Series({
        d: (nf.loc[:d - pd.Timedelta(days=NFCI_LAG_DAYS)].iloc[-1]
            if len(nf.loc[:d - pd.Timedelta(days=NFCI_LAG_DAYS)]) else np.nan)
        for d in p.index})

    p["ln_vix"], p["skew"] = np.log(m["vixcls"]), m["skew"]

    fr = french.copy()
    fr["mkt"] = (fr["mktrf_d"] + fr["rf_d"]) / 100
    fr["rv21"] = (fr["mkt"] ** 2).rolling(21).sum() * (252 / 21)   # annualised realised variance
    p["rv21"] = fr["rv21"].resample("ME").last()
    p["vrp"] = (m["vixcls"] / 100) ** 2 - p["rv21"]                # variance risk premium

    eq = (m["mktrf_m"] + m["rf_m"]) / 100
    p["ret_eq"] = eq
    p["ret_eq_l"] = eq          # Equity return dated t. Every specification regresses

    p["mom_12_1"] = (1 + eq).rolling(11).apply(np.prod, raw=True).shift(1) - 1
    p["mom_12_0"] = (1 + eq).rolling(12).apply(np.prod, raw=True) - 1

    for c in CREDIT_SPREAD_DV + EQUITY_BLOCK + ["oas_hy", "ebp"]:
        p["z_" + c] = expanding_z(p[c])

    return p.loc[SAMPLE_START:SAMPLE_END]


# ==============================================================================
# SECTION 4. PRELIMINARY DIAGNOSTICS  -> Thesis 3.3 and 4.1
# ==============================================================================

def diag_descriptives(p):
    rows = []
    for lab, c in [("HY total return", "ret_hy"), ("IG total return", "ret_ig"),
                   ("HY - IG (DV1)", "y_tot"), ("Credit exposure (DV3)", "dv3_credit"),
                   ("Equity market", "ret_eq")]:
        s = p[c].dropna()
        rows.append(dict(series=lab, n=len(s), mean_pa=s.mean() * 1200,
                         vol_pa=s.std() * np.sqrt(12) * 100, skew=s.skew(),
                         exkurt=s.kurtosis(), min_pct=s.min() * 100,
                         max_pct=s.max() * 100, rho1=s.autocorr(1)))
    return pd.DataFrame(rows).set_index("series")


def diag_predictor_descriptives(p, variables):
    rows = []
    for v in variables:
        s = p[v].dropna()
        rows.append(dict(variable=v, n=len(s), mean=s.mean(), sd=s.std(),
                         minimum=s.min(), maximum=s.max()))
    return pd.DataFrame(rows).set_index("variable")


def diag_skew_break(p, cut="2003-09-30"):
    rows = []
    for lab, c in [("Skewness index", "skew"), ("Log implied volatility", "ln_vix")]:
        for sub, sl in [("pre-2003", slice(None, cut)), ("post-2003", slice(cut, None))]:
            x = p[c].loc[sl].dropna()
            if sub == "post-2003":
                x = x.iloc[1:]
            rows.append(dict(series=lab, subsample=sub, n=len(x), mean=x.mean(),
                             sd=x.std(), rho1=x.autocorr(1)))
    return pd.DataFrame(rows).set_index(["series", "subsample"])


def diag_skew_subsample(p, cut="2003-09-30", dv="y_tot"):
    regs = CREDIT_SPREAD_DV + EQUITY_BLOCK
    rows = []
    for lab, sub in [("Full sample", p),
                     ("Pre-Sept 2003", p.loc[:cut]),
                     ("Post-Sept 2003", p.loc[cut:].iloc[1:])]:
        r, _ = fit_is(sub, dv, regs, 2)
        rows.append(dict(sample=lab, n=int(r.nobs), coef=r.params["skew"],
                         t=r.tvalues["skew"], p=r.pvalues["skew"]))
    return pd.DataFrame(rows).set_index("sample")


def diag_moodys_replication(p, dv="y_tot"):
    rows = []
    for lab, credit in [("Bloomberg OAS differential (baseline)", CREDIT_SPREAD_DV),
                        ("Moody's Baa - Aaa (freely available)", CREDIT_MOODYS)]:
        r, _ = fit_is(p, dv, credit + EQUITY_BLOCK, 2)
        _, p_credit = block_wald(r, credit)
        _, p_equity = block_wald(r, EQUITY_BLOCK)
        rows.append(dict(credit_block=lab, n=int(r.nobs), R2=r.rsquared,
                         credit_block_p=p_credit, equity_block_p=p_equity))
    out = pd.DataFrame(rows).set_index("credit_block")
    out.attrs["corr"] = p[["baa_aaa", "oas_diff"]].corr().iloc[0, 1]
    return out


def diag_stationarity(p, variables):
    rows = []
    for v in variables:
        s = p[v].dropna()
        try:
            kp = kpss(s, regression="c", nlags="auto")[1]
        except Exception:
            kp = np.nan
        adf_p = adfuller(s, autolag="AIC")[1]
        pp_p = PhillipsPerron(s, trend="c").pvalue
        verdict = ("stationary" if (adf_p < .05 and pp_p < .05 and kp > .05)
                   else "unit root" if (adf_p > .10 and pp_p > .10) else "borderline")
        rows.append(dict(variable=v, ADF_p=adf_p, PP_p=pp_p, KPSS_p=kp, verdict=verdict))
    return pd.DataFrame(rows).set_index("variable")


def diag_persistence(p, variables):
    rows = []
    for v in variables:
        s = p[v].dropna()
        rho = sm.OLS(s.values[1:], sm.add_constant(s.values[:-1])).fit().params[1]
        hl = np.log(.5) / np.log(rho) if 0 < rho < 1 else np.inf
        rows.append(dict(variable=v, rho=rho, half_life_m=hl, n=len(s),
                         n_eff=len(s) * (1 - rho) / (1 + rho) if rho < 1 else np.nan))
    return pd.DataFrame(rows).set_index("variable")


def diag_stambaugh(p, dv, variables):
    rows = []
    for v in variables:
        d = pd.concat([p[dv].shift(-1).rename("y1"), p[v].rename("x")], axis=1).dropna()
        if len(d) < 50:
            continue
        ry = sm.OLS(d["y1"], sm.add_constant(d["x"])).fit()
        xs = p[v].dropna()
        rx = sm.OLS(xs.values[1:], sm.add_constant(xs.values[:-1])).fit()
        phi = rx.params[1]
        vres = pd.Series(rx.resid, index=xs.index[1:])
        j = ry.resid.index.intersection(vres.index)
        ev, vv = ry.resid.loc[j], vres.loc[j]
        bias = -(np.cov(ev, vv)[0, 1] / np.var(vv, ddof=1)) * (1 + 3 * phi) / len(d)
        rows.append(dict(variable=v, phi=phi, corr_ev=np.corrcoef(ev, vv)[0, 1],
                         bias=bias, bias_in_SE=bias / ry.bse.iloc[1]))
    return pd.DataFrame(rows).set_index("variable")


def diag_collinearity(p, cols, label):
    X = p[cols].dropna()
    Xs = (X - X.mean()) / X.std()
    Xc = sm.add_constant(Xs)
    vif = pd.Series({c: variance_inflation_factor(Xc.values, i + 1)
                     for i, c in enumerate(cols)}, name=f"VIF_{label}")
    return vif, float(np.sqrt(np.linalg.cond(Xs.corr())))


def diag_smoothing(p, series):
    rows = []
    for lab, c in series:
        s = p[c].dropna()
        res = ARIMA(s.values, order=(0, 0, 2), trend="c").fit()
        th = np.r_[1.0, np.asarray(res.params)[1:3]]
        th = th / th.sum()
        xi = float((th ** 2).sum())
        a = acf(s, nlags=6, fft=False)[1:]
        lb = acorr_ljungbox(s, lags=[6], return_df=True)["lb_pvalue"].iloc[0]
        rows.append(dict(series=lab, rho1=a[0], rho2=a[1], LB6_p=lb,
                         theta0=th[0], theta1=th[1], theta2=th[2],
                         xi=xi, sharpe_inflation=1 / np.sqrt(xi)))
    return pd.DataFrame(rows).set_index("series")


def diag_variance_ratio(p, series, qs=(2, 3, 6)):
    def vr(s, q):
        x = s.dropna().values
        n, mu = len(x), x.mean()
        v1 = ((x - mu) ** 2).sum() / (n - 1)
        agg = np.convolve(x, np.ones(q), "valid")
        vq = ((agg - q * mu) ** 2).sum() / (len(agg) - 1)
        r = vq / (q * v1)
        z = (r - 1) * np.sqrt(n * q) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q))
        return r, z
    rows = []
    for lab, c in series:
        d = {"series": lab}
        for q in qs:
            d[f"VR{q}"], d[f"z{q}"] = vr(p[c], q)
        rows.append(d)
    return pd.DataFrame(rows).set_index("series")


def diag_leadlag(p):
    rows = []
    for lab, dep in [("HY total return", "ret_hy"), ("IG total return", "ret_ig"),
                     ("HY - IG", "y_tot")]:
        d = p[[dep, "ret_eq"]].copy()
        d["eq_l1"] = p["ret_eq"].shift(1)
        d["eq_l2"] = p["ret_eq"].shift(2)
        d = d.dropna()
        r = sm.OLS(d[dep], sm.add_constant(d[["ret_eq", "eq_l1", "eq_l2"]])) \
              .fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        rows.append(dict(series=lab, b_contemp=r.params["ret_eq"], t_contemp=r.tvalues["ret_eq"],
                         b_lag1=r.params["eq_l1"], t_lag1=r.tvalues["eq_l1"],
                         b_lag2=r.params["eq_l2"], t_lag2=r.tvalues["eq_l2"],
                         lag_share=r.params["eq_l1"] / (r.params["ret_eq"] + r.params["eq_l1"]),
                         R2=r.rsquared))
    return pd.DataFrame(rows).set_index("series")


def diag_duration(p):
    out = {}
    for lab, dep in [("HY", "ret_hy"), ("IG", "ret_ig"), ("HY - IG", "y_tot")]:
        d = p[[dep, "d_y10"]].dropna()
        r = sm.OLS(d[dep], sm.add_constant(d["d_y10"])) \
              .fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        out[lab] = dict(slope=r.params["d_y10"], emp_duration=-r.params["d_y10"],
                        t=r.tvalues["d_y10"], R2=r.rsquared)
    d = p[["y_xr", "d_y10"]].dropna()
    r = sm.OLS(d["y_xr"], sm.add_constant(d["d_y10"])).fit()
    out["HY - IG hedged"] = dict(slope=r.params["d_y10"], emp_duration=-r.params["d_y10"],
                                 t=np.nan, R2=r.rsquared)
    return pd.DataFrame(out).T


def diag_benchmark_order(p, dv="y_tot"):
    rows = []
    s = p[dv].dropna()
    for q in (1, 2, 3):
        d = pd.concat([s.rename("y")] + [s.shift(k).rename(f"l{k}") for k in range(1, q + 1)],
                      axis=1).dropna()
        r = sm.OLS(d["y"], sm.add_constant(d[[f"l{k}" for k in range(1, q + 1)]])).fit()
        lb = acorr_ljungbox(r.resid, lags=[6], return_df=True)["lb_pvalue"].iloc[0]
        rows.append(dict(model=f"AR({q})", R2=r.rsquared, aic=r.aic, bic=r.bic,
                         resid_LB6_p=lb, residuals_white=lb > 0.10))
    return pd.DataFrame(rows).set_index("model")


# ==============================================================================
# SECTION 5. IN-SAMPLE PREDICTIVE REGRESSIONS  -> Thesis 4.2 - 4.4
# ==============================================================================

def fit_is(p, dv, regs=None, arlags=2, extra=None, hac=HAC_LAGS, start=None, drop=None):
    d = pd.DataFrame({"y": p[dv].shift(-1)})
    for k in range(1, arlags + 1):
        d[f"ar{k}"] = p[dv].shift(k - 1)
    for r in (regs or []) + (extra or []):
        d[r] = p[r]
    d = d.dropna()
    if start:
        d = d.loc[start:]
    if drop:
        for a, b in drop:
            d = d.loc[~((d.index >= a) & (d.index <= b))]
    X = sm.add_constant(d.drop(columns="y"))
    fit = sm.OLS(d["y"], X)
    return (fit.fit() if hac == 0 else fit.fit(cov_type="HAC", cov_kwds={"maxlags": hac})), d


def block_wald(res, names):
    names = [n for n in names if n in res.params.index]
    if not names:
        return np.nan, np.nan
    R = np.zeros((len(names), len(res.params)))
    for i, n in enumerate(names):
        R[i, list(res.params.index).index(n)] = 1
    t = res.f_test(R)
    return float(np.squeeze(t.fvalue)), float(t.pvalue)


def holm(pvals, labels, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    m, adj, run = len(pvals), np.empty(len(pvals)), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * pvals[j]))
        adj[j] = run
    return pd.DataFrame({"test": labels, "p_raw": pvals, "p_holm": adj,
                         "sig_5pct": adj < alpha})


def bootstrap_predictive_p(p, dv, x, reps=BOOTSTRAP_REPS):
    d = pd.DataFrame({"y1": p[dv].shift(-1), "y0": p[dv], "y_1": p[dv].shift(1),
                      "x": p[x]}).dropna()
    t_obs = sm.OLS(d["y1"], sm.add_constant(d[["y0", "y_1", "x"]])) \
              .fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}).tvalues["x"]
    n = len(d)
    n0 = sm.OLS(d["y1"], sm.add_constant(d[["y0", "y_1"]])).fit()
    a, b1, b2 = n0.params
    e = n0.resid.values
    xv = d["x"].values
    rx = sm.OLS(xv[1:], sm.add_constant(xv[:-1])).fit()
    c, phi = rx.params
    v = np.r_[0.0, rx.resid]
    cnt = 0
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        es, vs = e[idx], v[idx]
        xs = np.empty(n); xs[0] = xv[0]
        ys = np.empty(n); ys[:2] = d["y1"].values[:2]
        for i in range(1, n):
            xs[i] = c + phi * xs[i - 1] + vs[i]
        for i in range(2, n):
            ys[i] = a + b1 * ys[i - 1] + b2 * ys[i - 2] + es[i]
        db = pd.DataFrame({"y1": ys[2:], "y0": ys[1:-1], "y_1": ys[:-2], "x": xs[2:]})
        tb = sm.OLS(db["y1"], sm.add_constant(db[["y0", "y_1", "x"]])).fit().tvalues["x"]
        cnt += abs(tb) >= abs(t_obs)
    return t_obs, cnt / reps


DEPENDENT_VARIABLES = {"DV1 y_tot (HY-IG)": "y_tot", "DV2a ret_hy": "dv2_hy",
                       "DV2b ret_ig": "dv2_ig", "DV3 credit exposure": "dv3_credit"}


def credit_block_for(dv):
    return CREDIT_SPREAD_DV if dv == "y_tot" else CREDIT_LEG_DV


def run_insample(p):
    ladder, enc, store = [], [], {}
    for lab, dv in DEPENDENT_VARIABLES.items():
        credit = credit_block_for(dv)
        specs = {"M0 mean": ([], 0, None), "M1 AR(1)": ([], 1, None),
                 "M2 AR(2) [benchmark]": ([], 2, None),
                 "M3 B2+credit": (credit, 2, None),
                 "M4 B2+credit(EBP)": (CREDIT_ALT, 2, None),
                 "M5 B2+equity": (EQUITY_BLOCK, 2, None),
                 "M6 B2+credit+equity": (credit + EQUITY_BLOCK, 2, None),
                 "M7 B3+credit+equity": (credit + EQUITY_BLOCK, 2, LAGGED_EQUITY)}
        r2_b2 = None
        for name, (regs, arl, ex) in specs.items():
            res, _ = fit_is(p, dv, regs, arl, ex)
            store[(lab, name)] = res
            if name.startswith("M2"):
                r2_b2 = res.rsquared
            ladder.append(dict(dv=lab, model=name, n=int(res.nobs), R2=res.rsquared,
                               incr_over_B2=res.rsquared - r2_b2 if r2_b2 is not None else np.nan))
        for mname in ("M6 B2+credit+equity", "M7 B3+credit+equity"):
            r = store[(lab, mname)]
            fe, pe = block_wald(r, EQUITY_BLOCK)
            fc, pc = block_wald(r, credit)
            enc.append(dict(dv=lab, model=mname, equity_F=fe, equity_p=pe,
                            credit_F=fc, credit_p=pc))
    return pd.DataFrame(ladder), pd.DataFrame(enc), store


def coef_table(res, hac_variants=None, p=None, dv=None, regs=None):
    t = pd.DataFrame({"coef": res.params, "se": res.bse,
                      "t": res.tvalues, "p": res.pvalues})
    t["sig"] = np.where(t.p < .01, "***", np.where(t.p < .05, "**",
                        np.where(t.p < .10, "*", "")))
    if hac_variants and p is not None:
        for L in hac_variants:
            r, _ = fit_is(p, dv, regs, 2, hac=L)
            t[f"t_hac{L}"] = r.tvalues.reindex(t.index)
    return t


def run_univariate(p, dv="y_tot"):
    base = fit_is(p, dv, [], 2)[0].rsquared
    rows = []
    for s in ["oas_diff", "oas_hy", "d3_oas_hy", "ebp", "term_spread", "nfci",
              "ln_vix", "skew", "vrp", "mom_12_1", "mom_12_0", "ret_eq_l"]:
        r, _ = fit_is(p, dv, [s], 2)
        rows.append(dict(signal=s, coef=r.params[s], t=r.tvalues[s], p=r.pvalues[s],
                         R2=r.rsquared, incr_over_B2=r.rsquared - base))
    u = pd.DataFrame(rows).set_index("signal")
    h = holm(u.p.values, u.index.tolist())
    u["p_holm"] = h.set_index("test").loc[u.index, "p_holm"].values
    return u


# ==============================================================================
# SECTION 6. OUT-OF-SAMPLE FORECASTING  -> Thesis 4.5
# ==============================================================================

def design_oos(p, dv, regs, arl=2, start=None, drop=None):
    d = pd.DataFrame({"y": p[dv].shift(-1)})
    for k in range(1, arl + 1):
        d[f"ar{k}"] = p[dv].shift(k - 1)
    for r in regs:
        d[r] = p[r]
    d = d.dropna()
    d.index = d.index + pd.offsets.MonthEnd(1)
    if start:
        d = d.loc[start:]
    if drop:
        for a, b in drop:
            d = d.loc[~((d.index >= a) & (d.index <= b))]
    return d


def recursive(p, dv, regs, arl=2, restrict=False, truncate=False, start=None, drop=None):
    d = design_oos(p, dv, regs, arl, start, drop)
    cols = [c for c in d.columns if c != "y"]
    out = {}
    for t in d.index[d.index >= OOS_START_TS]:
        tr = d.loc[d.index < t]
        if len(tr) < MIN_TRAIN_OOS:
            continue
        b = sm.OLS(tr["y"], sm.add_constant(tr[cols])).fit().params.copy()
        if restrict:
            for c in cols:
                s = EXPECTED_SIGNS.get(c, 0)
                if s != 0 and np.sign(b[c]) != s:
                    b[c] = 0.0
        f = float(np.r_[1.0, d.loc[t, cols].values.astype(float)] @ b.values)
        out[t] = max(f, 0.0) if (restrict and truncate) else f
    fc = pd.Series(out)
    return fc, d.reindex(fc.index)["y"]


def prevailing_mean(p, dv, start=None, drop=None):
    d = design_oos(p, dv, [], 0, start, drop)
    return pd.Series({t: d.loc[d.index < t, "y"].mean()
                      for t in d.index[d.index >= OOS_START_TS]})


def r2os(a, f, b):
    return 1 - ((a - f) ** 2).sum() / ((a - b) ** 2).sum()


def clark_west(a, f_restricted, f_unrestricted):
    z = (a - f_restricted) ** 2 - ((a - f_unrestricted) ** 2 - (f_restricted - f_unrestricted) ** 2)
    r = sm.OLS(z, np.ones(len(z))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    t = float(np.asarray(r.tvalues)[0])
    return t, float(stats.norm.sf(t))


def run_oos(p):
    bench, rows = {}, []
    for lab, dv in DEPENDENT_VARIABLES.items():
        credit = credit_block_for(dv)
        pm = prevailing_mean(p, dv)
        f1, _ = recursive(p, dv, [], 1)
        f2, a2 = recursive(p, dv, [], 2)
        j = pm.index.intersection(f1.index).intersection(f2.index)
        a, B0, B1, B2 = a2.loc[j].values, pm.loc[j].values, f1.loc[j].values, f2.loc[j].values
        bench[dv] = dict(idx=j, a=a, B0=B0, B1=B1, B2=B2)

        for nm, f in [("B0 prevailing mean", B0), ("B1 AR(1)", B1), ("B2 AR(2)", B2)]:
            row = dict(dv=lab, model=nm, MSPE=((a - f) ** 2).mean(),
                       R2os_vs_B0=r2os(a, f, B0), R2os_vs_B2=r2os(a, f, B2),
                       CW_t_B2=np.nan, CW_p_B2=np.nan, CW_t_B0=np.nan, CW_p_B0=np.nan)
            if nm == "B2 AR(2)":
                row["CW_t_B0"], row["CW_p_B0"] = clark_west(a, B0, B2)
            rows.append(row)

        models = {"M3 credit": (credit, False), "M3 credit (CT)": (credit, True),
                  "M4 credit EBP": (CREDIT_ALT, False),
                  "M5 equity": (EQUITY_BLOCK, False), "M5 equity (CT)": (EQUITY_BLOCK, True),
                  "M6 encompassing": (credit + EQUITY_BLOCK, False),
                  "M6 encompassing (CT)": (credit + EQUITY_BLOCK, True),
                  "M7 M6 + lagged equity": (credit + EQUITY_BLOCK + LAGGED_EQUITY, False),
                  "lagged equity only": (LAGGED_EQUITY, False)}
        for nm, (regs, ct) in models.items():
            f, _ = recursive(p, dv, regs, 2, restrict=ct,
                             truncate=ct and dv in ("y_tot", "dv3_credit"))
            fv = f.reindex(j).values
            ok = ~np.isnan(fv)
            t2, q2 = clark_west(a[ok], B2[ok], fv[ok])
            t0, q0 = clark_west(a[ok], B0[ok], fv[ok])
            rows.append(dict(dv=lab, model=nm, MSPE=((a[ok] - fv[ok]) ** 2).mean(),
                             R2os_vs_B0=r2os(a[ok], fv[ok], B0[ok]),
                             R2os_vs_B2=r2os(a[ok], fv[ok], B2[ok]),
                             CW_t_B2=t2, CW_p_B2=q2, CW_t_B0=t0, CW_p_B0=q0))

        for cn, sigs in {"combination credit": credit, "combination equity": EQUITY_BLOCK,
                         "combination all": credit + EQUITY_BLOCK}.items():
            fs = [recursive(p, dv, [s], 2)[0].reindex(j) for s in sigs]
            fv = pd.concat(fs, axis=1).mean(axis=1).values
            ok = ~np.isnan(fv)
            t2, q2 = clark_west(a[ok], B2[ok], fv[ok])
            t0, q0 = clark_west(a[ok], B0[ok], fv[ok])
            rows.append(dict(dv=lab, model=cn, MSPE=((a[ok] - fv[ok]) ** 2).mean(),
                             R2os_vs_B0=r2os(a[ok], fv[ok], B0[ok]),
                             R2os_vs_B2=r2os(a[ok], fv[ok], B2[ok]),
                             CW_t_B2=t2, CW_p_B2=q2, CW_t_B0=t0, CW_p_B0=q0))

    res = pd.DataFrame(rows)
    fam2 = res.dropna(subset=["CW_p_B2"])
    fam0 = res.dropna(subset=["CW_p_B0"])
    labels2 = (fam2.dv.str.split().str[0] + " | " + fam2.model).tolist()
    labels0 = (fam0.dv.str.split().str[0] + " | " + fam0.model).tolist()

    if HOLM_FAMILY == "pooled":
        pooled_p = np.r_[fam2.CW_p_B2.values, fam0.CW_p_B0.values]
        adj = holm(pooled_p, labels2 + labels0).p_holm.values
        res.loc[fam2.index, "CW_p_B2_holm"] = adj[:len(fam2)]
        res.loc[fam0.index, "CW_p_B0_holm"] = adj[len(fam2):]
    else:
        res.loc[fam2.index, "CW_p_B2_holm"] = holm(fam2.CW_p_B2.values, labels2).p_holm.values
        res.loc[fam0.index, "CW_p_B0_holm"] = holm(fam0.CW_p_B0.values, labels0).p_holm.values

    return res, bench


# ==============================================================================
# SECTION 7. FORECAST-LAYER PORTFOLIOS  -> Thesis 4.10 - 4.11
# ==============================================================================

def evaluate_portfolio(p, idx, w, label, hy="ret_hy", ig="ret_ig"):
    w = w.reindex(idx).astype(float)
    h, i_, f_ = p[hy].reindex(idx), p[ig].reindex(idx), p["rf"].reindex(idx)
    ok = w.notna() & h.notna() & i_.notna()
    w, h, i_, f_ = w[ok], h[ok], i_[ok], f_[ok]
    Rp = w * h + (1 - w) * i_
    ex = Rp - f_
    wd = (w.shift(1) * (1 + h)) / (w.shift(1) * (1 + h) + (1 - w.shift(1)) * (1 + i_))
    to = (w - wd).abs().dropna()
    dn = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    wealth = (1 + Rp).cumprod()
    d = dict(strategy=label, mean_w=w.mean(), ret_ann=Rp.mean() * 12,
             vol_ann=Rp.std() * np.sqrt(12),
             sharpe=(ex.mean() * 12) / (Rp.std() * np.sqrt(12)),
             sortino=(ex.mean() * 12) / (dn * np.sqrt(12)) if dn > 0 else np.nan,
             maxDD=float((1 - wealth / wealth.cummax()).max()),
             turnover_ann=to.mean() * 12)
    for g in GAMMAS_CE:
        d[f"CE_g{int(g)}"] = (Rp.mean() - .5 * g * Rp.var()) * 12
    d["CE_CRRA_g5"] = (((1 + Rp) ** (1 - 5.0)).mean() ** (1 / (1 - 5.0))) ** 12 - 1
    return d, Rp


def run_portfolio(p, bench):
    dv = "y_tot"
    idx = bench[dv]["idx"]
    sigma = p[dv].rolling(VOL_WINDOW).std().shift(1).reindex(idx)

    def unconstrained_mv(f):
        return (f.reindex(idx) / (GAMMA_WEIGHT * sigma ** 2)).clip(0, 1)

    F = {"B0 prevailing mean": prevailing_mean(p, dv),
         "B1 AR(1)": recursive(p, dv, [], 1)[0],
         "B2 AR(2)": recursive(p, dv, [], 2)[0],
         "M3 credit": recursive(p, dv, CREDIT_SPREAD_DV)[0],
         "M3 credit (CT)": recursive(p, dv, CREDIT_SPREAD_DV, restrict=True, truncate=True)[0],
         "M5 equity": recursive(p, dv, EQUITY_BLOCK)[0],
         "M6 encompassing": recursive(p, dv, CREDIT_SPREAD_DV + EQUITY_BLOCK)[0],
         "M7 M6 + lagged equity": recursive(p, dv, CREDIT_SPREAD_DV + EQUITY_BLOCK
                                            + LAGGED_EQUITY)[0],
         "spread only": recursive(p, dv, ["oas_diff"])[0],
         "combination credit": pd.concat([recursive(p, dv, [s])[0] for s in CREDIT_SPREAD_DV],
                                         axis=1).mean(axis=1),
         "combination equity": pd.concat([recursive(p, dv, [s])[0] for s in EQUITY_BLOCK],
                                         axis=1).mean(axis=1)}

    vh = p["ret_hy"].rolling(VOL_WINDOW).std().shift(1).reindex(idx)
    vi = p["ret_ig"].rolling(VOL_WINDOW).std().shift(1).reindex(idx)
    BM = {"50/50 static": pd.Series(.5, index=idx),
          "inverse volatility": ((1 / vh) / ((1 / vh) + (1 / vi))).clip(0, 1),
          "market weight 20/80": pd.Series(.2, index=idx)}

    rows, R = [], {}
    for k, w in BM.items():
        d, Rp = evaluate_portfolio(p, idx, w, k); rows.append(d); R[k] = Rp
    for k, f in F.items():
        d, Rp = evaluate_portfolio(p, idx, unconstrained_mv(f), k); rows.append(d); R[k] = Rp

    T = pd.DataFrame(rows).set_index("strategy")
    base = R["50/50 static"]
    ir, exs = [], []
    for s in T.index:
        j = R[s].index.intersection(base.index)
        dd = R[s].loc[j] - base.loc[j]
        ir.append((dd.mean() * 12) / (dd.std() * np.sqrt(12)) if dd.std() > 0 else np.nan)
        exs.append(dd.mean() * 12)
    T["IR_vs_5050"], T["excess_vs_5050"] = ir, exs
    T["breakeven_bp"] = np.where(T.turnover_ann > 1e-9,
                                 T.excess_vs_5050 / T.turnover_ann * 1e4, np.nan)

    act = design_oos(p, dv, [], 2)["y"].reindex(idx)
    dec = []
    for k in list(BM) + list(F):
        w = (BM[k] if k in BM else unconstrained_mv(F[k])).reindex(idx)
        m = w.notna() & act.notna()
        tot = (w[m] * act[m]).mean() * 12
        stat = w[m].mean() * act[m].mean() * 12
        dec.append(dict(strategy=k, mean_w=w[m].mean(), total=tot, static_tilt=stat,
                        timing_cov=tot - stat,
                        timing_share=(tot - stat) / tot if abs(tot) > 1e-12 else np.nan))

    return T, pd.DataFrame(dec).set_index("strategy"), R, idx


# ==============================================================================
# SECTION 8. ROBUSTNESS, BOOTSTRAP, SPA  -> Thesis 4.10 - 4.11
# ==============================================================================

def robustness_variant(p, label, dv="y_tot", credit=None, equity=None,
                       start=None, drop=None, extra=None, hy="ret_hy", ig="ret_ig"):
    credit = credit or CREDIT_SPREAD_DV
    equity = equity or EQUITY_BLOCK
    r6, _ = fit_is(p, dv, credit + equity + (extra or []), 2, start=start, drop=drop)
    _, pe = block_wald(r6, equity)
    _, pc = block_wald(r6, credit)
    pm = prevailing_mean(p, dv, start, drop)
    f2, a2 = recursive(p, dv, [], 2, start=start, drop=drop)
    f5, _ = recursive(p, dv, equity, start=start, drop=drop)
    f3, _ = recursive(p, dv, credit, restrict=True, truncate=True, start=start, drop=drop)
    fs, _ = recursive(p, dv, [credit[0]], start=start, drop=drop)
    j = a2.index.intersection(pm.index)
    if drop:
        for a_, b_ in drop:
            j = j[~((j >= a_) & (j <= b_))]
    a, B0, B2 = a2.reindex(j).values, pm.reindex(j).values, f2.reindex(j).values
    sig = p[dv].rolling(VOL_WINDOW).std().shift(1).reindex(j)
    out = dict(variant=label, n_is=int(r6.nobs), n_oos=len(j),
               IS_equity_p=pe, IS_credit_p=pc,
               OOS_equity_R2=r2os(a, f5.reindex(j).values, B2),
               OOS_credit_R2=r2os(a, f3.reindex(j).values, B2),
               OOS_equity_R2_vs_B0=r2os(a, f5.reindex(j).values, B0),
               OOS_credit_R2_vs_B0=r2os(a, f3.reindex(j).values, B0))
    for nm, f in [("SR_equity", f5), ("SR_credit", f3), ("SR_spread", fs)]:
        w = (f.reindex(j) / (GAMMA_WEIGHT * sig ** 2)).clip(0, 1)
        out[nm] = evaluate_portfolio(p, j, w, nm, hy, ig)[0]["sharpe"]
    out["SR_5050"] = evaluate_portfolio(p, j, pd.Series(.5, index=j), "b", hy, ig)[0]["sharpe"]
    return out


def run_robustness(p):
    rows = [robustness_variant(p, "Baseline"),
            robustness_variant(p, "R1 sample from 1997", start="1997-01-31"),
            robustness_variant(p, "R2 unsmoothed DV", dv="dv1_unsm"),
            robustness_variant(p, "R3 lagged equity control", extra=LAGGED_EQUITY),
            robustness_variant(p, "R5 oas_hy for oas_diff", credit=["oas_hy", "d3_oas_hy"]),
            robustness_variant(p, "R6 EBP credit baseline", credit=CREDIT_ALT),
            robustness_variant(p, "R7 duration-hedged DV", dv="y_xr", hy="xr_hy", ig="xr_ig"),
            robustness_variant(p, "R8 crisis excluded", drop=CRISIS_WINDOWS),
            robustness_variant(p, "R9 controls restored",
                               extra=["term_spread", "nfci", "ln_epu", "move"])]
    return pd.DataFrame(rows).set_index("variant")


def run_measurement_sensitivity(p, robustness, duration):
    keep = {"Baseline": "Baseline",
            "R7 duration-hedged DV": "R7 duration-hedged dependent variable",
            "R2 unsmoothed DV": "R2 unsmoothed dependent variable"}
    out = robustness.loc[[k for k in keep if k in robustness.index]].rename(index=keep)
    out = out[["IS_credit_p", "IS_equity_p", "OOS_credit_R2", "OOS_equity_R2",
               "SR_credit", "SR_equity", "SR_5050"]]
    out.attrs["rates_variance_share"] = float(duration.loc["HY - IG", "R2"])
    out.attrs["rates_variance_share_hedged"] = float(duration.loc["HY - IG hedged", "R2"])
    return out


def run_bootstrap_spa(p, R, idx):
    rf_j = p["rf"].reindex(idx)

    def sharpe(x):
        e = x - rf_j.reindex(x.index)
        return (e.mean() * 12) / (x.std() * np.sqrt(12))

    base = R["50/50 static"]
    names = [k for k in R if k != "50/50 static"]

    rows = []
    for k in names:
        j = R[k].index.intersection(base.index)
        M = pd.DataFrame({"s": R[k].loc[j], "b": base.loc[j], "f": rf_j.loc[j]})
        d0 = sharpe(R[k].loc[j]) - sharpe(base.loc[j])
        diffs = []
        for (dat,), _ in StationaryBootstrap(BLOCK_LENGTH, M,
                                             seed=SEED_BOOTSTRAP).bootstrap(BOOTSTRAP_REPS):
            e1, e2 = dat["s"] - dat["f"], dat["b"] - dat["f"]
            diffs.append((e1.mean() * 12) / (dat["s"].std() * np.sqrt(12))
                         - (e2.mean() * 12) / (dat["b"].std() * np.sqrt(12)))
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append(dict(strategy=k, sharpe=sharpe(R[k].loc[j]), diff_vs_5050=d0,
                         boot_p=2 * min((diffs <= 0).mean(), (diffs >= 0).mean()),
                         ci_low=lo, ci_high=hi))
    boot = pd.DataFrame(rows).set_index("strategy")

    j = base.index
    for k in names:
        j = j.intersection(R[k].index)
    bench_loss = -(base.loc[j] - rf_j.loc[j]).values
    model_loss = np.column_stack([-(R[k].loc[j] - rf_j.loc[j]).values for k in names])
    spa = SPA(bench_loss, model_loss, block_size=BLOCK_LENGTH,
              reps=BOOTSTRAP_REPS, seed=SEED_SPA)
    spa.compute()
    credit_set = [k for k in names if any(w in k for w in ("credit", "spread", "AR(2)"))]
    ml2 = np.column_stack([-(R[k].loc[j] - rf_j.loc[j]).values for k in credit_set])
    spa2 = SPA(bench_loss, ml2, block_size=BLOCK_LENGTH,
               reps=BOOTSTRAP_REPS, seed=SEED_SPA)
    spa2.compute()
    spa_out = pd.DataFrame([
        dict(set="all strategies", n=len(names), p_consistent=spa.pvalues["consistent"],
             p_lower=spa.pvalues["lower"], p_upper=spa.pvalues["upper"]),
        dict(set="credit subset", n=len(credit_set), p_consistent=spa2.pvalues["consistent"],
             p_lower=spa2.pvalues["lower"], p_upper=spa2.pvalues["upper"])]).set_index("set")
    return boot, spa_out


def subperiod_sharpe(p, R, idx, strategies):
    rf_j = p["rf"].reindex(idx)
    out = {}
    for k in strategies:
        r = {}
        for lab, (s, e) in SUBPERIODS.items():
            x = R[k].loc[s:e]
            f = rf_j.reindex(x.index)
            r[lab] = ((x - f).mean() * 12) / (x.std() * np.sqrt(12))
        out[k] = r
    return pd.DataFrame(out).T


# ==============================================================================
# SECTION 9. TACTICAL MANDATE  -> Thesis 4.6 - 4.9
# ==============================================================================

def taa_hdr(t):
    print(f"\n{'=' * 112}\n{t}\n{'=' * 112}")


def taa_design(p, dv, regs, arl=2):
    d = pd.DataFrame({"y": p[dv].shift(-1)})
    for k in range(1, arl + 1):
        d[f"ar{k}"] = p[dv].shift(k - 1)
    for r in regs:
        d[r] = p[r]
    d = d.dropna()
    d.index = d.index + pd.offsets.MonthEnd(1)
    return d


def taa_recursive(p, dv, regs, start, arl=2, min_train=MIN_TRAIN_OOS):
    d = taa_design(p, dv, regs, arl)
    cols = [c for c in d.columns if c != "y"]
    out = {}
    for t in d.index[d.index >= start]:
        tr = d.loc[d.index < t]
        if len(tr) < min_train:
            continue
        b = sm.OLS(tr["y"], sm.add_constant(tr[cols])).fit().params
        out[t] = float(np.r_[1.0, d.loc[t, cols].values.astype(float)] @ b.values)
    return pd.Series(out)


def taa_forecasts_for(p, regs, start):
    return taa_recursive(p, "y_tot", regs, start)


def taa_combination_forecast(p, signals, start):
    fs = [taa_recursive(p, "y_tot", [s], start) for s in signals]
    return pd.concat(fs, axis=1).mean(axis=1)


def taa_calibrate_kappa(p, sigma):
    rows, kappa = [], {}
    for name, regs in INFO_SETS.items():
        f = taa_forecasts_for(p, regs, CAL_START_TS).loc[CAL_START_TS:CAL_END_TS]
        s = (f / sigma.reindex(f.index)).dropna()
        k = TARGET_SIGMA_W / s.std()
        kappa[name] = k
        rows.append(dict(info_set=name, n_pseudo=len(s), sd_s=s.std(), kappa=k,
                         implied_sd_w=k * s.std()))
    f = taa_combination_forecast(p, COMBINATION_SIGNALS, CAL_START_TS).loc[CAL_START_TS:CAL_END_TS]
    s = (f / sigma.reindex(f.index)).dropna()
    kappa["I5 combination"] = TARGET_SIGMA_W / s.std()
    rows.append(dict(info_set="I5 combination", n_pseudo=len(s), sd_s=s.std(),
                     kappa=kappa["I5 combination"], implied_sd_w=TARGET_SIGMA_W))
    return kappa, pd.DataFrame(rows).set_index("info_set")


def taa_tilt(score, q=BAND_Q, kappa=1.0):
    return (kappa * score).clip(-q, q)


def taa_weights(score, w_strat, q=BAND_Q, kappa=1.0):
    return (w_strat + taa_tilt(score, q, kappa)).clip(0.0, 1.0)


def taa_evaluate(p, w, idx, w_strat, label):
    w = w.reindex(idx)
    rh, ri, rf = p["ret_hy"].reindex(idx), p["ret_ig"].reindex(idx), p["rf"].reindex(idx)
    ok = w.notna() & rh.notna() & ri.notna()
    w, rh, ri, rf = w[ok], rh[ok], ri[ok], rf[ok]

    Rp = w * rh + (1 - w) * ri
    Rb = w_strat * rh + (1 - w_strat) * ri
    active = Rp - Rb
    ex = Rp - rf

    wd = (w.shift(1) * (1 + rh)) / (w.shift(1) * (1 + rh) + (1 - w.shift(1)) * (1 + ri))
    to = (w - wd).abs().dropna()
    dn = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    wealth = (1 + Rp).cumprod()
    te = active.std() * np.sqrt(12)

    d = dict(strategy=label,
             w_mean=w.mean(), w_min=w.min(), w_max=w.max(), w_sd=w.std(),
             pct_at_band=float(((w - w_strat).abs() >= BAND_Q - 1e-9).mean()),
             IR=(active.mean() * 12) / te if te > 0 else np.nan,
             TE_realised=te, active_ret=active.mean() * 12,
             ret_ann=Rp.mean() * 12, vol_ann=Rp.std() * np.sqrt(12),
             sharpe=(ex.mean() * 12) / (Rp.std() * np.sqrt(12)),
             sortino=(ex.mean() * 12) / (dn * np.sqrt(12)) if dn > 0 else np.nan,
             maxDD=float((1 - wealth / wealth.cummax()).max()),
             turnover_ann=to.mean() * 12)
    for g in GAMMAS_CE:
        d[f"CE_g{int(g)}"] = (Rp.mean() - .5 * g * Rp.var()) * 12
    d["CE_CRRA_g5"] = (((1 + Rp) ** -4.0).mean() ** -0.25) ** 12 - 1
    d["breakeven_bp"] = (d["active_ret"] / d["turnover_ann"] * 1e4
                         if d["turnover_ann"] > 1e-9 else np.nan)
    return d, Rp, active


def run_tactical(p):
    taa_hdr("PRE-REGISTERED ALLOCATION PARAMETERS")
    print(f"  strategic weights   {1 - W_HY_STRATEGIC:.0%} IG / {W_HY_STRATEGIC:.0%} HY")
    print(f"  mandate band        +/- {BAND_Q:.0%}")
    print(f"  target sd(tilt)     {TARGET_SIGMA_W:.2f}   -> band/vol = {BAND_Q / TARGET_SIGMA_W:.1f}")
    print(f"  kappa calibration   {CAL_START_TS.date()} to {CAL_END_TS.date()} (pseudo-out-of-sample)")
    print(f"  evaluation          from {OOS_START_TS.date()}")

    sigma = p["y_tot"].rolling(VOL_WINDOW).std().shift(1)

    taa_hdr("TABLE B1  CALIBRATION OF KAPPA - pseudo-out-of-sample, all inputs pre-2010")
    kappa, cal = taa_calibrate_kappa(p, sigma)
    print(cal.round(4).to_string())
    save(cal, "b1_kappa_calibration")

    F = {n: taa_forecasts_for(p, r, OOS_START_TS) for n, r in INFO_SETS.items()}
    F["I5 combination"] = taa_combination_forecast(p, COMBINATION_SIGNALS, OOS_START_TS)
    idx = F["I2 credit"].index
    for k in F:
        F[k] = F[k].reindex(idx)
    sig = sigma.reindex(idx)
    S = {k: F[k] / sig for k in F}
    y = taa_design(p, "y_tot", [], 2)["y"].reindex(idx)
    print(f"\nEvaluation sample: {idx.min().date()} to {idx.max().date()}  n={len(idx)}")

    taa_hdr("TABLE B2  PASSIVE CONTROLS")
    rows, R, A = [], {}, {}
    vh = p["ret_hy"].rolling(VOL_WINDOW).std().shift(1).reindex(idx)
    vi = p["ret_ig"].rolling(VOL_WINDOW).std().shift(1).reindex(idx)
    P = {"P1 strategic 80/20": pd.Series(W_HY_STRATEGIC, index=idx),
         "P2 equal weight":    pd.Series(0.50, index=idx),
         "P3 inverse vol":     ((1 / vh) / ((1 / vh) + (1 / vi))).clip(0, 1)}
    for k, w in P.items():
        d, Rp, ac = taa_evaluate(p, w, idx, W_HY_STRATEGIC, k)
        rows.append(d); R[k] = Rp; A[k] = ac
    B2 = pd.DataFrame(rows).set_index("strategy")
    print(B2[["w_mean", "IR", "TE_realised", "sharpe", "sortino", "maxDD",
              "CE_g5", "turnover_ann"]].round(4).to_string())
    save(B2, "b2_passive")

    taa_hdr("TABLE B3  TACTICAL STRATEGIES - primary comparison")
    rows = []
    for k in ["I1 persistence", "I2 credit", "I3 equity", "I4 encompassing", "I5 combination"]:
        w = taa_weights(S[k], W_HY_STRATEGIC, kappa=kappa[k])
        d, Rp, ac = taa_evaluate(p, w, idx, W_HY_STRATEGIC, k)
        rows.append(d); R[k] = Rp; A[k] = ac
    B3 = pd.DataFrame(rows).set_index("strategy")
    B3["CE_gain_bp"] = (B3["CE_g5"] - float(B2.loc["P1 strategic 80/20", "CE_g5"])) * 1e4
    print("PRIMARY METRIC: information ratio against P1\n")
    print(B3[["w_mean", "w_min", "w_max", "w_sd", "pct_at_band", "IR", "TE_realised",
              "active_ret", "CE_gain_bp"]].round(4).to_string())
    print("\nSecondary and continuity metrics:")
    print(B3[["sharpe", "sortino", "maxDD", "CE_g5", "turnover_ann",
              "breakeven_bp"]].round(4).to_string())
    save(B3, "b3_tactical")

    taa_hdr("TABLE B4  STATIC TILT VERSUS TIMING")
    rows = []
    for k in B3.index:
        w = taa_weights(S[k], W_HY_STRATEGIC, kappa=kappa[k]).reindex(idx)
        m = w.notna() & y.notna()
        d = w[m] - W_HY_STRATEGIC
        tot = (d * y[m]).mean() * 12
        st = d.mean() * y[m].mean() * 12
        rows.append(dict(strategy=k, mean_tilt=d.mean(), total=tot, static=st,
                         timing=tot - st,
                         timing_share=(tot - st) / tot if abs(tot) > 1e-12 else np.nan))
    B4 = pd.DataFrame(rows).set_index("strategy")
    print(B4.round(4).to_string())
    print("\nActive return = tilt * (R_HY - R_IG) exactly, so only the covariance")
    print("term is tactical skill and the decomposition leaves no residual.")
    save(B4, "b4_decomposition")

    taa_hdr("TABLE B5  FRAMEWORK ROBUSTNESS")
    rows = []
    for q in (0.10, 0.20, 0.30):
        for k in ["I2 credit", "I3 equity"]:
            w = taa_weights(S[k], W_HY_STRATEGIC, q=q, kappa=kappa[k])
            d, _, _ = taa_evaluate(p, w, idx, W_HY_STRATEGIC, f"R-A q={q:.0%} | {k}")
            rows.append(d)
    for sw, style in ((0.05, "conservative"), (0.10, "base"),
                      (0.20, "active"), (0.30, "opportunistic")):
        for k in ["I2 credit", "I3 equity", "I4 encompassing"]:
            kk = kappa[k] * sw / TARGET_SIGMA_W
            w = taa_weights(S[k], W_HY_STRATEGIC, kappa=kk)
            d, _, _ = taa_evaluate(p, w, idx, W_HY_STRATEGIC,
                                   f"R-B sd_w={sw:.2f} ({style}) | {k}")
            rows.append(d)
    for sw, q in ((0.20, 0.40), (0.30, 0.60)):
        for k in ["I2 credit", "I3 equity"]:
            kk = kappa[k] * sw / TARGET_SIGMA_W
            w = taa_weights(S[k], W_HY_STRATEGIC, q=q, kappa=kk)
            d, _, _ = taa_evaluate(p, w, idx, W_HY_STRATEGIC,
                                   f"R-B sd_w={sw:.2f} q={q:.0%} | {k}")
            rows.append(d)
    kc = np.mean([kappa[k] for k in INFO_SETS])
    for k in ["I2 credit", "I3 equity"]:
        w = taa_weights(S[k], W_HY_STRATEGIC, kappa=kc)
        d, _, _ = taa_evaluate(p, w, idx, W_HY_STRATEGIC, f"R-C common kappa | {k}")
        rows.append(d)
    for k in ["I2 credit", "I3 equity"]:
        pct = S[k].expanding(min_periods=24).apply(lambda x: (x.iloc[-1] >= x).mean())
        w = (W_HY_STRATEGIC + BAND_Q * (2 * pct - 1)).clip(0, 1)
        d, _, _ = taa_evaluate(p, w, idx, W_HY_STRATEGIC, f"R-D rank map | {k}")
        rows.append(d)
    for k in ["I2 credit", "I3 equity"]:
        w = taa_weights(S[k], W_HY_ANCHOR_ALT, kappa=kappa[k])
        d, _, _ = taa_evaluate(p, w, idx, W_HY_ANCHOR_ALT, f"R-E anchor 50/50 | {k}")
        rows.append(d)
    for k in ["I2 credit", "I3 equity"]:
        kt = (TARGET_SIGMA_W / S[k].expanding(min_periods=24).std()).ffill()
        w = (W_HY_STRATEGIC + (kt * S[k]).clip(-BAND_Q, BAND_Q)).clip(0, 1)
        d, _, _ = taa_evaluate(p, w, idx, W_HY_STRATEGIC, f"R-F recursive kappa | {k}")
        rows.append(d)
    B5 = pd.DataFrame(rows).set_index("strategy")
    print(B5[["w_sd", "pct_at_band", "IR", "TE_realised", "sharpe",
              "turnover_ann"]].round(4).to_string())
    save(B5, "b5_robustness")

    taa_hdr("TABLE B6  STATIONARY BOOTSTRAP ON INFORMATION RATIO DIFFERENCES")
    print(f"H0: strategy IR against P1 equals zero. Block {BLOCK_LENGTH}, {BOOTSTRAP_REPS} reps.\n")
    rows = []
    for k in B3.index:
        M = pd.DataFrame({"a": A[k].dropna()})
        vals = []
        for (dat,), _ in StationaryBootstrap(BLOCK_LENGTH, M,
                                             seed=SEED_BOOTSTRAP).bootstrap(BOOTSTRAP_REPS):
            x = dat["a"]
            vals.append((x.mean() * 12) / (x.std() * np.sqrt(12)))
        vals = np.array(vals)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        rows.append(dict(strategy=k, IR=float(B3.loc[k, "IR"]),
                         boot_p=2 * min((vals <= 0).mean(), (vals >= 0).mean()),
                         ci_low=lo, ci_high=hi))
    B6 = pd.DataFrame(rows).set_index("strategy")
    print(B6.round(4).to_string())
    save(B6, "b6_bootstrap")

    taa_hdr("TABLE B7  HANSEN SPA ON ACTIVE RETURNS")
    names = list(B3.index)
    j = idx
    for k in names:
        j = j.intersection(A[k].dropna().index)
    bench_loss = np.zeros(len(j))          # the strategic portfolio has zero active return
    model_loss = np.column_stack([-A[k].loc[j].values for k in names])
    spa = SPA(bench_loss, model_loss, block_size=BLOCK_LENGTH,
              reps=BOOTSTRAP_REPS, seed=SEED_SPA)
    spa.compute()
    best = max(names, key=lambda n: B3.loc[n, "IR"])
    print(f"  strategies in set     {len(names)}")
    print(f"  SPA consistent p      {spa.pvalues['consistent']:.4f}")
    print(f"  SPA lower / upper     {spa.pvalues['lower']:.4f} / {spa.pvalues['upper']:.4f}")
    print(f"  best by IR            {best}  (IR {B3.loc[best, 'IR']:.3f})")
    print(f"\n  -> {'REJECT H0' if spa.pvalues['consistent'] < 0.05 else 'FAIL TO REJECT'}: "
          f"{'at least one strategy genuinely outperforms' if spa.pvalues['consistent'] < 0.05 else 'no strategy survives the data-snooping correction'}")
    save(pd.DataFrame([dict(n=len(names), p_consistent=spa.pvalues["consistent"],
                            p_lower=spa.pvalues["lower"], p_upper=spa.pvalues["upper"],
                            best=best)]), "b7_spa")

    taa_hdr("TABLE B8  SUBPERIOD INFORMATION RATIOS")
    out = {}
    for k in B3.index:
        r = {}
        for lab, (s, e) in SUBPERIODS.items():
            a = A[k].loc[s:e].dropna()
            r[lab] = (a.mean() * 12) / (a.std() * np.sqrt(12)) if a.std() > 0 else np.nan
        out[k] = r
    B8 = pd.DataFrame(out).T
    print(B8.round(3).to_string())
    save(B8, "b8_subperiods")

    tilt_path = taa_tilt(S["I2 credit"], kappa=kappa["I2 credit"])
    return dict(tilt=tilt_path, boot=B6, dec=B4, sub=B8, robustness=B5)


# ==============================================================================
# SECTION 10. FIGURES
# ==============================================================================

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.1,
})
K = ["#1a1a1a", "#6e6e6e", "#a8a8a8", "#404040", "#c8c8c8"]


def _save_figure(fig, stem: str) -> None:
    """Write a publication-quality PDF and a README-friendly PNG."""
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png")
    plt.close(fig)


def _shade_recessions(ax, p):
    if "usrec" not in p.columns:
        return
    r = p["usrec"].fillna(0)
    inside, start = False, None
    for d, v in r.items():
        if v > 0 and not inside:
            inside, start = True, d
        elif v == 0 and inside:
            ax.axvspan(start, d, color="0.85", zorder=0, lw=0)
            inside = False
    if inside:
        ax.axvspan(start, r.index[-1], color="0.85", zorder=0, lw=0)


def fig_3_1(p):
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    _shade_recessions(ax, p)
    ax.plot(p.index, p["oas_hy"], color=K[0], label="High yield")
    ax.plot(p.index, p["oas_ig"], color=K[1], label="Investment grade")
    ax.set_ylabel("Option-adjusted spread (%)")
    ax.set_title("Figure 3.1  Corporate bond index spreads, 1994-2026")
    ax.legend(loc="upper left")
    _save_figure(fig, "fig_3_1_spreads")


def fig_3_2(p):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
    a1.plot(p.index, np.exp(p["ln_vix"]), color=K[0], label="VIX")
    a1b = a1.twinx()
    a1b.plot(p.index, p["oas_hy"], color=K[2], label="HY spread (right)")
    a1b.set_ylabel("HY OAS (%)"); a1b.grid(False); a1b.spines["top"].set_visible(False)
    a1.set_ylabel("VIX")
    a1.set_title("Figure 3.2  Equity-implied signals and the high-yield spread")
    h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a1b.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, loc="upper left")

    a2.plot(p.index, p["skew"], color=K[0], label="CBOE SKEW")
    a2.axvline(pd.Timestamp("2003-09-30"), color=K[3], ls="--", lw=0.9)
    for sl in (slice(None, "2003-09-30"), slice("2003-10-31", None)):
        s = p.loc[sl, "skew"].dropna()
        a2.hlines(s.mean(), s.index.min(), s.index.max(), color=K[3], lw=1.4, ls=":")
    a2.annotate("VIX methodology revision, Sept 2003",
                xy=(pd.Timestamp("2003-09-30"), p["skew"].max() * 0.98),
                xytext=(6, -2), textcoords="offset points", fontsize=7.5, color=K[3])
    a2.set_ylabel("SKEW index"); a2.legend(loc="upper left")
    _save_figure(fig, "fig_3_2_signals")


def fig_4_1(p):
    series = [("HY total return", "ret_hy"), ("IG total return", "ret_ig"),
              ("HY - IG", "y_tot"), ("Equity market", "ret_eq")]
    n = 6
    x = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    w = 0.2
    for i, (lab, c) in enumerate(series):
        a = acf(p[c].dropna(), nlags=n, fft=False)[1:]
        ax.bar(x + (i - 1.5) * w, a, width=w, color=K[i], label=lab,
               edgecolor="white", linewidth=0.4)
    ci = 1.96 / np.sqrt(p["y_tot"].dropna().shape[0])
    ax.axhline(ci, color=K[3], ls="--", lw=0.8)
    ax.axhline(-ci, color=K[3], ls="--", lw=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.annotate("95% band", xy=(6.35, ci), fontsize=7.5, color=K[3], va="center")
    ax.set_xticks(x); ax.set_xlabel("Lag (months)"); ax.set_ylabel("Autocorrelation")
    ax.set_title("Figure 4.1  Return autocorrelation: bond indices against a traded market")
    ax.legend(ncol=2, loc="upper right")
    _save_figure(fig, "fig_4_1_autocorrelation")


def fig_4_2(p):
    d = pd.DataFrame({"eq": p["ret_eq"], "tsy": -7.0 * p["d_y10"],
                      "ig": p["xr_ig"], "hy": p["xr_hy"]}).dropna()
    win = 36
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    _shade_recessions(ax, p)
    pairs = [("eq", "tsy", "Equity ~ Treasury", K[0], "-"),
             ("eq", "ig", "Equity ~ IG credit", K[1], "-"),
             ("eq", "hy", "Equity ~ HY credit", K[2], "-"),
             ("ig", "hy", "IG ~ HY credit", K[3], ":")]
    for a, b, lab, col, ls in pairs:
        ax.plot(d.index, d[a].rolling(win).corr(d[b]), color=col, ls=ls, label=lab)
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(pd.Timestamp("2022-01-31"), color=K[3], ls="--", lw=0.9)
    ax.annotate("2022", xy=(pd.Timestamp("2022-02-28"), -0.85), fontsize=7.5, color=K[3])
    ax.set_ylabel(f"Rolling {win}-month correlation")
    ax.set_title("Figure 4.2  What the 2022 episode did and did not change")
    ax.legend(ncol=2, loc="lower left")
    _save_figure(fig, "fig_4_2_correlations")


def fig_4_3(forecasts, actual, benchmark, label_bench="AR(2)"):
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for i, (lab, f) in enumerate(forecasts.items()):
        j = actual.index.intersection(f.index)
        cssed = (((actual[j] - benchmark[j]) ** 2) - ((actual[j] - f[j]) ** 2)).cumsum()
        ax.plot(cssed.index, cssed * 1e4, color=K[i % len(K)], label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(f"Cumulative SSE difference vs {label_bench}\n(rising = model better)")
    ax.set_title("Figure 4.3  Out-of-sample forecast performance over time")
    ax.legend(loc="lower left")
    _save_figure(fig, "fig_4_3_cssed")


def fig_4_4(p, tilt, label="Credit and macro"):
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.plot(tilt.index, tilt * 100, color=K[0], lw=1.0,
            label=f"{label}: tactical tilt on HY")
    ax.axhline(0, color=K[3], lw=0.8)
    for b in (-BAND_Q * 100, BAND_Q * 100):
        ax.axhline(b, color=K[3], ls=":", lw=0.9)
    ax.annotate("mandate band", xy=(tilt.index[3], BAND_Q * 100 + 0.6),
                fontsize=7.5, color=K[3])
    ax.set_ylabel("Deviation from strategic weight (pp)")
    ax.set_ylim(-BAND_Q * 100 - 4, BAND_Q * 100 + 4)
    ax2 = ax.twinx()
    ax2.plot(tilt.index, p.loc[tilt.index, "oas_hy"], color=K[2], lw=1.0,
             label="HY spread (right, inverted)")
    ax2.set_ylabel("HY OAS (%)"); ax2.grid(False)
    ax2.spines["top"].set_visible(False); ax2.invert_yaxis()
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left")
    ax.set_title("Figure 4.4  Tactical tilt against the credit cycle")
    _save_figure(fig, "fig_4_4_tilt_path")


def fig_4_5(boot):
    b = boot.sort_values("IR")
    y = np.arange(len(b))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.hlines(y, b["ci_low"], b["ci_high"], color=K[2], lw=3.2, alpha=.85)
    ax.plot(b["IR"], y, "o", color=K[0], ms=5)
    ax.axvline(0, color="k", lw=.9)
    ax.set_yticks(y); ax.set_yticklabels(b.index, fontsize=8)
    ax.set_xlabel("Information ratio against the strategic portfolio "
                  "(95% bootstrap interval)")
    ax.set_title("Figure 4.5  No information set is distinguishable from the benchmark")
    ax.grid(axis="y", alpha=0)
    _save_figure(fig, "fig_4_5_information_ratios")


def fig_4_6(rb):
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    te = rb.index.values
    for i, c in enumerate(rb.columns):
        ax.plot(te, rb[c], marker="o", ms=5, color=K[i % len(K)],
                ls="-" if "Credit" in c else "--", label=c)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("Ex-ante tracking error budget (basis points)")
    ax.set_ylabel("Information ratio")
    ax.set_title("Figure 4.6  The comparison is invariant to mandate style")
    ax.legend(loc="upper left")
    _save_figure(fig, "fig_4_6_mandate_styles")


def fig_4_7(dec):
    d = dec.sort_values("timing")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.barh(y, d["static"] * 1e4, color=K[2], height=.6, label="Static tilt")
    ax.barh(y, d["timing"] * 1e4, left=d["static"] * 1e4, color=K[0],
            height=.6, label="Timing covariance")
    ax.axvline(0, color="k", lw=.9)
    ax.set_yticks(y); ax.set_yticklabels(d.index, fontsize=8)
    ax.set_xlabel("Contribution to annualised active return (basis points)")
    ax.set_title("Figure 4.7  Only the covariance term is tactical skill")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0)
    _save_figure(fig, "fig_4_7_attribution")


def fig_4_8(sub):
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    x = np.arange(len(sub.columns))
    for i, s in enumerate(sub.index):
        ax.plot(x, sub.loc[s], marker="o", ms=5, color=K[i % len(K)],
                ls="--" if "persistence" in s else "-", label=s)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(sub.columns)
    ax.set_ylabel("Information ratio")
    ax.set_title("Figure 4.8  Subperiod information ratios")
    ax.legend(ncol=2, loc="lower left", fontsize=7.5)
    _save_figure(fig, "fig_4_8_subperiods")


def fig_4_9(oos):
    d = oos[oos.dv.str.startswith("DV1")].set_index("model")
    show = [m for m in ["B0 prevailing mean", "B2 AR(2)", "M3 credit (CT)",
                        "M5 equity", "M6 encompassing"] if m in d.index]
    d = d.loc[show]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    h = 0.38
    ax.barh(y + h / 2, d["R2os_vs_B0"], height=h, color=K[0], label="vs prevailing mean")
    ax.barh(y - h / 2, d["R2os_vs_B2"], height=h, color=K[2], label="vs AR(2) benchmark")
    ax.axvline(0, color="k", lw=.9)
    ax.set_yticks(y); ax.set_yticklabels(d.index, fontsize=8)
    ax.set_xlabel("Out-of-sample $R^2$")
    ax.set_title("Figure 4.9  Out-of-sample performance against both benchmarks")
    ax.legend(loc="lower left"); ax.grid(axis="y", alpha=0)
    _save_figure(fig, "fig_4_9_both_benchmarks")


def make_figures(p, oos, oos_bits, taa_bits):
    fig_3_1(p); fig_3_2(p); fig_4_1(p); fig_4_2(p)
    fig_4_3(*oos_bits)
    fig_4_4(p, taa_bits["tilt"])
    fig_4_5(taa_bits["boot"])
    fig_4_6(taa_bits["mandate_styles"])
    fig_4_7(taa_bits["dec"])
    fig_4_8(taa_bits["sub"])
    fig_4_9(oos)
    print(f"\nFigures written to {FIG.resolve()}")
    for f in sorted(FIG.glob("*.pdf")):
        print(f"   {f.name}")


def mandate_style_table(B5):
    rows = {}
    for sw, te in ((0.05, 26), (0.10, 51), (0.20, 103), (0.30, 154)):
        r = {}
        for k, lab in (("I2 credit", "Credit and macro"),
                       ("I3 equity", "Equity-implied"),
                       ("I4 encompassing", "Encompassing")):
            m = [i for i in B5.index if f"sd_w={sw:.2f}" in i and "q=" not in i and k in i]
            if m:
                r[lab] = B5.loc[m[0], "IR"]
        rows[te] = r
    return pd.DataFrame(rows).T


# ==============================================================================
# SECTION 11. DRIVER
# ==============================================================================

def run_forecasting_layer(p):
    ALLX = CREDIT_SPREAD_DV + ["oas_hy", "ebp", "term_spread", "nfci"] \
        + EQUITY_BLOCK + ["mom_12_1"]
    SERIES = [("HY total return", "ret_hy"), ("IG total return", "ret_ig"),
              ("HY - IG", "y_tot"), ("Equity market", "ret_eq")]

    head("TABLE 4.1  DESCRIPTIVE STATISTICS")
    print(save(diag_descriptives(p), "t41_descriptives").round(3).to_string())

    head("TABLE 3.2  DESCRIPTIVE STATISTICS, PREDICTORS")
    print(save(diag_predictor_descriptives(
        p, ["oas_diff", "oas_hy", "d3_oas_hy", "ebp", "ln_vix", "skew", "vrp",
            "term_spread", "nfci"]), "t32_predictors").round(3).to_string())

    head("TABLE 3.3  STABILITY OF THE PUBLISHED INDICES ACROSS SEPT 2003")
    print(save(diag_skew_break(p), "t33_skew_break").round(3).to_string())

    head("TABLE 4.2  STATIONARITY")
    print(save(diag_stationarity(p, ["y_tot"] + ALLX), "t42_stationarity").round(4).to_string())

    head("TABLE 4.3  PERSISTENCE AND EFFECTIVE SAMPLE SIZE")
    print(save(diag_persistence(p, ["y_tot"] + ALLX), "t43_persistence").round(3).to_string())

    head("TABLE 4.4  STAMBAUGH BIAS MAGNITUDE")
    print(save(diag_stambaugh(p, "y_tot", ALLX), "t44_stambaugh").round(4).to_string())

    head("TABLE 4.5  COLLINEARITY")
    v1, c1 = diag_collinearity(p, CREDIT_SPREAD_DV + EQUITY_BLOCK, "M6")
    v2, c2 = diag_collinearity(p, ["oas_hy", "ebp", "nfci", "term_spread", "d3_oas_hy"],
                               "full_credit")
    print(pd.concat([v1, v2], axis=1).round(2).to_string())
    print(f"condition number  M6 = {c1:.1f}   full credit block = {c2:.1f}")
    save(pd.concat([v1, v2], axis=1), "t45_collinearity")

    head("TABLE 4.6  RETURN SMOOTHING (Getmansky, Lo and Makarov 2004)")
    print(save(diag_smoothing(p, SERIES), "t46_smoothing").round(4).to_string())

    head("TABLE 4.7  VARIANCE RATIOS")
    print(save(diag_variance_ratio(p, SERIES), "t47_variance_ratios").round(3).to_string())

    head("TABLE 4.8  ADJUSTMENT LAG RELATIVE TO THE EQUITY MARKET")
    print(save(diag_leadlag(p), "t48_leadlag").round(4).to_string())

    head("TABLE 4.9  EMPIRICAL DURATION")
    duration = save(diag_duration(p), "t49_duration")
    print(duration.round(4).to_string())

    head("TABLE 4.10  BENCHMARK LAG ORDER")
    print(save(diag_benchmark_order(p), "t410_benchmark_order").round(4).to_string())

    head("TABLE 4.11  IN-SAMPLE MODEL LADDER")
    ladder, enc, store = run_insample(p)
    print(save(ladder.set_index(["dv", "model"]), "t411_ladder").round(4).to_string())

    head("TABLE 4.11b  SKEWNESS COEFFICIENT ACROSS THE SEPT 2003 BREAK")
    print(save(diag_skew_subsample(p), "t411b_skew_subsample").round(4).to_string())

    head("TABLE 4.12  ENCOMPASSING TESTS")
    print(save(enc.set_index(["dv", "model"]), "t412_encompassing").round(4).to_string())

    head("TABLE 4.13  COEFFICIENTS, M6 AND M7 ON DV1")
    for nm, regs in [("M6 B2+credit+equity", CREDIT_SPREAD_DV + EQUITY_BLOCK),
                     ("M7 B3+credit+equity", CREDIT_SPREAD_DV + EQUITY_BLOCK + LAGGED_EQUITY)]:
        r = store[("DV1 y_tot (HY-IG)", nm)]
        t = coef_table(r, HAC_SENSITIVITY, p, "y_tot", regs)
        print(f"\n--- {nm}  n={int(r.nobs)}  R2={r.rsquared:.4f} ---")
        print(t.round(4).to_string())
        save(t, f"t413_coefs_{nm.split()[0]}")

    head("TABLE 4.14  UNIVARIATE REGRESSIONS WITH HOLM CORRECTION")
    print(save(run_univariate(p), "t414_univariate").round(4).to_string())

    head("TABLE 4.15  BOOTSTRAP P-VALUES, PERSISTENT REGRESSORS")
    rows = []
    for x in ["oas_diff", "oas_hy", "nfci", "term_spread", "ebp"]:
        t_obs, pb = bootstrap_predictive_p(p, "y_tot", x)
        r, _ = fit_is(p, "y_tot", [x], 2)
        rows.append(dict(variable=x, t_HAC=t_obs, p_asymptotic=r.pvalues[x], p_bootstrap=pb))
    print(save(pd.DataFrame(rows).set_index("variable"), "t415_bootstrap_p").round(4).to_string())

    head("TABLE 4.16  OUT-OF-SAMPLE FORECAST EVALUATION, BOTH BENCHMARKS")
    oos, bench = run_oos(p)
    print(save(oos.set_index(["dv", "model"]), "t416_oos").round(4).to_string())

    head("TABLE 4.17  PORTFOLIO PERFORMANCE (mapping 1, unconstrained)")
    T, dec_fc, R, idx = run_portfolio(p, bench)
    print(T[["mean_w", "ret_ann", "vol_ann", "sharpe", "sortino", "maxDD"]].round(4).to_string())
    print()
    print(T[["CE_g3", "CE_g5", "CE_g10", "CE_CRRA_g5", "IR_vs_5050",
             "turnover_ann", "breakeven_bp"]].round(4).to_string())
    save(T, "t417_portfolio")

    head("TABLE 4.18  STATIC TILT VERSUS TIMING (mapping 1)")
    print(save(dec_fc, "t418_tilt_timing").round(4).to_string())

    head("TABLE 4.20  ROBUSTNESS R1 - R9")
    robustness = save(run_robustness(p), "t420_robustness")
    print(robustness.round(4).to_string())

    head("TABLE 4.20b  MEASUREMENT SENSITIVITY: DURATION AND SMOOTHING")
    ms = run_measurement_sensitivity(p, robustness, duration)
    print(ms.round(4).to_string())
    save(ms, "t420b_measurement_sensitivity")

    head("TABLE 4.21  STATIONARY BOOTSTRAP AND SPA (mapping 1)")
    boot_fc, spa_out = run_bootstrap_spa(p, R, idx)
    print(save(boot_fc, "t421_bootstrap").round(4).to_string())
    print()
    print(save(spa_out, "t421_spa").round(4).to_string())

    head("TABLE 4.22  SUBPERIOD SHARPE RATIOS (mapping 1)")
    sel = ["50/50 static", "inverse volatility", "B2 AR(2)", "M3 credit (CT)",
           "M5 equity", "M6 encompassing", "spread only", "combination credit"]
    print(save(subperiod_sharpe(p, R, idx, sel), "t422_subperiods").round(3).to_string())

    head("TABLE 3.4  FREE-DATA REPLICATION OF THE CREDIT BLOCK")
    moodys = diag_moodys_replication(p)
    print(save(moodys, "t34_moodys_replication").round(4).to_string())

    return oos, bench, R, idx


def run_all():
    print("Replication pipeline: Equity-Implied Signals in Corporate Bond Allocation")
    print(f"Project directory: {PROJECT_DIR}")

    run_stage("[1/8] Checking inputs", validate_environment)
    monthly, daily, french = run_stage("[2/8] Loading data", load_raw)
    p = run_stage("[3/8] Constructing variables", build_variables, monthly, daily, french)
    save(p, "panel")

    oos, bench, R, idx = run_stage(
        "[4/8] Running diagnostics, forecasts, portfolios, and robustness",
        run_forecasting_layer,
        p,
    )
    taa = run_stage("[5/8] Evaluating tactical mandate", run_tactical, p)

    print("[6/8] Rebuilding figure inputs")
    pm = prevailing_mean(p, "y_tot")
    f2, a2 = recursive(p, "y_tot", [], 2)
    j = pm.index.intersection(f2.index)
    forecast_paths = {
        "Credit (restricted)": recursive(p, "y_tot", CREDIT_SPREAD_DV,
                                         restrict=True, truncate=True)[0].reindex(j),
        "Equity block": recursive(p, "y_tot", EQUITY_BLOCK)[0].reindex(j),
        "Encompassing": recursive(p, "y_tot", CREDIT_SPREAD_DV + EQUITY_BLOCK)[0].reindex(j),
        "Prevailing mean": pm.reindex(j)}
    taa["mandate_styles"] = mandate_style_table(taa["robustness"])

    print("[7/8] Writing figures")
    if PRINT_TABLES:
        make_figures(p, oos, (forecast_paths, a2.loc[j], f2.loc[j], "AR(2)"), taa)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            make_figures(p, oos, (forecast_paths, a2.loc[j], f2.loc[j], "AR(2)"), taa)

    print("[8/8] Summarizing outputs")
    tables = len(list(OUT.glob("*.csv")))
    pdf_figures = len(list(FIG.glob("*.pdf")))
    png_figures = len(list(FIG.glob("*.png")))
    print("\nCompleted successfully.")
    print(f"Tables: {tables}")
    print(f"Figures: {pdf_figures} PDF, {png_figures} PNG")
    print(f"Output: {OUT.resolve()}")


if __name__ == "__main__":
    run_all()
